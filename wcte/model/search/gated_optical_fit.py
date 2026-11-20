"""Fit optical geometry to calibrated charge and the gated first pulse."""

import numpy as np

from analog_optical import endpoint
from gated_photon_time import evaluate
from joint_prompt_fit import Joint, wavefront
from scipy.special import softmax
from wctering.receiver.angular_panels import finite_integrate
from wctering.receiver.finite_source import quadrature


class PromptOptics(Joint):
    """Each path has a relative birth offset and the same measured gate."""

    def __init__(self, event, geometry, background, strength, extra_light=None, minimum_opening=0.0):
        super().__init__(event, geometry, background)
        if minimum_opening:
            for j in range(self.k):
                index = j * self.size + 7
                self.bounds[index] = minimum_opening, self.bounds[index][1]
                self.x[index] = max(minimum_opening, self.x[index])
            self.clock = self.start_clock(self.x)
        center = np.median(event.times_ns)
        self.gate = np.asarray(event.timing_selection["prompt_window_ns"]) - center
        self.sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
        self.clock_start = self.clock
        self.strength = strength
        self.extra_light = extra_light
        self.x = np.r_[self.x, self.clock_start, np.zeros(self.k - 1)]
        self.bounds = [
            *self.bounds,
            (self.clock_start - 5.0, self.clock_start + 5.0),
            *[(-4.0, 4.0)] * (self.k - 1),
        ]

    def evaluate(self, x):
        if self.last is not None and np.array_equal(x, self.last[0]):
            return self.last[1:]
        n = self.k * self.size
        split = len(x) - self.k
        response = [self.background]
        charge_gradient, flight, time_gradient = [], [], []
        for p in x[:n].reshape(-1, self.size):
            shape, jac = self.component(p)
            response.append(shape)
            charge_gradient.append(jac)
            mean, d = wavefront(self.positions, p)
            flight.append(mean)
            time_gradient.append(d)
        shapes = np.asarray(response)
        weights = softmax(np.r_[0.0, x[n : split - 1]])
        amplitude = weights * np.exp(x[split - 1])
        rates = amplitude[:, None] * shapes
        cone_mean = rates.sum(0)
        mean = cone_mean
        means = np.column_stack(flight) + x[split] + np.r_[0.0, x[split + 1 :]]
        observed = rates[:, self.ix].T
        if self.extra_light is not None:
            light, pulse = self.extra_light
            mean = cone_mean + light.sum(0) if light.ndim == 2 else cone_mean + light
            observed = np.column_stack([observed, np.atleast_2d(light)[:, self.ix].T])
            means = np.column_stack([means, pulse])
        ll, slope = self.obs.score(mean, self.ix)
        timing, d_rates, d_time = evaluate(
            self.time, *self.gate, means, observed, self.sigma, self.obs.offsets, self.obs.coefficients
        )
        if self.extra_light is not None:
            # Fixed sections contribute at their own arrival times.
            # The optical geometry derivatives only move the cone light.
            d_rates = d_rates[:, : self.k + 1]
            d_time = d_time[:, : self.k]
            observed = observed[:, : self.k + 1]
        jac = np.zeros((len(mean), len(x)))
        extra = np.zeros(len(x))
        for j, (ds, dt) in enumerate(zip(charge_gradient, time_gradient)):
            block = slice(self.size * j, self.size * (j + 1))
            jac[:, block] = amplitude[j + 1] * ds
            extra[block] = d_rates[:, j + 1] @ jac[self.ix, block] + d_time[:, j] @ dt
        jac[:, n : split - 1] = rates[1:].T - cone_mean[:, None] * weights[1:]
        jac[:, split - 1] = cone_mean
        contribution = (observed * d_rates).sum(0)
        extra[n : split - 1] = contribution[1:] - weights[1:] * contribution.sum()
        extra[split - 1] = contribution.sum()
        extra[split] = d_time.sum()
        extra[split + 1 :] = d_time[:, 1:].sum(0)
        # Use the timing information to scale the clock in the optimizer.
        # The pseudo-Jacobian has only those previously empty coordinates.
        metric = jac.copy()
        metric[self.ix, split] = np.sqrt(mean[self.ix]) / self.sigma
        for j in range(1, self.k):
            metric[self.ix, split + j] = observed[:, j + 1] / (np.sqrt(mean[self.ix]) * self.sigma)
        self.last = (
            x.copy(),
            ll + self.strength * float(timing.sum()),
            slope @ jac - self.strength * extra,
            (observed / mean[self.ix, None]).T,
            mean,
            metric,
        )
        return self.last[1:]


def fit(event, geometry, background, budget, strength=0.55, extra_light=None, *, minimum_opening=0.0):
    """Refit geometry, with optional fixed light rows and their absolute times."""
    if event.timing_selection["prompt_window_ns"] is None:
        return None
    try:
        model = PromptOptics(event, geometry, background, strength, extra_light, minimum_opening)
    except ValueError:
        return None
    x, calls = endpoint(model, budget)
    if x is None:
        return None
    split = len(x) - model.k
    state = model.evaluate(x)
    amplitude = np.exp(x[split - 1]) * softmax(np.r_[0.0, x[model.k * model.size : split - 1]])
    fine = [background]
    for p in x[: model.k * model.size].reshape(-1, model.size):
        source = finite_integrate(model.pos, model.normals, p, *quadrature(16))[0]
        if source.sum() <= 0.0:
            return None
        fine.append(source / source.sum())
    rates = amplitude[:, None] * np.asarray(fine)
    mean = rates.sum(0)
    if extra_light is not None:
        extra = extra_light[0]
        fixed = extra.sum(0) if extra.ndim == 2 else extra
        mean += fixed
    ll = model.obs.score(mean, model.ix)[0]
    error = abs(ll - model.obs.score(state[3], model.ix)[0])
    if error >= 0.01:
        return None
    shares = (rates[:, model.ix] / mean[model.ix]).T
    if extra_light is not None:
        shares[:, 0] += fixed[model.ix] / mean[model.ix]
    if set(shares.argmax(1)) - {0} != set(range(1, model.k + 1)):
        return None
    support = [float(np.sum(1.0 - model.obs.no_source(mean[model.ix], row))) for row in shares.T[1:]]
    if min(support) < 30.0:
        return None
    return (
        ll,
        shares,
        dict(
            geometry=model.geometry(x[:split]),
            photons=amplitude,
            relative_clock_ns=float(x[split]),
            timing_offsets_ns=x[split + 1 :].tolist(),
            joint_score=state[0],
            quadrature_ll_error=float(error),
            observed_support=support,
            evaluations=calls,
        ),
    )
