"""Jointly refine charged paths that can be separated by neutral flight."""

import numpy as np
from scipy.optimize import minimize

from balanced_search import feasible
from charge_starts import weighted_quantile
from gated_photon_time import evaluate
from independent_arcs import Independent, LocalCharts
from physical_pulse import wavefront
from recoil_competitor import render
from timed_track_fit import interval_flight
from track_coordinates import fit as optimize
from wctering.seeding.coherent_timing import C


class TimedIndependent(Independent):
    """Move independent paths under calibrated charge and relative times."""

    def __init__(self, event, model, particles, timing, clock):
        super().__init__(event.pmt_ids, event.charges_pe, model.bg, particles, 8)
        self.event, self.timing, self.clock = event, timing, clock
        self.timed_state = None

    def primary_light(self, x):
        parts, derivatives, core_derivatives = [], [], []
        for p, mass in zip(x.reshape(-1, 12), self.particles):
            core, total, jac, core_jac = render(self, p, mass, self.order, core_gradient=True)
            parts.append([core, total - core])
            derivatives.append(jac)
            core_derivatives.append(core_jac[self.ix])
        self.core_jacobian = np.asarray(core_derivatives)
        return np.asarray(parts), np.hstack(derivatives)

    def __call__(self, x, detail=False):
        old = self.timed_state
        if old is not None and np.array_equal(x, old["parameters"]):
            return old if detail else (old["objective"], old["gradient"])
        state = dict(super().__call__(x, True))
        _, light, core, derivative = self.primary_cache
        response, background = np.exp(x[-2:])
        ring = response * core[:, self.ix].T
        sky = background * self.bg[self.ix]
        other = response * light[self.ix] - ring.sum(1)
        rates = np.column_stack([sky + other, ring])
        arrivals, jac = [], []
        for p, mass in zip(x[:-2].reshape(-1, 12), self.particles):
            t, dt, _, _ = interval_flight(self.event, p, mass)
            arrivals.append(t)
            jac.append(dt)
        arrival = np.column_stack(arrivals) + self.clock
        timing = self.timing
        ll, d_rate, d_time = evaluate(
            timing.time,
            timing.low,
            timing.high,
            arrival,
            rates,
            timing.sigma,
            self.obs.offsets,
            self.obs.coefficients,
        )
        contrast = d_rate[:, 1:] - d_rate[:, :1]
        spatial = response * (
            d_rate[:, 0] @ derivative[self.ix] + np.einsum("ij,jin->jn", contrast, self.core_jacobian).ravel()
        )
        spatial += np.einsum("ij,jin->jn", d_time, jac).ravel()
        efficiency = (contrast * ring).sum() + d_rate[:, 0] @ (other + ring.sum(1))
        background_slope = d_rate[:, 0] @ sky
        # Use exact charge and a midpoint timing approximation during the
        # geometry search. The finalist below restores finite-track timing.
        state["objective"] -= 0.5 * ll.sum() / self.total
        state["gradient"] = (
            state["gradient"] - 0.5 * np.r_[spatial, efficiency, background_slope] / self.total
        )
        self.timed_state = state
        return state if detail else (state["objective"], state["gradient"])


class ApertureCharts(LocalCharts):
    """Constrain just the first of the independent tracks to the beam port."""

    def __init__(self, model, x, port):
        super().__init__(model, x)
        self.radius = port.radius
        self.bounds[2] = port.z, port.z

    def full(self, y):
        x, jac, margin, derivative = super().full(y)
        aperture = (self.radius**2 - x[:2] @ x[:2]) / self.radius
        slope = -(2.0 / self.radius) * x[:2] @ jac[:2]
        return x, jac, np.r_[margin, aperture], np.vstack([derivative, slope])


def profile(event, model, x, timing, *, fixed_delays=None):
    """Fit one clock and causal relative births at fixed track geometry."""
    model(x, True)
    _, light, core, _ = model.primary_cache
    paths = x[:-2].reshape(-1, 12)
    times, tails = [], []
    for path, mass in zip(paths, model.particles):
        arrival, intensity, _ = wavefront(event.pmt_positions_mm, model.normals[model.ix], path, mass)
        times.extend(arrival)
        tails.append(intensity[1])
    arrival = np.column_stack(times)
    tails = np.asarray(tails)
    direct = core[:, model.ix]
    remainder = np.maximum(light[model.ix] - direct.sum(0), 0.0)
    tails *= np.minimum(1.0, remainder / np.maximum(tails.sum(0), 1e-300))
    diffuse = np.maximum(remainder - tails.sum(0), 0.0)
    charged = np.stack([direct, tails], axis=1).reshape(-1, len(diffuse)).T
    sky = model.bg[model.ix]
    weight = np.minimum(event.charges_pe, 2.0) * direct
    if np.any(weight.sum(1) <= 0.0) and fixed_delays is None:
        return None
    if fixed_delays is None:
        clock = np.array(
            [weighted_quantile(timing.time - t, w, [0.5])[0] for t, w in zip(arrival[:, ::2].T, weight)]
        )
    else:
        all_light = (charged[:, ::2] + charged[:, 1::2]).T
        illuminated = np.flatnonzero(all_light.sum(1) > 0.0)
        flight = (arrival[:, ::2] + np.r_[0.0, fixed_delays])[..., illuminated]
        weights = np.minimum(event.charges_pe, 2.0) * all_light[illuminated]
        clock = np.array(
            [weighted_quantile((timing.time[:, None] - flight).ravel(), weights.T.ravel(), [0.5])[0]]
        )
    # Even an unseen carrier must reach the secondary at or below c.
    # The entrance birth gives a necessary lower bound independent of
    # whether the carrier was emitted before or after the primary stopped.
    minimum_delay = np.linalg.norm(paths[1:, :3] - paths[0, :3], axis=1) * 1000.0 / C
    if fixed_delays is None:
        latest = minimum_delay + timing.high - timing.low
        initial_delay = np.clip(clock[1:] - clock[0], minimum_delay, latest)
        delay_bounds = list(zip(minimum_delay, latest))
    else:
        if len(fixed_delays) != len(minimum_delay) or np.any(fixed_delays < minimum_delay - 1e-8):
            return None
        initial_delay = fixed_delays
        delay_bounds = list(zip(fixed_delays, fixed_delays))
    initial = np.r_[x[-2:], clock[0], initial_delay]
    bounds = [*model.bounds[-2:], (clock[0] - 4.0, clock[0] + 4.0), *delay_bounds]

    def objective(z, detail=False):
        response, background = np.exp(z[:2])
        rates = np.column_stack([background * sky + response * diffuse, response * charged])
        offsets = z[2] + np.r_[0.0, z[3:]]
        centers = arrival + np.repeat(offsets, 2)
        mean = response * light + background * model.bg
        if detail:
            shares = (response * direct / mean[model.ix]).T
            return mean, rates, centers, shares, offsets, z[:2]
        ll, residual = model.obs.score(mean, model.ix)
        lt, d_rate, d_time = evaluate(
            timing.time,
            timing.low,
            timing.high,
            centers,
            rates,
            timing.sigma,
            model.obs.offsets,
            model.obs.coefficients,
        )
        d_signal = response * np.column_stack([diffuse, charged])
        d_clock = d_time.sum(0).reshape(-1, 2).sum(1)
        slope = np.r_[
            response * (residual @ light) - np.sum(d_rate * d_signal),
            background * (residual @ model.bg - d_rate[:, 0] @ sky),
            -d_clock.sum(),
            -d_clock[1:],
        ]
        return -(ll + lt.sum()) / model.total, slope / model.total

    fit = minimize(
        objective, initial, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=24, ftol=1e-9)
    )
    z = fit.x if np.isfinite(fit.fun) and fit.fun < objective(initial)[0] else initial
    return objective(z, detail=True)


def refine(event, model, paths, particles, normalization, timing, clock, port, budget):
    """Refit independent tracks and check the charge and timing integrals."""
    x = np.r_[np.asarray(paths).ravel(), normalization]
    trial = TimedIndependent(event, model, particles, timing, clock)
    chart = ApertureCharts(trial, x, port)
    result, x = optimize(trial, x, budget, budget, full_likelihood=True, chart=chart, check_initial=True)
    if not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5 or x[:2] @ x[:2] > port.radius**2 + 1e-9:
        return None
    args = event.pmt_ids, event.charges_pe, model.bg, particles
    coarse = Independent(*args, 16)(x, True)
    fine = Independent(*args, 32)
    error = abs(fine(x, True)["ll"] - coarse["ll"])
    if error >= 0.01:
        return None
    fitted = profile(event, fine, x, timing)
    if fitted is None:
        return None
    mean, rates, centers, shares, offsets, norm = fitted
    detail = dict(
        materials=particles,
        parameters=np.r_[x[:-2], norm].tolist(),
        clock_ns=offsets.tolist(),
        evaluations=int(result.nfev),
        quadrature_ll_error=float(error),
    )
    return mean, rates, centers, shares, detail
