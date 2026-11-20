"""Fit a small optical bank with C's calibrated hit and unhit likelihood."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from readout import observation
from wctering.likelihood.crossing import intervals
from wctering.likelihood.water import WaterOccupancyModel
from wctering.serialization import copy_geometry


def contain_finite(parameters):
    """Clip a finite seed to the nearest connected water interval."""
    p = parameters
    water = intervals(p[:3], p[3:6])
    lo, hi = min(water, key=lambda row: max(row[0]["t"], -row[1]["t"], 0.0))
    left, right = max(lo["t"] + 1e-4, -0.5 * p[6]), min(hi["t"] - 1e-4, 0.5 * p[6])
    if left >= right:
        raise ValueError("No usable finite start in water")
    p[:3] += 0.5 * (left + right) * p[3:6]
    p[6] = right - left


class Analog(WaterOccupancyModel):
    """Use the released optical geometry with measured PE as the observation."""

    def __init__(self, event, geometry, background, kind, order=8):
        super().__init__(event.pmt_ids, event.charges_pe, copy_geometry(geometry), background, kind, order)
        if kind == "finite":
            for j in range(self.k):
                contain_finite(self.x[9 * j : 9 * j + 9])
        self.obs = observation(event.charges_pe)
        self.total = max(float(event.charges_pe.sum() / self.obs.gain), 1.0)
        self.x[-1] = np.log(self.total)

    def evaluate(self, x):
        if self.last is not None and np.array_equal(x, self.last[0]):
            return self.last[1:]
        sources = [self.background]
        geometry_jacobian = []
        n = self.size * self.k
        for geometry in x[:n].reshape(self.k, self.size):
            shape, jacobian = self.component(geometry)
            sources.append(shape)
            geometry_jacobian.append(jacobian)
        shapes = np.asarray(sources)
        weights = softmax(np.r_[0.0, x[n:-1]])
        amplitude = np.exp(x[-1])
        components = amplitude * weights[:, None] * shapes
        mean = components.sum(0)
        ll, residual = self.obs.score(mean, self.ix)
        mean_gradient = np.zeros((len(mean), len(x)))
        for j, jacobian in enumerate(geometry_jacobian):
            mean_gradient[:, self.size * j : self.size * (j + 1)] = amplitude * weights[j + 1] * jacobian
        mean_gradient[:, n:-1] = components[1:].T - mean[:, None] * weights[1:]
        mean_gradient[:, -1] = mean
        posterior = components[:, self.ix] / mean[self.ix]
        self.last = x.copy(), ll, residual @ mean_gradient, posterior, mean, mean_gradient
        return self.last[1:]


class SearchLimit(Exception):
    pass


def endpoint(model, budget):
    x = model.x
    limits = np.asarray(model.bounds)
    ll, _, _, mean, jac = model.evaluate(x)
    information = np.sum(jac * jac / np.maximum(mean[:, None], 1e-8), axis=0)
    scale = np.clip(1 / np.sqrt(np.maximum(information, 1e-8)), 1e-5, 1.0)
    best = None
    best_ll = -np.inf
    calls = 0

    def keep(y):
        nonlocal best, best_ll
        z = np.clip(x + scale * y, *limits.T)
        if (model.water_upper - model.water_values(z)).min() < -1e-7:
            return
        if abs(model.axis_norms(z) - 1).max() > 1e-6:
            return
        value = model.evaluate(z)[0]
        if value > best_ll:
            best, best_ll = z, value

    def objective(y):
        nonlocal calls
        if calls >= budget:
            raise SearchLimit
        calls += 1
        value, gradient, *_ = model.evaluate(np.clip(x + scale * y, *limits.T))
        return -value / model.total, (gradient / model.total) * scale

    constraints = [
        dict(
            type=c["type"],
            fun=lambda y, f=c["fun"]: f(x + scale * y),
            jac=lambda y, j=c["jac"]: j(x + scale * y) * scale,
        )
        for c in model.constraints
    ]
    bounds = (limits - x[:, None]) / scale[:, None]
    keep(np.zeros(len(x)))
    try:
        fit = minimize(
            objective,
            np.zeros(len(x)),
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            callback=keep,
            options=dict(maxiter=budget, ftol=1e-10),
        )
        keep(fit.x)
    except SearchLimit:
        pass
    except ValueError as error:
        if str(error) != "No optical support":
            raise
    return best, calls
