"""Refit the geometry of an independent prompt source in unexplained light."""

import numpy as np
from scipy.optimize import minimize

from analog_optical import SearchLimit
from wctering.likelihood.geometry import BOX, NORMALS, UPPER
from wctering.likelihood.water_boundary import clearance
from wctering.receiver.angular_panels import finite_integrate
from wctering.receiver.finite_source import quadrature


def optical_shape(model, path, order):
    """Normalize the optical response over hit and unhit PMTs."""
    shape, derivative, _ = finite_integrate(model.pos, model.normals, path, *quadrature(order))
    total = shape.sum()
    if total <= 1e-250:
        return np.zeros_like(shape), np.zeros_like(derivative)
    normalized = shape / total
    return normalized, (derivative - normalized[:, None] * derivative.sum(0)) / total


def local_fit(model, x, initial, fixed, budget):
    """Move one residual path while keeping the incumbent light fixed."""
    pole = x[3:6].copy()
    t = np.cross(pole, np.eye(3)[np.argmin(abs(pole))])
    t /= np.linalg.norm(t)
    frame = np.column_stack([t, np.cross(pole, t)])
    y = np.r_[x[:3], 0.0, 0.0, x[6:], np.log(initial)]
    limits = np.array(
        [
            *BOX,
            (-0.65, 0.65),
            (-0.65, 0.65),
            (0.015, 2.8),
            (np.deg2rad(12.0), np.deg2rad(42.3)),
            tuple(np.log(np.deg2rad([0.8, 6.0]))),
            (np.log(0.01), np.log(1e5)),
        ]
    )
    y = np.clip(y, *limits.T)
    units = np.r_[0.06, 0.06, 0.06, 0.14, 0.14, 0.07, 0.035, 0.3, 0.35]
    bounds = (limits - y[:, None]) / units[:, None]
    work, evaluation, calls = None, None, 0

    def geometry(z):
        p = y + units * z
        raw = pole + frame @ p[3:5]
        norm = np.linalg.norm(raw)
        axis = raw / norm
        transform = np.zeros((9, 9))
        transform[:3, :3] = np.eye(3)
        transform[3:6, 3:5] = (frame - np.outer(axis, axis @ frame)) / norm
        transform[6:, 5:8] = np.eye(3)
        return np.r_[p[:3], axis, p[5:8]], transform * units, np.exp(p[-1])

    def water(z):
        nonlocal work
        if work is None or not np.array_equal(work[0], z):
            p, dx, _ = geometry(z)
            end = 0.5 * p[6] * p[3:6]
            prism = np.r_[UPPER - (p[:3] - end) @ NORMALS.T, UPPER - (p[:3] + end) @ NORMALS.T]
            sign = np.repeat([-1.0, 1.0], len(NORMALS))
            norm = np.tile(NORMALS, (2, 1))
            prism_jac = np.column_stack(
                [
                    -norm,
                    -0.5 * p[6] * sign[:, None] * norm,
                    -0.5 * sign * (norm @ p[3:6]),
                    np.zeros((len(sign), 2)),
                ]
            )
            g, c, d, l = clearance(p[:3], p[3:6], p[6])
            jac = np.vstack([prism_jac, np.column_stack([c, d, l, np.zeros((len(g), 2))])])
            work = z.copy(), np.r_[prism, g], jac @ dx
        return work[1:]

    def score(z):
        nonlocal evaluation
        if evaluation is not None and np.array_equal(z, evaluation[0]):
            return evaluation[1:]
        p, dx, intensity = geometry(z)
        shape, jac = optical_shape(model, p, 8)
        ll, derivative = model.obs.score(fixed + intensity * shape, model.ix)
        slope = intensity * (derivative @ jac @ dx)
        slope[-1] = units[-1] * intensity * (derivative @ shape)
        evaluation = z.copy(), ll, slope
        return evaluation[1:]

    z = np.zeros(len(y))
    best = z
    best_score = score(z)[0] if min(water(z)[0]) >= -1e-7 else -np.inf

    def objective(z):
        nonlocal calls
        if calls >= budget:
            raise SearchLimit
        calls += 1
        ll, slope = score(z)
        return -ll, slope

    def keep(z):
        nonlocal best, best_score
        if min(water(z)[0]) >= -1e-7:
            ll = score(z)[0]
            if ll > best_score:
                best, best_score = z.copy(), ll

    try:
        result = minimize(
            objective,
            z,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=[dict(type="ineq", fun=lambda z: water(z)[0], jac=lambda z: water(z)[1])],
            callback=keep,
            options=dict(maxiter=budget, ftol=1e-7),
        )
        keep(result.x)
    except SearchLimit:
        pass
    path, _, photons = geometry(best)
    if min(water(best)[0]) < -1e-7:
        return None
    return path, photons, calls
