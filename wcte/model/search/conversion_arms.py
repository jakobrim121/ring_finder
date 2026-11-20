"""Fit the two photon-conversion arms with a shared vertex and light yield."""

import numpy as np
from scipy.optimize import minimize

from charge_starts import allowed_chord
from geometry import hardware_margin
from material import material
from source import raw
from wctering.likelihood.constraints import direction


def optimize(
    model, start, fixed, response, order, budget, *, maximum_range=None, prior=None, support_sides=None
):
    """Refine the pair from the measured broad cone using analytic derivatives."""
    poles = start[:, 3:6]
    frames = []
    for pole in poles:
        side = np.cross(pole, np.eye(3)[np.argmin(abs(pole))])
        side /= np.linalg.norm(side)
        frames.append(np.column_stack([side, np.cross(pole, side)]))
    water = material("electron")
    minimum = water["range_at"](water["energy"][0] + 1.0) / 1000.0
    maximum = water["maximum_range_mm"] / 1000.0 * 0.999
    if maximum_range is not None:
        maximum = min(maximum, maximum_range)
    scale = np.r_[[0.055] * 3, [0.16] * 4, [0.35] * 2]
    limits = np.array(
        [(-0.18, 0.18)] * 3
        + [(-0.34, 0.34)] * 4
        + list(zip(np.log(minimum / start[:, 9]), np.log(maximum / start[:, 9])))
    )

    def geometry(z):
        x = scale * z
        paths = start.copy()
        paths[:, :3] += x[:3]
        paths[:, 9] *= np.exp(x[-2:])
        axes = []
        for j, (pole, frame) in enumerate(zip(poles, frames)):
            paths[j, 3:6], slope = direction(pole + frame @ x[3 + 2 * j : 5 + 2 * j])
            axes.append(slope @ frame)
        return paths, axes

    previous = None

    def evaluate(z):
        nonlocal previous
        if previous is not None and np.array_equal(z, previous[0]):
            return previous[1:]
        paths, axes = geometry(z)
        total = np.zeros(len(model.active))
        jacobian = np.zeros((len(total), len(z)))
        cores, core_derivatives = [], []
        for j, (path, chart) in enumerate(zip(paths, axes)):
            parts, derivative = raw(model.pos, model.normals, path, order, False, "electron")
            length = 1000.0 * path[9]
            light = length * parts.sum(0)
            jac = length * derivative.sum(0)
            total += light
            jacobian[:, :3] += jac[:, :3]
            jacobian[:, 3 + 2 * j : 5 + 2 * j] = jac[:, 3:6] @ chart
            jacobian[:, -2 + j] = jac[:, 9] * path[9] + light
            if support_sides is not None:
                core = length * parts[0, model.ix]
                derivative = length * derivative[0, model.ix]
                gradient = np.zeros((len(core), len(z)))
                gradient[:, :3] = derivative[:, :3]
                gradient[:, 3 + 2 * j : 5 + 2 * j] = derivative[:, 3:6] @ chart
                gradient[:, -2 + j] = derivative[:, 9] * path[9] + core
                cores.append(core)
                core_derivatives.append(gradient)
        mean = fixed + response * total
        ll, residual = model.obs.score(mean, model.ix)
        slope = (residual @ jacobian) * response
        if prior is not None:
            penalty, derivative = prior(paths, axes)
            ll -= penalty
            slope += derivative
        scores, gradients = [], []
        if support_sides is not None:
            from visibility_profile import support_score

            observed = mean[model.ix]
            d_mean = response * jacobian[model.ix]
            for side, core, derivative in zip(support_sides, cores, core_derivatives):
                if not side:
                    continue
                support, full, other = support_score(
                    observed, observed - response * core, model.obs.offsets, model.obs.coefficients
                )
                scores.append((side * (support - 30.0) - 0.025) / 100.0)
                gradient = (full + other) @ d_mean - response * (other @ derivative)
                gradients.append((side / 100.0) * gradient * scale)
        previous = z.copy(), -ll, slope * scale, np.asarray(scores), np.asarray(gradients)
        return previous[1:]

    def value(z):
        loss, gradient, _, _ = evaluate(z)
        return loss, gradient

    def contained(paths):
        for path in paths:
            chord = allowed_chord(path[:3], path[3:6])
            if chord is None or chord[0] > 0.0 or path[9] >= chord[1] or hardware_margin(path) <= 1e-5:
                return False
        return True

    origin = np.zeros(len(scale))
    if not contained(start):
        return None
    best = np.inf, None
    calls = 0

    class BudgetReached(Exception):
        pass

    def objective(z):
        nonlocal calls
        if calls >= budget and support_sides is not None:
            raise BudgetReached
        calls += 1
        return value(z)

    def keep(z):
        nonlocal best
        paths, _ = geometry(z)
        if contained(paths):
            loss, _, margin, _ = evaluate(z)
            if margin.min(initial=0.0) < -1e-6:
                return
            if loss < best[0]:
                best = loss, z.copy()

    keep(origin)
    method = "L-BFGS-B" if support_sides is None else "SLSQP"
    constraints = (
        []
        if support_sides is None
        else [dict(type="ineq", fun=lambda z: evaluate(z)[2], jac=lambda z: evaluate(z)[3])]
    )
    options = dict(maxiter=budget, ftol=1e-8)
    if support_sides is None:
        options["maxfun"] = budget
    try:
        result = minimize(
            objective,
            origin,
            jac=True,
            method=method,
            constraints=constraints,
            bounds=limits / scale[:, None],
            callback=keep,
            options=options,
        )
        keep(result.x)
    except BudgetReached:
        pass
    return geometry(best[1])[0] if best[1] is not None else None
