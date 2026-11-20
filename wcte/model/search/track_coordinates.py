"""Optimize on the unit-axis and transverse-bend manifold directly."""

import numpy as np
from scipy.optimize import minimize, OptimizeResult


class Chart:
    """Use two axis coordinates, two bend coordinates, range and length.

    The long-axis component of a bend is redundant, and length is better
    conditioned than the ratio of length to range. This map introduces no
    new physics. A single local axis chart covers the hemisphere of a start.
    """

    def __init__(self, model, x):
        self.model = model
        self.axes = []
        starts = [3 + 9 * j for j in range(len(model.particles))]
        lower, upper = np.asarray(model.bounds).T
        y = list(x[:3])
        bounds = list(model.bounds[:3])
        for start in starts:
            axis = x[start : start + 3]
            unit = np.eye(3)[np.argmin(abs(axis))]
            tangent = np.cross(axis, unit)
            tangent /= np.linalg.norm(tangent)
            frame = np.column_stack([tangent, np.cross(axis, tangent)])
            self.axes.append((axis, frame))
            r, f = x[start + 6 : start + 8]
            y.extend([0.0, 0.0, *(frame.T @ x[start + 3 : start + 6]), r, r * f, x[start + 8]])
            max_bend = upper[start + 3]
            # The projected tangent-plane chart can require a longer
            # coordinate vector after the direction moves away from its pole.
            bounds.extend(
                [(-2.0, 2.0)] * 2
                + [(-3 * max_bend, 3 * max_bend)] * 2
                + [
                    (lower[start + 6], upper[start + 6]),
                    (0.0001 * lower[start + 6], upper[start + 6]),
                    model.bounds[start + 8],
                ]
            )
        self.initial = np.r_[y, x[-2:]]
        self.bounds = np.array([*bounds, *model.bounds[-2:]])
        self.cached = None

    def full(self, y):
        if self.cached is not None and np.array_equal(y, self.cached[0]):
            return self.cached[1:]
        x = np.r_[y[:3], np.zeros(self.model.ng - 3), y[-2:]]
        jacobian = np.zeros((len(x), len(y)))
        jacobian[:3, :3] = np.eye(3)
        jacobian[-2:, -2:] = np.eye(2)
        # The two explicit length inequalities replace the old f bounds.
        bounds = np.zeros((2 * len(self.axes), len(y)))
        margins = np.empty(len(bounds))
        for j, (axis, frame) in enumerate(self.axes):
            i, k = 3 + 9 * j, 3 + 7 * j
            z = axis + frame @ y[k : k + 2]
            norm = np.linalg.norm(z)
            unit = z / norm
            da = (frame - np.outer(unit, unit @ frame)) / norm
            approximate_bend = frame @ y[k + 2 : k + 4]
            along = unit @ approximate_bend
            bend = approximate_bend - along * unit
            ran, length, width = y[k + 4 : k + 7]
            fraction = length / ran
            x[i : i + 9] = np.r_[unit, bend, ran, np.clip(fraction, 0.0001, 1.0), width]
            jacobian[i : i + 3, k : k + 2] = da
            jacobian[i + 3 : i + 6, k : k + 2] = -along * da - np.outer(unit, approximate_bend @ da)
            jacobian[i + 3 : i + 6, k + 2 : k + 4] = frame - np.outer(unit, unit @ frame)
            jacobian[i + 6, k + 4] = 1.0
            # Trial evaluations outside the range envelope render the nearest
            # valid length. SLSQP sees the unclipped, analytic constraint.
            if 0.0001 <= fraction <= 1.0:
                jacobian[i + 7, k + 4 : k + 6] = -fraction / ran, 1 / ran
            jacobian[i + 8, k + 6] = 1.0
            margins[2 * j : 2 * j + 2] = length - 0.0001 * ran, ran - length
            bounds[2 * j : 2 * j + 2, k + 4 : k + 6] = [(-0.0001, 1.0), (1.0, -1.0)]
        self.cached = y.copy(), x, jacobian, margins, bounds
        return self.cached[1:]


class EvaluationLimit(Exception):
    pass


def fit(
    model,
    start,
    iterations,
    maximum_evaluations,
    *,
    full_likelihood=False,
    chart=None,
    check_initial=False,
):
    """Scale a reduced chart by the expected PE information at its start."""
    if chart is None:
        chart = Chart(model, start)
    x, mapping, _, _ = chart.full(chart.initial)
    initial = model(x, True)
    _, signal, _, jac = model.primary_cache
    eff, diffuse = np.exp(x[-2:])
    mean = eff * signal + diffuse * model.bg
    j = np.column_stack([eff * jac, eff * signal, diffuse * model.bg]) @ mapping
    diagonal = np.sum(j * j / np.maximum(mean[:, None], 1e-10), axis=0)
    scale = np.clip(1 / np.sqrt(np.maximum(diagonal, 1e-4)), 1e-5, 0.5)
    # The Fisher scale above uses the total expected information. Use the
    # total log likelihood with it so the local curvature is order one.
    objective_scale = model.total if full_likelihood else 1.0
    evaluations = 0
    best, best_value = x, initial["objective"]
    if check_initial and (min(model.inequality(x)) < -1e-6 or min(chart.full(chart.initial)[2]) < -1e-7):
        best_value = np.inf
    cached = None

    def coordinates(z):
        return chart.full(chart.initial + scale * z)

    def state(z):
        nonlocal cached
        if cached is not None and np.array_equal(z, cached[0]):
            return cached[1:]
        x, mapping, _, _ = coordinates(z)
        value, gradient = model(x)
        cached = z.copy(), x, value, (gradient @ mapping) * scale
        return cached[1:]

    def objective(z):
        nonlocal evaluations
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            raise EvaluationLimit
        evaluations += 1
        _, value, gradient = state(z)
        return objective_scale * value, objective_scale * gradient

    def inequality(z):
        x, _, margin, _ = coordinates(z)
        return np.r_[model.inequality(x), margin]

    def inequality_jacobian(z):
        x, mapping, _, bounds = coordinates(z)
        derivative = np.vstack([model.inequality_jacobian(x) @ mapping, bounds])
        return derivative * scale

    def keep(z):
        nonlocal best, best_value
        x, _, margins, _ = coordinates(z)
        if min(margins) < -1e-7 or min(model.inequality(x)) < -1e-6:
            return
        x, value, _ = state(z)
        if value < best_value:
            best, best_value = x, value

    bounds = chart.bounds - chart.initial[:, None]
    bounds = bounds / scale[:, None]
    try:
        result = minimize(
            objective,
            np.zeros(len(scale)),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=[dict(type="ineq", fun=inequality, jac=inequality_jacobian)],
            callback=keep,
            options=dict(maxiter=iterations, ftol=1e-9),
        )
        keep(result.x)
    except EvaluationLimit:
        result = OptimizeResult(nfev=evaluations, success=False)
    return result, best
