"""Geometry feasibility, charge-likelihood conversion, and amplitude initialization."""

import numpy as np
from scipy.optimize import minimize
from wctering.logmath import logsumexp


class Stationary(Exception):
    def __init__(self, x):
        self.x = x.copy()


def feasible(c, tolerance=1e-7):
    return (
        c["water_margin_m"] >= -tolerance and c["axis_unit_error"] <= tolerance and c["box_margin"] >= -1e-8
    )


def convert(obj, r):
    q = obj.total
    result = dict(r)
    result["conditional_ll"] = r["ll"]
    result["ll"] = r["ll"] + q * np.log(q) - q
    result["density"] = r["expected_charge_all"][obj.ix]
    result["expected_training_charge"] = q
    result["expected_total_charge"] = float(r["expected_charge_all"].sum())
    return result


def profile(obj, x):
    result = obj(x, True)
    shapes = result["shapes"][:, obj.ix]
    mass = result["training_component_probability"]
    offset = obj.k * obj.size

    def fun(z):
        logits = np.concatenate(([0.0], z))
        p = np.exp(logits - logsumexp(logits))
        terms = p[:, None] * shapes
        density = terms.sum(0)
        prob = p @ mass
        return (
            -float(obj.w @ np.log(density / prob)) / obj.total,
            p[1:] * mass[1:] / prob - (terms[1:] / density @ obj.w) / obj.total,
        )

    fitted = minimize(
        fun,
        x[offset:],
        jac=True,
        method="L-BFGS-B",
        bounds=obj.bounds[offset:],
        options=dict(maxiter=200, ftol=1e-14, gtol=1e-10),
    )
    y = x.copy()
    y[offset:] = fitted.x
    if obj(y)[0] > result["objective"] + 1e-11:
        y = x.copy()
    return y, dict(success=bool(fitted.success), message=str(fitted.message), nit=fitted.nit)


def could_be_stationary(model, x, gradient, tolerance, feasibility_tolerance):
    """Cheap necessary conditions for the existing full KKT certificate.

    Amplitude logits and occupancy intensity do not appear in geometric
    constraints. Unless their own bounds are active, their gradient components
    pass through the KKT projection unchanged. Use the same subtractions and
    active-bound threshold as ChargeModel.kkt, including its boundary-rounding rule.
    """
    for j in range(model.k * model.size, len(x)):
        lower, upper = model.bounds[j]
        if x[j] - lower > 1e-6 and upper - x[j] > 1e-6 and abs(gradient[j]) > tolerance:
            return False
    return bool(abs(model.axis_norms(x) - 1).max() <= feasibility_tolerance)
