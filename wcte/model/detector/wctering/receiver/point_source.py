"""Point cone with analytic unobstructed spherical-cap exposure.

The cap is an isolated surface in homogeneous water. Refraction, module
obstruction and the beam window are separate effects, not included here.
"""

from functools import lru_cache
import math
import numpy as np
from wctering.receiver.normalize import normalize
from numba import njit
from scipy.special import roots_legendre

Z0 = 34.597 / 53.0
T2 = 1.0 - Z0 * Z0
T = math.sqrt(T2)


@njit(cache=True)
def cap(mu):
    """Axially normalized projected positive flux and d/d(cos incidence)."""
    if mu >= T:
        return mu, 1.0
    if mu <= -T:
        return 0.0, 0.0
    h = math.sqrt(max(T2 - mu * mu, 0.0))
    t2 = 1.0 - mu * mu
    phi = math.atan2(h, -mu * Z0)
    value = (math.atan2(h, Z0) - Z0 * h + mu * T2 * phi) / (math.pi * T2)
    derivative = (T2 * phi + mu * Z0 * h / t2) / (math.pi * T2)
    # Only floating-point cancellation at the vanishing-flux endpoint.
    return max(value, 0.0), min(max(derivative, 0.0), 1.0)


@lru_cache(maxsize=16)
def quadrature(order):
    x, w = roots_legendre(order)
    return np.ascontiguousarray(x * 0.5), np.ascontiguousarray(w * 0.5)


def _point_integral(positions, normals, x):
    if x.shape != (9,) or not np.isfinite(x).all() or np.linalg.norm(x[3:6]) < 1e-10:
        raise ValueError("Invalid source")
    nodes, weights = quadrature(1)
    from wctering.receiver.native import library, integrate as native_integrate

    if library() is not None:
        raw, jac = native_integrate(positions, normals, x)
    else:
        from wctering.receiver.buffered import point_integrate

        raw, jac = point_integrate(positions, normals, x, nodes, weights)
    return normalize(raw, jac)


# The uniform-source Jacobian has a length column at index 6. A point
# source omits only that column; build the fixed selector once.
_POINT_COLUMNS = np.array([0, 1, 2, 3, 4, 5, 7, 8])


def point_source(positions, normals, parameters):
    """Cache the final eight-column point derivative, including its layout."""
    from wctering.receiver.cache import evaluate

    from wctering.receiver.point_shape import backend, evaluate as compiled_point

    def light(p, n, x):
        functions = backend()
        if functions is not None:
            return compiled_point(*functions, p, n, x)
        y = np.concatenate((x[:6], [0.0], x[6:]))
        shape, jacobian = _point_integral(p, n, y)
        return shape, jacobian[:, _POINT_COLUMNS]

    return evaluate(light, positions, normals, parameters, kind="point", order=1)
