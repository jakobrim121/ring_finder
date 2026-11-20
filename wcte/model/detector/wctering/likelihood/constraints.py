"""Unit-axis and water-boundary constraints for the likelihood models.

The compiled functions evaluate geometry only. last_input reuses an exact
repeated evaluation within one model; returned copies protect the cache."""

import numpy as np
from numba import njit


@njit(cache=True)
def direction(z):
    norm = np.sqrt(z @ z)
    if norm < 1e-10:
        raise ValueError("Undefined axis")
    a = z / norm
    return a, (np.eye(3) - np.outer(a, a)) / norm


@njit(cache=True)
def axis_norms(x, size, count):
    values = np.empty(count)
    for ring in range(count):
        axis = x[size * ring + 3 : size * ring + 6]
        values[ring] = axis @ axis
    return values


@njit(cache=True)
def axis_jacobian(x, size, count):
    jacobian = np.zeros((count, len(x)))
    for ring in range(count):
        jacobian[ring, size * ring + 3 : size * ring + 6] = 2 * x[size * ring + 3 : size * ring + 6]
    return jacobian


@njit(cache=True)
def water_values(x, size, count, normals):
    values = np.empty(36 * count)
    for ring in range(count):
        p = x[size * ring : size * (ring + 1)]
        a, _ = direction(p[3:6])
        length = p[6] if size == 9 else 0.0
        for end, sign in enumerate((-1.0, 1.0)):
            values[36 * ring + 18 * end : 36 * ring + 18 * (end + 1)] = normals @ (
                p[:3] + sign * 0.5 * length * a
            )
    return values


@njit(cache=True)
def water_jacobian(x, size, count, normals):
    jacobian = np.zeros((36 * count, len(x)))
    for ring in range(count):
        p = x[size * ring : size * (ring + 1)]
        a, da = direction(p[3:6])
        length = p[6] if size == 9 else 0.0
        for end, sign in enumerate((-1.0, 1.0)):
            rows = slice(36 * ring + 18 * end, 36 * ring + 18 * (end + 1))
            offset = size * ring
            jacobian[rows, offset : offset + 3] = normals
            if size == 9:
                jacobian[rows, offset + 3 : offset + 6] = sign * 0.5 * length * (normals @ da)
                jacobian[rows, offset + 6] = sign * 0.5 * (normals @ a)
    return jacobian


from functools import wraps


def last_input(function):
    """Keep one result per method and model, without rounding parameters.

    SLSQP, its callback and the final certificate often ask for the same
    constraint separately. Return a copy so callers retain independent arrays.
    This cache contains geometry evaluations, never fitted event answers.
    """

    @wraps(function)
    def evaluate(self, parameters):
        cache = self.__dict__.setdefault("_constraint_cache", {})
        key = parameters.shape, parameters.dtype.str, parameters.tobytes()
        previous = cache.get(function.__qualname__)
        if previous is None or previous[0] != key:
            previous = key, function(self, parameters)
            cache[function.__qualname__] = previous
        return previous[1].copy()

    return evaluate
