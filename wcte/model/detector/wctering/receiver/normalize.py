"""Normalize a light source over every active PMT and differentiate it.

The NumPy one-dimensional total keeps its original summation algorithm. The
Jacobian columns keep their original hit order. The compiled pass only avoids
three temporary matrices; it never changes the optical formula or quadrature.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _normalized(raw, jacobian, total):
    rows, columns = jacobian.shape
    derivative_total = np.zeros(columns)
    for hit in range(rows):
        for parameter in range(columns):
            derivative_total[parameter] += jacobian[hit, parameter]
    shape = np.empty(rows)
    derivative = np.empty((rows, columns))
    for hit in range(rows):
        shape[hit] = raw[hit] / total
        for parameter in range(columns):
            correction = shape[hit] * derivative_total[parameter]
            derivative[hit, parameter] = (jacobian[hit, parameter] - correction) / total
    return shape, derivative


def normalize(raw, jacobian):
    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("No optical support")
    return _normalized(raw, jacobian, total)
