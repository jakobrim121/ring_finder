"""Charge and radial variance for every candidate cone in one pass.

Columns are candidate axes; rows are the original retained-hit order. There
is no fastmath, reassociation, parallel reduction or change in cone proposals.
NumPy retains the exponential and score evaluation outside this fused loop.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def shell_moments(angles, charges, kernel, opening):
    count = angles.shape[1]
    charge = np.zeros(count)
    squared_residual = np.zeros(count)
    for hit in range(len(charges)):
        for axis in range(count):
            weight = charges[hit] * kernel[hit, axis]
            residual = angles[hit, axis] - opening
            charge[axis] += weight
            squared_residual[axis] += weight * (residual * residual)
    return charge, squared_residual


@njit(cache=True)
def advance_kernel(kernel, growth, factor):
    """Apply both original multiplications with one matrix traversal."""
    for hit in range(kernel.shape[0]):
        for axis in range(kernel.shape[1]):
            value = kernel[hit, axis] * growth[hit, axis]
            kernel[hit, axis] = value * factor
