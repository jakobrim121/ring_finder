"""Mixture responsibilities and normalization reductions.

The likelihood formulas live in charge.py and occupancy.py. These small
compiled loops remove temporary arrays while preserving component order,
PMT reduction order and the original C/Fortran memory layouts."""

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def charge_vectors(shapes, weights, indices, charges):
    """Mixture density and responsibilities in the original component order.

    The transposed allocation preserves the column-contiguous layout produced
    by the original indexed NumPy expression. Subsequent BLAS calls therefore
    receive the same strides as before. No independent PMT sums are reordered.
    """
    posterior = np.empty((len(indices), len(weights))).T
    density = np.empty(len(indices))
    weighted_inverse = np.empty(len(indices))
    for hit in range(len(indices)):
        total = -0.0
        for component in range(len(weights)):
            term = weights[component] * shapes[component, indices[hit]]
            posterior[component, hit] = term
            total += term
        density[hit] = total
        weighted_inverse[hit] = charges[hit] / total
        for component in range(len(weights)):
            posterior[component, hit] /= total
    return density, posterior, weighted_inverse


@njit(cache=True, error_model="numpy")
def occupancy_vectors(observed, probability, log_probability, expected):
    """Return likelihood summands and derivatives before the original sums.

    NumPy still computes expm1, logarithms and the final pairwise reductions.
    This pass only avoids temporary arrays between elementary operations.
    """
    terms = np.empty(len(observed))
    derivative = np.empty(len(observed))
    residual = np.empty(len(observed))
    for hit in range(len(observed)):
        y = observed[hit]
        terms[hit] = y * log_probability[hit] - (1 - y) * expected[hit]
        derivative[hit] = 1.0 - y / probability[hit]
        residual[hit] = derivative[hit] * expected[hit]
    return terms, derivative, residual


@njit(cache=True)
def full_mass(shapes):
    """Preserve the original indexed matrix's receiver-by-receiver sum order.

    ``shapes[:, mask]`` was column-contiguous after boolean indexing. Its
    reduction accumulated each component in PMT order. Summing the contiguous
    input rows directly would instead use NumPy's pairwise reduction.
    """
    mass = np.zeros(len(shapes))
    for receiver in range(shapes.shape[1]):
        for component in range(len(shapes)):
            mass[component] += shapes[component, receiver]
    return mass


@njit(cache=True)
def full_jacobian_mass(jacobian):
    """Match reduction after row indexing, including column-contiguous inputs.

    A selected point-source Jacobian is column-contiguous. The original row
    selection copied it into row order before summation; removing that copy
    must not switch to NumPy's pairwise reduction on the contiguous axis.
    """
    mass = np.zeros(jacobian.shape[1])
    for receiver in range(len(jacobian)):
        for parameter in range(jacobian.shape[1]):
            mass[parameter] += jacobian[receiver, parameter]
    return mass
