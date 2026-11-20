"""Stable log-sum-exp with the qualified NumPy arithmetic order.

The public wrapper selects a compiled one-dimensional path on the validated
NumPy ABI, and readable NumPy/SciPy fallbacks elsewhere. The compiled helper
uses process-owned ufunc pointers supplied at runtime, never saved addresses."""

from functools import lru_cache
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp


@lru_cache(maxsize=1)
def _vector_backend():
    """Unsupported platforms retain the readable NumPy implementation below."""
    from wctering.receiver.numpy_loops import runtime

    try:
        function, loops, _, _ = runtime()
        return logsumexp_vector, function, loops
    except RuntimeError:
        return None


def logsumexp(values, axis=None):
    """Sum positive exponentials stably, preserving the reference reduction.

    SciPy's general implementation supports weighted, signed, complex and
    non-NumPy inputs. Our hot paths use unweighted float64 arrays. For these,
    retain its maximum-term separation and NumPy summation order while avoiding
    the generic dispatch and the unused overflow calculation. Unusual inputs
    still use SciPy. Keep this qualified against the installed SciPy version.
    """
    values = np.asarray(values)
    if values.dtype != np.float64 or not values.size:
        return scipy_logsumexp(values, axis=axis)
    if values.ndim == 1 and axis is None:
        backend = _vector_backend()
        if backend is not None:
            compiled, function, loops = backend
            answer, finite = compiled(function, loops, values)
            if finite:
                return np.float64(answer)
    axes = tuple(range(values.ndim)) if axis is None else axis
    maximum = np.max(values, axis=axes, keepdims=True)
    if not np.isfinite(maximum).all():
        return scipy_logsumexp(values, axis=axis)
    at_maximum = values == maximum
    remainder = values.copy()
    remainder[at_maximum] = -np.inf
    multiplicity = np.sum(at_maximum.astype(values.dtype), axis=axes, keepdims=True, dtype=values.dtype)
    mass = np.sum(np.exp(remainder - maximum), axis=axes, keepdims=True, dtype=values.dtype) / multiplicity
    answer = np.log1p(mass) + np.log(multiplicity) + maximum
    answer = np.squeeze(answer, axis=axes)
    return answer[()] if answer.ndim == 0 else answer


from numba import njit
from wctering.receiver.numpy_loops import unary, total


@njit(cache=True, error_model="numpy")
def logsumexp_vector(function, loops, values):
    """Keep SciPy's maximum separation, multiplicity and exact reduction tree."""
    maximum = np.max(values)
    if not np.isfinite(maximum):
        return np.nan, False
    remainder = np.empty(len(values))
    multiplicity = 0.0
    for index in range(len(values)):
        if values[index] == maximum:
            remainder[index] = -np.inf
            multiplicity += 1.0
        else:
            remainder[index] = values[index] - maximum
    mass = total(function, loops[2], unary(function, loops[4], remainder)) / multiplicity
    log_mass = unary(function, loops[5], np.array([mass]))[0]
    log_multiplicity = unary(function, loops[1], np.array([multiplicity]))[0]
    return (log_mass + log_multiplicity) + maximum, True
