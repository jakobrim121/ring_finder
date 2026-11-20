"""One compiled point-source wrapper, retaining NumPy's normalization sum."""

from functools import lru_cache
import numpy as np
from numba import njit
from wctering.receiver.bridge import point
from wctering.receiver.normalize import _normalized
from wctering.receiver.numpy_loops import total


@lru_cache(maxsize=1)
def backend():
    from wctering.logmath import _vector_backend
    from wctering.receiver.native import library, bridge_functions

    numpy_backend = _vector_backend()
    if numpy_backend is None or library(False) is None:
        return None
    _, numpy_function, loops = numpy_backend
    return bridge_functions(False)[0], numpy_function, loops


@njit(cache=True)
def evaluate(point_function, numpy_function, loops, positions, normals, parameters):
    if (
        parameters.shape != (8,)
        or not np.isfinite(parameters).all()
        or np.sqrt(parameters[3:6] @ parameters[3:6]) < 1e-10
    ):
        raise ValueError("Invalid source")
    uniform = np.empty(9)
    uniform[:6] = parameters[:6]
    uniform[6] = 0.0
    uniform[7:] = parameters[6:]
    raw, jacobian = point(point_function, positions, normals, uniform)
    mass = total(numpy_function, loops[2], raw)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("No optical support")
    shape, derivative = _normalized(raw, jacobian, mass)
    # Advanced column indexing in the original returns a Fortran-layout copy.
    # Preserve that layout: subsequent NumPy BLAS dispatch depends on strides.
    selected = np.empty((8, len(raw))).T
    for column in range(8):
        original_column = column if column < 6 else column + 1
        for hit in range(len(raw)):
            selected[hit, column] = derivative[hit, original_column]
    return shape, selected
