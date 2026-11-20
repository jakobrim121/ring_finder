"""One compiled angular wrapper, preserving the original fallback and sums.

Both native function pointers are runtime arguments owned by cached libraries.
NumPy's installed sum loop retains its exact reduction tree. The raw optical
kernel, thresholds, Jacobian layout and normalization formula are unchanged.
"""

import ctypes
from functools import lru_cache
import numpy as np
from numba import njit
from wctering.receiver.numpy_loops import total
from wctering.receiver.bridge import finite
from wctering.receiver.normalize import _normalized


@lru_cache(maxsize=1)
def backend():
    from wctering.logmath import _vector_backend
    from wctering.receiver.angular_panels import library
    from wctering.receiver.native import library as original, bridge_functions

    loops = _vector_backend()
    native = library()
    if loops is None or native is None or original(False) is None:
        return None
    _, numpy_function, numpy_loops = loops
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    arguments = [
        size,
        pointer,
        pointer,
        pointer,
        size,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    function = ctypes.CFUNCTYPE(ctypes.c_int, *arguments)(ctypes.cast(native.finite_rays, pointer).value)
    return function, bridge_functions(False)[1], numpy_function, numpy_loops


@njit(cache=True)
def evaluate(function, original, numpy_function, loops, positions, normals, x, nodes, weights):
    values = np.empty(len(positions))
    jacobian = np.empty((len(positions), 9))
    counts = np.empty(len(positions), dtype=np.int64)
    status = function(
        len(positions),
        positions.ctypes,
        normals.ctypes,
        x.ctypes,
        len(nodes),
        nodes.ctypes,
        weights.ctypes,
        values.ctypes,
        jacobian.ctypes,
        counts.ctypes,
    )
    # A failed kernel can leave partially written arrays; never inspect them.
    fallback = status != 0
    mass = 0.0
    if not fallback:
        fallback = not np.isfinite(values).all()
        if not fallback:
            mass = total(numpy_function, loops[2], values)
            fallback = mass < 1e-16
    if fallback:
        values, jacobian, _ = finite(original, positions, normals, x, nodes, weights)
        mass = total(numpy_function, loops[2], values)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("No optical support")
    return _normalized(values, jacobian, mass)
