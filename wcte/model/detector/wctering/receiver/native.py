"""One interface to reference and vector implementations of the original panels.

The reference finite kernel preserves V101's scalar libm arithmetic. The vector
kernel changes transcendental rounding while keeping panel/node order. The
workflow explicitly requests reference arithmetic for sensitive count retries.
Point-source calculations retain the original scalar maths in either library.
"""

import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from wctering.receiver import bridge
from wctering.receiver.build import ARRAY, FINITE_ARGUMENTS, load_library


def library(reference=False):
    """Canonicalize default and explicit flags before entering the two-slot cache."""
    return _library(bool(reference))


@lru_cache(maxsize=2)
def _library(reference):
    native = load_library(("point.c", "finite.c"), vector_math=not reference)
    if native is None:
        return None
    try:
        native.point_rays.argtypes = [
            ctypes.c_size_t,
            ARRAY,
            ARRAY,
            ARRAY,
            ARRAY,
            ARRAY,
        ]
        native.point_rays.restype = None
        native.finite_rays.argtypes = FINITE_ARGUMENTS
        native.finite_rays.restype = None
        from numba.extending import get_cython_function_address

        native.set_dot_function.argtypes = [ctypes.c_void_p]
        native.set_dot_function.restype = None
        native.set_dot_function(get_cython_function_address("scipy.linalg.cython_blas", "ddot"))
        # A length-three BLAS dot can have backend-specific rounding. Enable
        # the packed equivalent only after its independent native probe agrees.
        native.enable_packed_dot.argtypes = []
        native.enable_packed_dot.restype = ctypes.c_int
        if "libblis-mt.so.5.1.0" in Path("/proc/self/maps").read_text():
            native.enable_packed_dot()
        return native
    except (OSError, AttributeError):
        return None


@lru_cache(maxsize=2)
def bridge_functions(reference):
    native = library(reference)
    return (
        bridge.function_pointer(native.point_rays),
        bridge.function_pointer(native.finite_rays, finite=True),
    )


def integrate(positions, normals, parameters):
    """Unnormalized point light and its nine uniform-source derivatives."""
    return bridge.point(
        bridge_functions(False)[0],
        np.ascontiguousarray(positions),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(parameters),
    )


def finite_integrate(positions, normals, parameters, nodes, weights, *, reference=False):
    """Evaluate original distance panels with the selected native arithmetic."""
    if library(bool(reference)) is None:
        # The angular-panel wrapper can call this entry point directly on
        # macOS. Use the existing reference Numba kernel when x86 native
        # libraries are unavailable, just as finite_source.raw already does.
        from wctering.receiver.finite_source import integrate as fallback

        return fallback(positions, normals, parameters, nodes, weights)
    return bridge.finite(
        bridge_functions(bool(reference))[1],
        np.ascontiguousarray(positions),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(parameters),
        nodes,
        weights,
    )
