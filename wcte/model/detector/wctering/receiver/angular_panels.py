"""Receiver-aware angular panels, with fallback for ill-conditioned sources.

This wrapper uses the quadrature order supplied by its caller. The occupancy
model uses it for initial order-16 fits, then certifies endpoints with the
original integral. Short, near-axis and almost-dark sources use the original
distance panels immediately. This is not an arithmetic-equivalence claim.
"""

import ctypes
from functools import lru_cache
import numpy as np
from wctering.receiver.build import FINITE_ARGUMENTS, load_library


@lru_cache(maxsize=1)
def library():
    native = load_library(("angular_panels.c",), vector_math=True)
    if native is not None:
        native.finite_rays.argtypes = FINITE_ARGUMENTS
        native.finite_rays.restype = ctypes.c_int
    return native


def finite_integrate(positions, normals, parameters, nodes, weights):
    from wctering.receiver.native import finite_integrate as original

    if library() is None:
        return original(positions, normals, parameters, nodes, weights)
    values = np.empty(len(positions))
    jacobian = np.empty((len(positions), 9))
    counts = np.empty(len(positions), dtype=np.int64)
    status = library().finite_rays(
        len(positions),
        np.ascontiguousarray(positions),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(parameters),
        len(nodes),
        nodes,
        weights,
        values,
        jacobian,
        counts,
    )
    if status or not np.isfinite(values).all() or values.sum() < 1e-16:
        return original(positions, normals, parameters, nodes, weights)
    return values, jacobian, counts
