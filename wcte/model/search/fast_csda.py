"""Evaluate the calibrated charged-track light for a proposed start."""

import ctypes
from functools import lru_cache

import numpy as np

from lookup import COEFFICIENTS, COUNT, LOW, STEP
from wctering.receiver.build import ARRAY, load_library


@lru_cache(maxsize=1)
def library():
    handle = load_library(("csda_batch.c",), vector_math=True)
    if handle is not None:
        handle.csda_batch.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            *([ARRAY] * 8),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ARRAY,
        ]
        handle.csda_batch.restype = None
    return handle


def integrate(positions, normals, x, us, weights, path, emission):
    native = library()
    if native is None:
        from integration import integrate as scalar

        return scalar(positions, normals, x, us, weights, path, emission)[0]

    points, axes, _, _ = path
    width = 12 * np.sqrt(emission[:, 6])
    cosines = np.column_stack(
        [
            np.cos(np.minimum(emission[:, 0] + width, np.pi)),
            np.cos(np.maximum(emission[:, 0] - width, 0.0)),
        ]
    )
    values = np.empty(len(positions))
    data = (positions, normals, weights, points, axes, emission, cosines, COEFFICIENTS)
    native.csda_batch(
        len(positions),
        len(us),
        *(np.ascontiguousarray(a, dtype=np.float64) for a in data),
        COUNT,
        LOW,
        STEP,
        values,
    )
    return values
