"""Allocate native outputs and pass array pointers in one compiled wrapper.

The function pointer is a runtime argument, never a cached process address.
The loaded library owns the unchanged physics and remains alive in native.py.
"""

import ctypes
import numpy as np
from numba import njit


def function_pointer(function, finite=False):
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    arguments = (
        [
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
        if finite
        else [size] + [pointer] * 5
    )
    prototype = ctypes.CFUNCTYPE(None, *arguments)
    return prototype(ctypes.cast(function, pointer).value)


@njit(cache=True)
def point(function, positions, normals, parameters):
    values = np.empty(len(positions))
    jacobian = np.empty((len(positions), 9))
    function(
        len(positions),
        positions.ctypes,
        normals.ctypes,
        parameters.ctypes,
        values.ctypes,
        jacobian.ctypes,
    )
    return values, jacobian


@njit(cache=True)
def finite(function, positions, normals, parameters, nodes, weights):
    values = np.empty(len(positions))
    jacobian = np.empty((len(positions), 9))
    counts = np.empty(len(positions), dtype=np.int64)
    function(
        len(positions),
        positions.ctypes,
        normals.ctypes,
        parameters.ctypes,
        len(nodes),
        nodes.ctypes,
        weights.ctypes,
        values.ctypes,
        jacobian.ctypes,
        counts.ctypes,
    )
    return values, jacobian, counts
