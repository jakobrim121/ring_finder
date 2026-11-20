"""Use NumPy's exact float64 loops from compiled likelihood bookkeeping.

This prototype is pinned to NumPy 1.26.4, like the fitter environment. It uses
the documented experimental capsule ABI, with a process-local lifetime owner.
No function address is compiled into Numba's on-disk cache. See the 1.26.4
_get_strided_loop docstring and numpy/core/src/umath/matmul.c.src for the ABI.
"""

import ctypes as ct
from functools import lru_cache
import numpy as np
from numba import njit
from wctering.receiver.build import load_library


class CallInfo(ct.Structure):
    _fields_ = [
        ("loop", ct.c_void_p),
        ("context", ct.c_void_p),
        ("auxdata", ct.c_void_p),
        ("requires_pyapi", ct.c_byte),
        ("no_floatingpoint_errors", ct.c_byte),
    ]


@lru_cache(maxsize=1)
def runtime():
    """Own capsules and library for as long as their pointers can be called."""
    if np.__version__ != "1.26.4":
        raise RuntimeError("The experimental NumPy loop bridge requires 1.26.4")
    native = load_library(("numpy_loops.c",), vector_math=False)
    if native is None:
        raise RuntimeError("NumPy loop bridge could not be compiled")
    function = ct.CFUNCTYPE(ct.c_int, *([ct.c_void_p] * 4))(
        ct.cast(native.call_numpy_loop, ct.c_void_p).value
    )
    get_pointer = ct.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ct.c_void_p
    get_pointer.argtypes = [ct.py_object, ct.c_char_p]
    capsules, pointers = [], []
    f8 = np.dtype("float64")
    for ufunc, strides, reduction in (
        (np.expm1, (8, 8), False),
        (np.log, (8, 8), False),
        (np.add, (0, 8, 0), True),
        (np.matmul, None, False),
        (np.exp, (8, 8), False),
        (np.log1p, (8, 8), False),
        (np.arccos, (8, 8), False),
    ):
        dtypes, capsule = ufunc._resolve_dtypes_and_context(
            (f8, f8, None) if reduction else (f8,) * ufunc.nargs, reduction=reduction
        )
        assert dtypes == (f8,) * ufunc.nargs
        ufunc._get_strided_loop(capsule, fixed_strides=strides)
        address = get_pointer(capsule, b"numpy_1.24_ufunc_call_info")
        info = ct.cast(address, ct.POINTER(CallInfo)).contents
        if info.requires_pyapi:
            raise RuntimeError("A numeric loop unexpectedly requires Python")
        pointers.append([info.loop, info.context, info.auxdata or 0])
        capsules.append(capsule)
    return function, np.asarray(pointers, dtype=np.uintp), capsules, native


@njit(cache=True)
def unary(function, info, values):
    """One original, contiguous NumPy unary loop."""
    result = np.empty(len(values))
    data = np.array([values.ctypes.data, result.ctypes.data], dtype=np.uintp)
    dimensions = np.array([len(values)], dtype=np.intp)
    strides = np.array([8, 8], dtype=np.intp)
    if function(info.ctypes, data.ctypes, dimensions.ctypes, strides.ctypes):
        raise RuntimeError("NumPy unary loop failed")
    return result


@njit(cache=True)
def total(function, info, values):
    """Preserve NumPy's actual reduction tree and positive-zero identity."""
    result = np.zeros(1)
    data = np.array([result.ctypes.data, values.ctypes.data, result.ctypes.data], dtype=np.uintp)
    dimensions = np.array([len(values)], dtype=np.intp)
    strides = np.array([0, 8, 0], dtype=np.intp)
    if function(info.ctypes, data.ctypes, dimensions.ctypes, strides.ctypes):
        raise RuntimeError("NumPy sum loop failed")
    return result[0]


@njit(cache=True)
def matmul(function, info, left, right):
    """Two-dimensional matmul using NumPy's own BLAS/layout dispatch.

    A vector becomes a one-row/one-column view before this call. Outer count
    is one; the next dimensions are (rows, contracted length, columns).
    The first three strides describe outer iteration, the next six the two
    core axes of each operand. Neither input is copied or transposed.
    """
    result = np.empty((left.shape[0], right.shape[1]))
    data = np.array([left.ctypes.data, right.ctypes.data, result.ctypes.data], dtype=np.uintp)
    dimensions = np.array([1, left.shape[0], left.shape[1], right.shape[1]], dtype=np.intp)
    strides = np.array(
        [
            0,
            0,
            0,
            left.strides[0],
            left.strides[1],
            right.strides[0],
            right.strides[1],
            result.strides[0],
            result.strides[1],
        ],
        dtype=np.intp,
    )
    if function(info.ctypes, data.ctypes, dimensions.ctypes, strides.ctypes):
        raise RuntimeError("NumPy matmul loop failed")
    return result
