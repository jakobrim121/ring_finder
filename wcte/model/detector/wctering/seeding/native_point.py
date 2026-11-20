"""Optional C implementation of the four-parameter timing fit.

The readable arithmetic lives in point_fit.c. A missing compiler or failed
load selects the NumPy fallback in multilateration.py. Compilation happens
once per source version, before event workers are forked.
"""

from pathlib import Path
import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def load_native():
    if os.environ.get("WCTE_ANISOTROPY_NATIVE", "1") == "0":
        return None
    source = Path(__file__).with_name("point_fit.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get("WCTE_CACHE_DIR", tempfile.gettempdir())) / "wctering"
    library = cache / f"point_fit_{digest}.so"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not library.exists():
            # Publish only a completed library; simultaneous imports cannot
            # observe a half-written compiler output.
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                built = Path(temporary) / "point_fit.so"
                subprocess.run(
                    [
                        os.environ.get("CC", "cc"),
                        "-shared",
                        "-fPIC",
                        "-O3",
                        "-o",
                        str(built),
                        str(source),
                        "-lm",
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                os.replace(built, library)
        native = ctypes.CDLL(str(library))
        native.bounded_lm.restype = None
        native.bounded_lm.argtypes = [
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            ctypes.c_double,
            DOUBLE_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
        ]
        return native
    except (OSError, subprocess.SubprocessError):
        return None


LIBRARY = load_native()


def solve(positions, times, sqrt_weights, light_speed, initial, max_iterations):
    arrays = [np.ascontiguousarray(a, np.float64) for a in (positions, times, sqrt_weights, initial)]
    pointers = [a.ctypes.data_as(DOUBLE_POINTER) for a in arrays]
    parameters, cost, hessian = np.empty(4), np.empty(1), np.empty((4, 4))
    LIBRARY.bounded_lm(
        *pointers[:3],
        float(light_speed),
        pointers[3],
        len(times),
        int(max_iterations),
        parameters.ctypes.data_as(DOUBLE_POINTER),
        cost.ctypes.data_as(DOUBLE_POINTER),
        hessian.ctypes.data_as(DOUBLE_POINTER),
    )
    return parameters, float(cost[0]), hessian, int(max_iterations), True
