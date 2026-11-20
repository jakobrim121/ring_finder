"""Compile the optional optical kernels once, without changing CPU threading.

All kernels use one CPU. Vector lanes process independent quadrature nodes;
they do not create worker threads. Unsupported platforms use the Python/Numba
implementations. Physics algebra keeps contraction disabled. The optional ordered-math helpers
use explicit FMA instructions only to reproduce the verified scalar library.
"""

import ctypes
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np

ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
INTEGERS = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
FINITE_ARGUMENTS = [
    ctypes.c_size_t,
    ARRAY,
    ARRAY,
    ARRAY,
    ctypes.c_size_t,
    ARRAY,
    ARRAY,
    ARRAY,
    ARRAY,
    INTEGERS,
]


@lru_cache(maxsize=1)
def verified_math_library():
    """Only use the SIMD adaptation with the exact independently tested libm.

    Another libc build can use different coefficients or contraction rules.
    Unknown libraries retain system scalar calls; a version string is not proof
    of arithmetic compatibility. This hash identifies software, never events.
    """
    expected = "2309ff027f58037e2be0604e93f23da016e7b2e7a96e62eaa02c892f5d6fc2b2"
    try:
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].endswith("/libm.so.6")
        }
        return len(paths) == 1 and hashlib.sha256(Path(paths.pop()).read_bytes()).hexdigest() == expected
    except OSError:
        return False


def load_library(names, *, vector_math):
    """Load a source/options-addressed binary, or return None for the fallback."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return None
    features = Path("/proc/cpuinfo").read_text()
    if "avx2" not in features:
        return None
    ordered = (
        ("point.c" in names or "angular_panels.c" in names) and "fma" in features and verified_math_library()
    )
    if ordered:
        names = (*names, "math_tables.c")
    sources = [Path(__file__).with_name(name) for name in names]
    # Source and included headers jointly define a cached native binary.
    headers = sorted(Path(__file__).parent.glob("*.h"))
    options = [
        "-O3",
        "-mavx2",
        "-ffp-contract=off",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-shared",
        "-fPIC",
    ]
    if ordered:
        options += ["-mfma", "-DORDERED_MATH=1"]
    if not vector_math:
        options += ["-DREFERENCE_MATH=1"]
    if (vector_math or ordered) and "avx512f" in features and "avx512dq" in features:
        options += ["-mavx512f", "-mavx512dq", "-DLANES=8", "-DPOINT_LANES=8"]
    # Optical-only override leaves timing seeding's compiler unchanged.
    # Different compilers must not silently share the same cached binary.
    compiler = os.environ.get("WCTE_OPTICS_CC", os.environ.get("CC", "cc"))
    identity = (
        b"".join(p.read_bytes() for p in [*sources, *headers])
        + repr((options, platform.libc_ver(), compiler)).encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:24]
    cache = Path(os.environ.get("WCTE_CACHE_DIR", tempfile.gettempdir())) / "wctering"
    output = cache / f"optical_{digest}.so"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            # Publish atomically: simultaneous independent processes must never
            # load another process's partly written shared library.
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                built = Path(temporary) / "optical.so"
                libraries = ["-lmvec", "-lm"] if vector_math else ["-lm"]
                subprocess.run(
                    [
                        compiler,
                        *options,
                        *map(str, sources),
                        "-o",
                        str(built),
                        *libraries,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                os.replace(built, output)
        return ctypes.CDLL(str(output))
    except (OSError, subprocess.SubprocessError):
        return None
