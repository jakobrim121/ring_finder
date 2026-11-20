"""Optionally accelerate finite optical integration on Apple Silicon."""

from contextlib import contextmanager
import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np


@lru_cache(maxsize=1)
def library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None, None, "The native optics kernel requires macOS on Apple Silicon."

    from wctering.receiver import native

    source = Path(native.__file__).with_name("finite.c")
    try:
        default = Path(
            subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True, stderr=subprocess.PIPE, timeout=15
            ).strip()
        )
        compiler = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--find", "clang"], text=True, stderr=subprocess.PIPE, timeout=15
        ).strip()
        tested = default.with_name("MacOSX15.4.sdk")
        sdks = (tested, default, *sorted(default.parent.glob("MacOSX*.sdk"), reverse=True))
        with tempfile.TemporaryDirectory(prefix="wcte-native-optics-") as temp:
            target = Path(temp) / "finite.dylib"
            for sdk in dict.fromkeys(path.resolve() for path in sdks if path.is_dir()):
                result = subprocess.run(
                    [
                        compiler,
                        "-O3",
                        "-ffp-contract=off",
                        "-fno-math-errno",
                        "-fno-trapping-math",
                        "-DREFERENCE_MATH=1",
                        "-dynamiclib",
                        "-isysroot",
                        str(sdk),
                        str(source),
                        "-o",
                        str(target),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    handle = ctypes.CDLL(str(target))
                    break
            else:
                return None, None, "The installed macOS SDKs could not link native optics."
    except (OSError, subprocess.SubprocessError) as error:
        return None, None, f"Could not load native optics ({type(error).__name__})."

    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    handle.finite_rays.argtypes = [
        ctypes.c_size_t,
        array,
        array,
        array,
        ctypes.c_size_t,
        array,
        array,
        array,
        array,
        integers,
    ]
    handle.finite_rays.restype = None
    return handle, sdk.name, None


@contextmanager
def enabled():
    handle, sdk, reason = library()
    if handle is None:
        yield dict(active=False, reason=reason)
        return
    from wctering.receiver import finite_source, native
    from wctering.receiver.cache import event_cache

    previous_raw, previous_finite = finite_source.raw, native.finite_integrate

    def integrate(positions, normals, parameters, nodes, weights, *, reference=False):
        positions, normals, parameters, nodes, weights = (
            np.ascontiguousarray(value, dtype=np.float64)
            for value in (positions, normals, parameters, nodes, weights)
        )
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or normals.shape != positions.shape
            or parameters.shape != (9,)
            or nodes.ndim != 1
            or weights.shape != nodes.shape
            or not len(nodes)
        ):
            return previous_finite(positions, normals, parameters, nodes, weights, reference=reference)
        values = np.empty(len(positions))
        jacobian = np.empty((len(positions), 9))
        counts = np.empty(len(positions), dtype=np.int64)
        handle.finite_rays(
            len(positions),
            positions,
            normals,
            parameters,
            len(nodes),
            nodes,
            weights,
            values,
            jacobian,
            counts,
        )
        return values, jacobian, counts

    def raw(positions, normals, parameters, order=16, *, scalar=False):
        x = np.asarray(parameters, dtype=float)
        if x.shape != (9,) or not np.isfinite(x).all() or x[6] < 0.0 or np.linalg.norm(x[3:6]) < 1e-10:
            return previous_raw(positions, normals, parameters, order, scalar=scalar)
        return integrate(positions, normals, x, *finite_source.quadrature(order), reference=scalar)

    finite_source.raw = raw
    native.finite_integrate = integrate
    try:
        with event_cache(64):
            yield dict(active=True, backend="macos_arm64", sdk=sdk, scalar_transcendentals=True)
    finally:
        finite_source.raw = previous_raw
        native.finite_integrate = previous_finite
