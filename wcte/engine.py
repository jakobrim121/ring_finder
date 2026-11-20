"""Connect the detector, optical likelihood and finite-track search."""

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


PACKAGE = Path(__file__).resolve().parent
MODEL = PACKAGE / "model"
DATA = PACKAGE / "data"


def limit_threads():
    """Use one numerical thread in each worker process."""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


@lru_cache(maxsize=1)
def initialize():
    """Initialize read-only fit resources and a versioned local compiler cache."""
    limit_threads()
    from llvmlite.binding import get_host_cpu_features, get_host_cpu_name
    import numba

    # Numba embeds calibration constants. Cache entries must also depend on
    # the source and data that generated those constants.
    identity = {
        "model": sha256((MODEL / "manifest.json").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": get_host_cpu_name(),
        "features": get_host_cpu_features().flatten(),
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "numba", "llvmlite", "threadpoolctl")
        },
    }
    # Compiled geometry constants are specific to the run's active PMT survey.
    if survey := os.environ.get("WCTE_SURVEY"):
        identity["survey"] = sha256(Path(survey).read_bytes()).hexdigest()
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root = Path(os.environ.get("WCTE_NUMBA_CACHE", Path.home() / ".cache" / "wcte"))
    cache = cache_root / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    numba.config.CACHE_DIR = str(cache)

    # Retain the qualified import order for the shared numerical kernels.
    sys.path[:0] = [str(MODEL / path) for path in ("search", "detector", "calibrated_response")]
    sys.path.insert(0, str(PACKAGE / "reviews"))
    track_models = PACKAGE / "track_models"
    sys.path.append(str(track_models))
    import compact_search

    return compact_search


def fit_base(raw, event_id, *, use_water_chord=True, early_candidate=None):
    """Fit the first hypothesis and return the hits and seeds for later reviews."""
    return initialize().fit_event(
        raw,
        event_id,
        use_water_chord=use_water_chord,
        early_candidate=early_candidate,
    )


def review_continuation(prepared, prediction, detail, particle):
    """Retain a single incoming track or resolve a continuation with timed light."""
    import numpy as np
    from beam_backtrack import experiment

    count = prediction["fractions"].shape[1] - 1
    if count > 1:
        return prediction, detail, {"applied": False, "reason": "Outside zero- and one-ring scope"}
    try:
        if count == 0:
            from zero_primary import experiment as try_primary

            report, choices = try_primary(prepared, prediction, detail, particle)
            report["final_comparison"] = "finite_primary"
            shares = choices.get("zero_primary")
        else:
            report, choices = experiment(prepared, prediction, detail, particle)
            if prepared["detail"].get("chord") is not None and not choices:
                report["fine_chord"], choices = experiment(
                    prepared,
                    prediction,
                    detail,
                    particle,
                    fine_start=True,
                )
                report["final_comparison"] = "fine_chord"
            else:
                report["final_comparison"] = "bounded_continuation"
            shares = choices.get("backtrack")
        if shares is None:
            return prediction, detail, dict(report, applied=False)
        if not (
            np.isfinite(shares).all()
            and np.min(shares, initial=0.0) >= -1e-14
            and np.allclose(shares.sum(1), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ArithmeticError("Candidate charge shares are invalid")
        return dict(prediction, fractions=shares), detail, dict(report, applied=True)
    except (ArithmeticError, ValueError) as error:
        return prediction, detail, dict(applied=False, candidate_failure=str(error))
