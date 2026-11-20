"""Preserve an independently supported third ring after a lossy aperture merge."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from measured_multiplicity_head import load, measured


@lru_cache(maxsize=4)
def read_package(path):
    models = tuple(sorted(path.glob("count_head_*.npz")))
    if not models:
        return (), None
    with np.load(path / "beam_clock.npz", allow_pickle=False) as data:
        t = data["beam_clock_ns"]
    return tuple(load(model) for model in models), t


def review(event, particle, previous, current, diagnostics, models):
    info = diagnostics.get("aperture_multiplicity", {})
    if (
        not info.get("applied")
        or info.get("previous_count") != 3
        or info.get("proposed_count") != 2
        or previous["fractions"].shape[1] != 4
        or current["fractions"].shape[1] != 3
    ):
        return current, None
    retained = info.get("old_core_retained", ())
    unclaimed = info.get("old_core_unexplained_pe", ())
    if not any(
        missed >= 20.0 or (kept <= 0.66 and missed >= 12.0)
        for kept, missed in zip(retained, unclaimed, strict=True)
    ):
        return current, None
    package, reference = read_package(Path(models))
    if not package:
        return current, None
    heads = np.stack([measured(event, particle, head, reference) for head in package])
    confidence = np.median(heads, axis=0)
    promote = bool(confidence[3] >= 0.66 and confidence[3] >= 2.0 * confidence[2])
    outcome = dict(
        applied=promote,
        probabilities=confidence.tolist(),
        old_core_retained=list(retained),
        unclaimed_pe=list(unclaimed),
        estimates=len(heads),
    )
    return (dict(current, fractions=previous["fractions"]) if promote else current), outcome
