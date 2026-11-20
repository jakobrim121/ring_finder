"""Load detector positions and infer PMT-facing normals from each module sphere.

Positions and module centers are in millimetres. This is geometry loading only;
the old, unused 6,265-line physical-likelihood collection has been removed."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectorGeometry:
    pmt_ids: np.ndarray
    positions_mm: np.ndarray
    inward_normals: np.ndarray
    id_to_index: dict[int, int]
    module_centers_mm: dict[int, np.ndarray]

    @classmethod
    def from_csv(cls, path: Path) -> "DetectorGeometry":
        table = np.genfromtxt(path, delimiter=",", names=True)
        ids = np.asarray(table["pmt_id"], dtype=np.int64)
        positions = np.column_stack([table["x_mm"], table["y_mm"], table["z_mm"]]).astype(float)
        normals = np.zeros_like(positions)
        centers: dict[int, np.ndarray] = {}
        for module in sorted(set(map(int, ids // 100))):
            selected = ids // 100 == module
            points = positions[selected]
            # |x-c|^2=r^2 -> 2*x.c + (r^2-|c|^2)=|x|^2.
            design = np.column_stack([2.0 * points, np.ones(len(points))])
            solution, *_ = np.linalg.lstsq(design, np.sum(points * points, axis=1), rcond=None)
            center = np.asarray(solution[:3], dtype=float)
            direction = points - center
            direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1.0e-12)
            centers[module] = center
            normals[selected] = direction
        return cls(
            pmt_ids=ids,
            positions_mm=positions,
            inward_normals=normals,
            id_to_index={int(value): index for index, value in enumerate(ids)},
            module_centers_mm=centers,
        )


@lru_cache(maxsize=1)
def load_run_survey() -> DetectorGeometry | None:
    """Load an explicitly selected survey once, before any fits in this process."""
    path = os.environ.get("WCTE_SURVEY")
    if not path:
        return None
    with np.load(path, allow_pickle=False) as survey:
        ids = survey["pmt_ids"]
        positions = survey["positions_mm"]
        normals = survey["inward_normals"]
        slots = survey["module_slots"]
        centres = survey["module_centers_mm"]
    if (
        ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or positions.shape != (len(ids), 3) or normals.shape != positions.shape
        or slots.ndim != 1 or slots.dtype.kind not in "iu"
        or centres.shape != (len(slots), 3) or len(np.unique(slots)) != len(slots)
        or not np.isin(ids // 100, slots).all()
        or not all(np.isfinite(a).all() for a in (positions, normals, centres))
    ):
        raise ValueError(f"Invalid WCTE survey geometry in {path}")
    return DetectorGeometry(ids, positions, normals, {int(p): i for i, p in enumerate(ids)},
                            dict(zip(map(int, slots), centres, strict=True)))


@lru_cache(maxsize=1)
def load_wcte_physical_geometry() -> DetectorGeometry:
    """Use the selected run survey or the packaged WCSim detector."""

    return load_run_survey() or DetectorGeometry.from_csv(
        Path(__file__).resolve().parent / "data" / "wcte_pmt_positions.csv"
    )
