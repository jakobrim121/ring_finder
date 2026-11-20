"""Frozen complete-mPMT fold assignment used by the robustness test."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_FOLD_PATH = PACKAGE_ROOT / "data/module_folds.json"


@lru_cache(maxsize=4)
def load_module_fold_plan(path: str | Path = DEFAULT_FOLD_PATH) -> dict[str, Any]:
    plan = json.loads(Path(path).resolve().read_text())
    if int(plan["n_folds"]) != 4:
        raise ValueError("this release requires exactly four module folds")
    plan["fold_by_module_id"] = {int(module): int(fold) for module, fold in plan["fold_by_module_id"].items()}
    return plan


def fold_ids_for_hits(pmt_ids: np.ndarray, path: str | Path = DEFAULT_FOLD_PATH) -> np.ndarray:
    plan = load_module_fold_plan(path)
    mapping = plan["fold_by_module_id"]
    modules = np.asarray(pmt_ids, dtype=np.int64) // 100
    missing = sorted(set(map(int, modules)).difference(mapping))
    if missing:
        raise KeyError(f"hit modules are absent from the frozen fold plan: {missing}")
    return np.asarray([mapping[int(module)] for module in modules], dtype=np.int64)
