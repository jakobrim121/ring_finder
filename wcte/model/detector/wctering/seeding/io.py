"""WCTE mapping, geometry, and completed detector-NPZ input handling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wctering.seeding.config import RingFinderConfig, coerce_config
from wctering.seeding.detector import load_run_survey

PROMPT_PEAK_CHUNK_SIZE = 512
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_MAPPING_PATH = DATA_DIR / "wcsim_wcte_mapping.txt"
DEFAULT_GEOMETRY_PATH = DATA_DIR / "wcte_pmt_positions.csv"


@dataclass(frozen=True)
class EventHits:
    """Prepared detector-only hit arrays for one event."""

    event_id: int
    raw_pmt_ids: np.ndarray
    pmt_ids: np.ndarray
    pmt_positions_mm: np.ndarray
    charges_pe: np.ndarray
    times_ns: np.ndarray
    input_indices: np.ndarray
    input_accounting: Mapping[str, int]
    timing_selection: Mapping[str, Any]


def _charge_capped_kde_mode_ns(
    times_ns: np.ndarray,
    charges_pe: np.ndarray,
    *,
    bandwidth_ns: float,
    charge_cap_pe: float,
) -> float:
    """Return a deterministic Gaussian-KDE mode for a nonempty hit set.

    Individual PMT charge is capped while locating the time mode so an isolated
    high-charge afterpulse cannot outweigh a temporally coherent prompt cluster.
    The arrays are sorted first, making the earliest mode win an exact tie.
    """

    order = np.argsort(np.asarray(times_ns, dtype=float), kind="stable")
    time = np.asarray(times_ns, dtype=float)[order]
    weight = np.minimum(np.asarray(charges_pe, dtype=float)[order], float(charge_cap_pe))
    density = np.empty(len(time), dtype=float)
    chunk_size = max(1, int(PROMPT_PEAK_CHUNK_SIZE))
    for start in range(0, len(time), chunk_size):
        stop = min(start + chunk_size, len(time))
        scaled = (time[start:stop, None] - time[None, :]) / float(bandwidth_ns)
        density[start:stop] = np.exp(-0.5 * scaled**2) @ weight
    mode = float(time[int(np.argmax(density))])
    local = np.abs(time - mode) <= float(bandwidth_ns)
    return float(np.average(time[local], weights=weight[local]))


def prompt_time_mask(
    charges_pe: np.ndarray,
    times_ns: np.ndarray,
    *,
    config: RingFinderConfig | Mapping[str, Any] | None = None,
    window_ns: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the observed event's global prompt mode, with no absolute-clock prior.

    Legacy absolute search bounds remain in serialized configurations for
    compatibility, but do not restrict this policy's KDE candidates.
    """
    cfg = coerce_config(config)
    q = np.asarray(charges_pe, float)
    t = np.asarray(times_ns, float)
    if q.ndim != 1 or t.ndim != 1 or len(q) != len(t):
        raise ValueError("charge and time must be aligned one-dimensional arrays")
    valid = np.isfinite(q) & np.isfinite(t) & (q > 0)
    if window_ns is not None:
        low, high = map(float, window_ns)
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError("The supplied prompt time window must be finite and increasing")
        selected = valid & (t >= low) & (t < high)
        return selected, dict(
            policy="user_prompt_window", method="supplied time interval",
            valid=bool(selected.any()), prompt_peak_ns=None,
            prompt_window_ns=[low, high], interval_convention="lower inclusive, upper exclusive",
        )
    info = dict(
        method="event-relative capped-charge Gaussian KDE over all mapped active-PMT records",
        policy="global_observed_prompt_mode_v1",
        peak_search_window_ns=None,
        absolute_search_settings_applied=False,
        peak_search_fallback_to_all_times=False,
        peak_candidate_records=int(valid.sum()),
        kde_bandwidth_ns=float(cfg.prompt_peak_bandwidth_ns),
        per_record_charge_cap_pe=float(cfg.prompt_peak_charge_cap_pe),
        lower_offset_from_peak_ns=-float(cfg.prompt_pre_peak_ns),
        upper_offset_from_peak_ns=float(cfg.prompt_post_peak_ns),
        interval_convention="lower inclusive, upper exclusive",
    )
    if not valid.any():
        return np.zeros(len(t), bool), dict(
            info,
            valid=False,
            reason="no finite positive-charge active-PMT record",
            prompt_peak_ns=None,
            prompt_window_ns=None,
        )
    anchor = float(np.min(t[valid]))
    relative = t[valid] - anchor
    peak = _charge_capped_kde_mode_ns(
        relative,
        q[valid],
        bandwidth_ns=cfg.prompt_peak_bandwidth_ns,
        charge_cap_pe=cfg.prompt_peak_charge_cap_pe,
    )
    selected = np.zeros(len(t), bool)
    selected[valid] = (relative >= peak - cfg.prompt_pre_peak_ns) & (
        relative < peak + cfg.prompt_post_peak_ns
    )
    return selected, dict(
        info,
        valid=True,
        prompt_peak_ns=float(anchor + peak),
        prompt_window_ns=[
            float(anchor + peak - cfg.prompt_pre_peak_ns),
            float(anchor + peak + cfg.prompt_post_peak_ns),
        ],
    )


@lru_cache(maxsize=8)
def load_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[int, int]:
    """Map zero-based WCSim tube IDs to ``100*slot + zero-based PMT`` IDs."""

    table = np.atleast_2d(np.loadtxt(Path(path).resolve()))
    return {int(row[0]) - 1: int(row[1]) * 100 + int(row[2]) - 1 for row in table}


def active_wcte_ids() -> set[int]:
    """Use the same active channel population for hit cuts and light exposure."""
    survey = load_run_survey()
    return set(survey.id_to_index) if survey is not None else set(load_mapping().values())


@lru_cache(maxsize=8)
def load_pmt_positions(
    path: str | Path = DEFAULT_GEOMETRY_PATH,
) -> dict[int, np.ndarray]:
    """Load the frozen WCTE design-frame PMT centers in millimetres."""

    survey = load_run_survey() if path == DEFAULT_GEOMETRY_PATH else None
    if survey is not None:
        return {int(p): survey.positions_mm[i] for p, i in survey.id_to_index.items()}
    table = np.atleast_2d(np.loadtxt(Path(path).resolve(), delimiter=",", skiprows=1))
    return {int(row[0]): np.asarray(row[1:4], dtype=float) for row in table}


def prepare_event_arrays(
    event_id: int,
    raw_pmt_ids: np.ndarray,
    charges_pe: np.ndarray,
    times_ns: np.ndarray,
    *,
    mapping: Mapping[int, int] | None = None,
    positions: Mapping[int, np.ndarray] | None = None,
    config: RingFinderConfig | Mapping[str, Any] | None = None,
    mapped_pmt_ids: bool = False,
    prompt_window_ns: tuple[float, float] | None = None,
) -> EventHits:
    """Apply the exact peak-relative prompt, mapping, and geometry selection."""

    active_config = coerce_config(config)
    raw = np.asarray(raw_pmt_ids)
    charge = np.asarray(charges_pe, dtype=float)
    time = np.asarray(times_ns, dtype=float)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("digi_hit_pmt must be a one-dimensional integer array")
    if charge.ndim != 1 or time.ndim != 1:
        raise ValueError("charge and time must be one-dimensional")
    if not (len(raw) == len(charge) == len(time)):
        raise ValueError("PMT, charge, and time arrays must have equal lengths")

    positions = load_pmt_positions() if positions is None else positions
    if mapping is None:
        mapping = {pmt_id: pmt_id for pmt_id in positions} if mapped_pmt_ids else load_mapping()
    finite_positive = np.isfinite(charge) & np.isfinite(time) & (charge > 0.0)
    active_indices: list[int] = []
    active_mapped_ids: list[int] = []
    missing_mapping_all = 0
    inactive_geometry_all = 0
    for index in np.flatnonzero(finite_positive):
        mapped = mapping.get(int(raw[index]))
        if mapped is None:
            missing_mapping_all += 1
        elif int(mapped) not in positions:
            inactive_geometry_all += 1
        else:
            active_indices.append(int(index))
            active_mapped_ids.append(int(mapped))

    active = np.asarray(active_indices, dtype=int)
    active_ids = np.asarray(active_mapped_ids, dtype=np.int64)
    active_prompt, timing = prompt_time_mask(charge[active], time[active],
                                             config=active_config, window_ns=prompt_window_ns)
    kept = active[active_prompt]
    pmt_ids = active_ids[active_prompt]
    xyz = (
        np.asarray([positions[int(pmt)] for pmt in pmt_ids], dtype=float)
        if len(pmt_ids)
        else np.empty((0, 3), dtype=float)
    )

    prompt = np.zeros(len(raw), dtype=bool)
    window = timing.get("prompt_window_ns")
    if window is not None:
        low, high = map(float, window)
        prompt = finite_positive & (time >= low) & (time < high)
    missing_mapping_prompt = 0
    inactive_geometry_prompt = 0
    for index in np.flatnonzero(prompt):
        mapped = mapping.get(int(raw[index]))
        if mapped is None:
            missing_mapping_prompt += 1
        elif int(mapped) not in positions:
            inactive_geometry_prompt += 1

    return EventHits(
        event_id=int(event_id),
        raw_pmt_ids=np.asarray(raw[kept], dtype=np.int64),
        pmt_ids=pmt_ids,
        pmt_positions_mm=xyz,
        charges_pe=charge[kept],
        times_ns=time[kept],
        input_indices=kept,
        input_accounting={
            "digitized_records": int(len(raw)),
            "finite_positive_records_all_times": int(np.count_nonzero(finite_positive)),
            "usable_mapped_active_records_all_times": int(len(active)),
            "missing_mapping_all_time_records": int(missing_mapping_all),
            "inactive_geometry_all_time_records": int(inactive_geometry_all),
            "finite_positive_records_prompt_window": int(np.count_nonzero(prompt)),
            "missing_mapping_prompt_records": int(missing_mapping_prompt),
            "inactive_geometry_prompt_records": int(inactive_geometry_prompt),
            "usable_prompt_records": int(len(kept)),
            "usable_prompt_unique_pmts": int(len(np.unique(pmt_ids))),
        },
        timing_selection=timing,
    )
