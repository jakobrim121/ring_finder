"""Detector-only cone-support diagnostics for forced ring partitions.

The partitioners deliberately return a complete hard assignment so that the
point-source hypotheses can always be fit.  That optimization convention is
not, by itself, evidence that every assigned pulse belongs to a physical ring.
This module therefore constructs auditable broad-cone, local-continuity, and
joint background probes. The strict intersection supplies extra-ring support
metrics; a final label zero is reserved for pulses that fail both physical
views. The original hard proposal is left unchanged for timing fits and
notebook compatibility.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Mapping, Sequence

import numpy as np

from wctering.seeding.io import load_pmt_positions


CORE_CONE_BAND_DEG = 10.0
BROAD_CONE_BAND_DEG = 18.0
MODULE_NEIGHBOR_DISTANCE_MM = 650.0
MODULE_TIME_RESIDUAL_TOLERANCE_NS = 2.0
LIGHT_SPEED_MM_PER_NS = 299.792458


@lru_cache(maxsize=1)
def _fixed_module_centers_mm() -> dict[int, np.ndarray]:
    """Return design-frame centers for the frozen 19-PMT detector modules."""

    grouped: dict[int, list[np.ndarray]] = {}
    for pmt_id, position in load_pmt_positions().items():
        grouped.setdefault(int(pmt_id) // 100, []).append(np.asarray(position, dtype=float))
    return {
        module: np.mean(np.asarray(positions, dtype=float), axis=0) for module, positions in grouped.items()
    }


def _invalid(
    reason: str,
    n_hits: int,
    expected_ring_count: int,
    core_band_deg: float,
    broad_band_deg: float,
) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "expected_ring_count": int(expected_ring_count),
        "core_band_deg": float(core_band_deg),
        "broad_band_deg": float(broad_band_deg),
        "component_support": [],
        "minimum_core_assigned_charge_fraction": None,
        "minimum_broad_assigned_charge_fraction": None,
        "background_aware_partition_labels": np.zeros(n_hits, dtype=np.int64),
    }


def partition_cone_support_diagnostics(
    positions_mm: Sequence[Sequence[float]],
    charges_pe: Sequence[float],
    labels: Sequence[int],
    rings: Sequence[Mapping[str, Any]],
    expected_ring_count: int,
    *,
    core_band_deg: float = CORE_CONE_BAND_DEG,
    broad_band_deg: float = BROAD_CONE_BAND_DEG,
) -> dict[str, Any]:
    """Measure how much assigned raw charge actually supports each cone.

    A pulse supports a cone when its angular residual from that cone is within
    ``core_band_deg`` or ``broad_band_deg``.  The broad-band labels use zero as
    an explicit unassigned/background category.  They are a diagnostic view of
    the forced proposal; this function never mutates ``labels``.
    """

    positions = np.asarray(positions_mm, dtype=float)
    charges = np.asarray(charges_pe, dtype=float)
    hard_labels = np.asarray(labels, dtype=np.int64)
    n_hits = int(len(charges)) if charges.ndim == 1 else 0
    expected = int(expected_ring_count)

    if expected <= 0:
        return _invalid(
            "expected_ring_count must be positive",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )
    if not (positions.ndim == 2 and positions.shape == (n_hits, 3) and hard_labels.shape == (n_hits,)):
        return _invalid(
            "positions, charges, and labels are not aligned",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )
    if len(rings) != expected:
        return _invalid(
            "ring parameter count does not match expected_ring_count",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )
    if not (
        math.isfinite(float(core_band_deg))
        and math.isfinite(float(broad_band_deg))
        and 0.0 < float(core_band_deg) <= float(broad_band_deg)
    ):
        return _invalid(
            "cone bands must be finite, positive, and ordered",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )
    if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(charges)) and np.all(charges >= 0.0)):
        return _invalid(
            "positions and non-negative charges must be finite",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )
    positive_labels = sorted(set(map(int, hard_labels[hard_labels > 0])))
    required_labels = list(range(1, expected + 1))
    if positive_labels != required_labels or np.any(hard_labels <= 0):
        return _invalid(
            "forced proposal does not contain exactly the expected positive labels",
            n_hits,
            expected,
            core_band_deg,
            broad_band_deg,
        )

    background_labels = np.zeros(n_hits, dtype=np.int64)
    component_support: list[dict[str, Any]] = []
    for label, ring in enumerate(rings, start=1):
        if not isinstance(ring, Mapping):
            return _invalid(
                "ring parameters are unavailable",
                n_hits,
                expected,
                core_band_deg,
                broad_band_deg,
            )
        try:
            vertex = np.asarray(ring["vertex"], dtype=float)
            direction = np.asarray(ring["direction"], dtype=float)
            opening_deg = float(ring["theta_c"])
        except (KeyError, TypeError, ValueError):
            return _invalid(
                "ring parameters are incomplete",
                n_hits,
                expected,
                core_band_deg,
                broad_band_deg,
            )
        direction_norm = float(np.linalg.norm(direction))
        if not (
            vertex.shape == (3,)
            and direction.shape == (3,)
            and np.all(np.isfinite(vertex))
            and np.all(np.isfinite(direction))
            and math.isfinite(opening_deg)
            and direction_norm > 1.0e-12
        ):
            return _invalid(
                "ring parameters are not finite physical vectors",
                n_hits,
                expected,
                core_band_deg,
                broad_band_deg,
            )

        assigned = hard_labels == label
        assigned_charge = float(np.sum(charges[assigned]))
        vectors = positions[assigned] - vertex
        distances = np.linalg.norm(vectors, axis=1)
        if not np.any(assigned) or assigned_charge <= 0.0 or np.any(distances <= 1.0e-9):
            return _invalid(
                "a proposed component has no finite positive-charge support",
                n_hits,
                expected,
                core_band_deg,
                broad_band_deg,
            )

        # Measure angular distance to the cone surface, not distance to its axis.
        # Core and broad bands serve different support/background diagnostics.
        unit = vectors / distances[:, None]
        axis = direction / direction_norm
        angles_deg = np.degrees(np.arccos(np.clip(unit @ axis, -1.0, 1.0)))
        residual_deg = np.abs(angles_deg - opening_deg)
        core_local = residual_deg <= float(core_band_deg)
        broad_local = residual_deg <= float(broad_band_deg)
        assigned_indices = np.flatnonzero(assigned)
        background_labels[assigned_indices[broad_local]] = label
        core_charge = float(np.sum(charges[assigned][core_local]))
        broad_charge = float(np.sum(charges[assigned][broad_local]))
        component_support.append(
            {
                "ring_label": int(label),
                "assigned_hits": int(np.count_nonzero(assigned)),
                "assigned_charge_pe": assigned_charge,
                "core_support_hits": int(np.count_nonzero(core_local)),
                "core_support_charge_pe": core_charge,
                "core_assigned_charge_fraction": core_charge / assigned_charge,
                "broad_support_hits": int(np.count_nonzero(broad_local)),
                "broad_support_charge_pe": broad_charge,
                "broad_assigned_charge_fraction": broad_charge / assigned_charge,
                "maximum_assigned_angular_residual_deg": float(np.max(residual_deg)),
            }
        )

    total_charge = float(np.sum(charges))
    background = background_labels == 0
    return {
        "valid": True,
        "expected_ring_count": expected,
        "core_band_deg": float(core_band_deg),
        "broad_band_deg": float(broad_band_deg),
        "component_support": component_support,
        "minimum_core_assigned_charge_fraction": float(
            min(item["core_assigned_charge_fraction"] for item in component_support)
        ),
        "minimum_broad_assigned_charge_fraction": float(
            min(item["broad_assigned_charge_fraction"] for item in component_support)
        ),
        "background_aware_partition_labels": background_labels,
        "background_hits": int(np.count_nonzero(background)),
        "background_charge_pe": float(np.sum(charges[background])),
        "background_charge_fraction": (
            float(np.sum(charges[background])) / total_charge if total_charge > 0.0 else None
        ),
        "interpretation": (
            "label zero marks the part of the forced proposal lying outside "
            "the broad angular band; the original hard labels remain unchanged"
        ),
    }


def _invalid_continuity(
    reason: str,
    n_hits: int,
    expected_ring_count: int,
    module_neighbor_distance_mm: float,
    module_time_residual_tolerance_ns: float,
) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "expected_ring_count": int(expected_ring_count),
        "module_neighbor_distance_mm": float(module_neighbor_distance_mm),
        "module_time_residual_tolerance_ns": float(module_time_residual_tolerance_ns),
        "component_continuity": [],
        "minimum_neighbor_assigned_charge_fraction": None,
        "minimum_two_neighbor_assigned_charge_fraction": None,
        "minimum_largest_connected_charge_fraction": None,
        "continuity_aware_partition_labels": np.zeros(n_hits, dtype=np.int64),
    }


def _largest_connected_charge(
    adjacency: np.ndarray,
    module_charges: np.ndarray,
) -> tuple[float, int]:
    # Sum charge in each connected module component with a depth-first walk.
    # A bright isolated module must not masquerade as an extended connected ring.
    visited: set[int] = set()
    largest_charge = 0.0
    connected_components = 0
    for start in range(len(module_charges)):
        if start in visited:
            continue
        connected_components += 1
        visited.add(start)
        stack = [start]
        component_charge = 0.0
        while stack:
            current = stack.pop()
            component_charge += float(module_charges[current])
            for neighbor in np.flatnonzero(adjacency[current]):
                index = int(neighbor)
                if index not in visited:
                    visited.add(index)
                    stack.append(index)
        largest_charge = max(largest_charge, component_charge)
    return largest_charge, connected_components


def partition_spatiotemporal_continuity_diagnostics(
    positions_mm: Sequence[Sequence[float]],
    times_ns: Sequence[float],
    charges_pe: Sequence[float],
    pmt_ids: Sequence[int],
    labels: Sequence[int],
    fits: Sequence[Mapping[str, Any]],
    expected_ring_count: int,
    *,
    group_index: float,
    module_neighbor_distance_mm: float = MODULE_NEIGHBOR_DISTANCE_MM,
    module_time_residual_tolerance_ns: float = (MODULE_TIME_RESIDUAL_TOLERANCE_NS),
) -> dict[str, Any]:
    """Measure local detector continuity without using particle truth.

    Pulses are first aggregated into fixed 19-PMT detector modules.  Two active
    modules in the same proposed ring are adjacent when their design centers
    are within one detector-module pitch (including the wall/end-cap guard
    band) and their fitted propagation-time residuals agree within two nominal
    timing resolutions.  A physical extended trace should contain local path
    interiors, whereas a collection of isolated delta-ray/tail pulses does not.

    ``continuity_aware_partition_labels`` is an explicit unassigned-hit view:
    label zero marks pulses in modules with no spatial-and-time neighbor.  The
    forced hard proposal is not mutated.
    """

    positions = np.asarray(positions_mm, dtype=float)
    times = np.asarray(times_ns, dtype=float)
    charges = np.asarray(charges_pe, dtype=float)
    ids = np.asarray(pmt_ids, dtype=np.int64)
    hard_labels = np.asarray(labels, dtype=np.int64)
    n_hits = int(len(charges)) if charges.ndim == 1 else 0
    expected = int(expected_ring_count)
    invalid_arguments = (
        expected <= 0
        or not math.isfinite(float(group_index))
        or float(group_index) <= 0.0
        or not math.isfinite(float(module_neighbor_distance_mm))
        or float(module_neighbor_distance_mm) <= 0.0
        or not math.isfinite(float(module_time_residual_tolerance_ns))
        or float(module_time_residual_tolerance_ns) <= 0.0
    )
    if invalid_arguments:
        return _invalid_continuity(
            "continuity parameters must be finite and positive",
            n_hits,
            expected,
            module_neighbor_distance_mm,
            module_time_residual_tolerance_ns,
        )
    if not (
        positions.ndim == 2
        and positions.shape == (n_hits, 3)
        and times.shape == (n_hits,)
        and ids.shape == (n_hits,)
        and hard_labels.shape == (n_hits,)
    ):
        return _invalid_continuity(
            "positions, times, charges, PMT IDs, and labels are not aligned",
            n_hits,
            expected,
            module_neighbor_distance_mm,
            module_time_residual_tolerance_ns,
        )
    if len(fits) != expected:
        return _invalid_continuity(
            "point-source fit count does not match expected_ring_count",
            n_hits,
            expected,
            module_neighbor_distance_mm,
            module_time_residual_tolerance_ns,
        )
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(times))
        and np.all(np.isfinite(charges))
        and np.all(charges >= 0.0)
    ):
        return _invalid_continuity(
            "positions, times, and non-negative charges must be finite",
            n_hits,
            expected,
            module_neighbor_distance_mm,
            module_time_residual_tolerance_ns,
        )
    positive_labels = sorted(set(map(int, hard_labels[hard_labels > 0])))
    if positive_labels != list(range(1, expected + 1)) or np.any(hard_labels <= 0):
        return _invalid_continuity(
            "forced proposal does not contain exactly the expected positive labels",
            n_hits,
            expected,
            module_neighbor_distance_mm,
            module_time_residual_tolerance_ns,
        )

    fixed_centers = _fixed_module_centers_mm()
    continuity_labels = np.zeros(n_hits, dtype=np.int64)
    component_continuity: list[dict[str, Any]] = []
    for label, fit in enumerate(fits, start=1):
        if not isinstance(fit, Mapping):
            return _invalid_continuity(
                "point-source fit parameters are unavailable",
                n_hits,
                expected,
                module_neighbor_distance_mm,
                module_time_residual_tolerance_ns,
            )
        try:
            vertex = np.asarray(fit["vertex_mm"], dtype=float)
            t0_ns = float(fit["t0_ns"])
        except (KeyError, TypeError, ValueError):
            return _invalid_continuity(
                "point-source fit parameters are incomplete",
                n_hits,
                expected,
                module_neighbor_distance_mm,
                module_time_residual_tolerance_ns,
            )
        assigned = hard_labels == label
        assigned_charge = float(np.sum(charges[assigned]))
        if not (
            vertex.shape == (3,)
            and np.all(np.isfinite(vertex))
            and math.isfinite(t0_ns)
            and np.any(assigned)
            and assigned_charge > 0.0
        ):
            return _invalid_continuity(
                "a proposed component lacks finite fitted support",
                n_hits,
                expected,
                module_neighbor_distance_mm,
                module_time_residual_tolerance_ns,
            )

        assigned_indices = np.flatnonzero(assigned)
        component_positions = positions[assigned]
        component_times = times[assigned]
        component_charges = charges[assigned]
        component_ids = ids[assigned]
        component_modules = component_ids // 100
        propagation_ns = (
            float(group_index) * np.linalg.norm(component_positions - vertex, axis=1) / LIGHT_SPEED_MM_PER_NS
        )
        # Compare residual times after subtracting the component point-source
        # prediction; absolute arrival times would confuse geometry with delay.
        time_residuals = component_times - t0_ns - propagation_ns
        unique_modules = np.unique(component_modules)
        module_charges: list[float] = []
        module_residuals: list[float] = []
        module_centers: list[np.ndarray] = []
        # Aggregate within a module so its 19 PMTs do not count as 19 independent
        # neighboring detector locations. Use fixed design centers when known.
        for module in unique_modules:
            in_module = component_modules == module
            module_charge = float(np.sum(component_charges[in_module]))
            module_charges.append(module_charge)
            module_residuals.append(
                float(
                    np.average(
                        time_residuals[in_module],
                        weights=component_charges[in_module],
                    )
                )
            )
            fixed = fixed_centers.get(int(module))
            module_centers.append(
                np.asarray(fixed, dtype=float)
                if fixed is not None
                else np.average(
                    component_positions[in_module],
                    axis=0,
                    weights=component_charges[in_module],
                )
            )

        module_charge_array = np.asarray(module_charges, dtype=float)
        module_residual_array = np.asarray(module_residuals, dtype=float)
        module_center_array = np.asarray(module_centers, dtype=float)
        spatial_distance = np.linalg.norm(
            module_center_array[:, None, :] - module_center_array[None, :, :],
            axis=2,
        )
        residual_difference = np.abs(module_residual_array[:, None] - module_residual_array[None, :])
        # An edge requires both nearby geometry and compatible residual time.
        # Remove self-edges before counting one- and two-neighbor support.
        adjacency = (spatial_distance <= float(module_neighbor_distance_mm)) & (
            residual_difference <= float(module_time_residual_tolerance_ns)
        )
        np.fill_diagonal(adjacency, False)
        degree = np.sum(adjacency, axis=1)
        neighbor_modules = degree >= 1
        two_neighbor_modules = degree >= 2
        largest_charge, connected_components = _largest_connected_charge(adjacency, module_charge_array)
        neighbor_charge = float(np.sum(module_charge_array[neighbor_modules]))
        two_neighbor_charge = float(np.sum(module_charge_array[two_neighbor_modules]))
        retained_modules = unique_modules[neighbor_modules]
        retained_local = np.isin(component_modules, retained_modules)
        continuity_labels[assigned_indices[retained_local]] = label
        retained_ids = component_ids[retained_local]
        component_continuity.append(
            {
                "ring_label": int(label),
                "assigned_hits": int(np.count_nonzero(assigned)),
                "assigned_unique_pmts": int(len(np.unique(component_ids))),
                "assigned_modules": int(len(unique_modules)),
                "assigned_charge_pe": assigned_charge,
                "modules_with_neighbor": int(np.count_nonzero(neighbor_modules)),
                "modules_with_two_neighbors": int(np.count_nonzero(two_neighbor_modules)),
                "neighbor_charge_pe": neighbor_charge,
                "neighbor_assigned_charge_fraction": (neighbor_charge / assigned_charge),
                "two_neighbor_charge_pe": two_neighbor_charge,
                "two_neighbor_assigned_charge_fraction": (two_neighbor_charge / assigned_charge),
                "largest_connected_charge_pe": float(largest_charge),
                "largest_connected_assigned_charge_fraction": (float(largest_charge) / assigned_charge),
                "connected_component_count": int(connected_components),
                "continuity_retained_hits": int(np.count_nonzero(retained_local)),
                "continuity_retained_unique_pmts": int(len(np.unique(retained_ids))),
                "propagation_time_residual_rms_ns": float(
                    np.sqrt(
                        np.average(
                            time_residuals**2,
                            weights=component_charges,
                        )
                    )
                ),
            }
        )

    total_charge = float(np.sum(charges))
    background = continuity_labels == 0
    return {
        "valid": True,
        "expected_ring_count": expected,
        "module_id_rule": "mapped PMT ID // 100",
        "module_neighbor_distance_mm": float(module_neighbor_distance_mm),
        "module_time_residual_tolerance_ns": float(module_time_residual_tolerance_ns),
        "group_index": float(group_index),
        "component_continuity": component_continuity,
        "minimum_neighbor_assigned_charge_fraction": float(
            min(item["neighbor_assigned_charge_fraction"] for item in component_continuity)
        ),
        "minimum_two_neighbor_assigned_charge_fraction": float(
            min(item["two_neighbor_assigned_charge_fraction"] for item in component_continuity)
        ),
        "minimum_largest_connected_charge_fraction": float(
            min(item["largest_connected_assigned_charge_fraction"] for item in component_continuity)
        ),
        "minimum_continuity_retained_unique_pmts": int(
            min(item["continuity_retained_unique_pmts"] for item in component_continuity)
        ),
        "continuity_aware_partition_labels": continuity_labels,
        "isolated_background_hits": int(np.count_nonzero(background)),
        "isolated_background_charge_pe": float(np.sum(charges[background])),
        "isolated_background_charge_fraction": (
            float(np.sum(charges[background])) / total_charge if total_charge > 0.0 else None
        ),
        "interpretation": (
            "label zero marks modules with no spatially adjacent, fitted-time-"
            "compatible neighbor; the original hard labels remain unchanged"
        ),
    }


def partition_background_cleanup_diagnostics(
    charges_pe: Sequence[float],
    pmt_ids: Sequence[int],
    labels: Sequence[int],
    cone_support: Mapping[str, Any],
    continuity: Mapping[str, Any],
    expected_ring_count: int,
) -> dict[str, Any]:
    """Build strict support diagnostics and conservative background labels.

    The strict intersection is used only to decide whether an extra fitted ring
    has enough jointly supported charge.  The returned background view is more
    conservative: a pulse is label-zero only when it fails both the broad-cone
    view and the same-label fitted-time detector-module continuity view.  Thus
    connected tail light and sparse on-cone light remain assigned.  The forced
    proposal remains available unchanged for fitting and audit purposes.
    """

    charges = np.asarray(charges_pe, dtype=float)
    ids = np.asarray(pmt_ids, dtype=np.int64)
    hard = np.asarray(labels, dtype=np.int64)
    n_hits = int(len(charges)) if charges.ndim == 1 else 0
    expected = int(expected_ring_count)
    common = {
        "valid": False,
        "expected_ring_count": expected,
        "combined_partition_labels": np.zeros(n_hits, dtype=np.int64),
        "strict_intersection_partition_labels": np.zeros(n_hits, dtype=np.int64),
        "background_partition_labels": np.zeros(n_hits, dtype=np.int64),
        "component_cleanup": [],
        "minimum_retained_charge_pe": None,
        "minimum_retained_capped_charge_pe": None,
        "minimum_retained_unique_pmts": None,
    }
    if not (
        expected > 0
        and charges.shape == (n_hits,)
        and ids.shape == (n_hits,)
        and hard.shape == (n_hits,)
        and np.all(np.isfinite(charges))
        and np.all(charges >= 0.0)
    ):
        return {**common, "reason": "cleanup inputs are invalid or unaligned"}
    if sorted(set(map(int, hard[hard > 0]))) != list(range(1, expected + 1)):
        return {**common, "reason": "forced proposal labels are incomplete"}
    if not bool(cone_support.get("valid", False)):
        return {**common, "reason": "broad-cone diagnostics are unavailable"}
    if not bool(continuity.get("valid", False)):
        return {**common, "reason": "continuity diagnostics are unavailable"}
    cone = np.asarray(
        cone_support.get("background_aware_partition_labels", []),
        dtype=np.int64,
    )
    path = np.asarray(
        continuity.get("continuity_aware_partition_labels", []),
        dtype=np.int64,
    )
    if cone.shape != (n_hits,) or path.shape != (n_hits,):
        return {**common, "reason": "diagnostic label arrays are unaligned"}
    if not np.all(np.isin(cone, range(expected + 1))) or not np.all(np.isin(path, range(expected + 1))):
        return {**common, "reason": "diagnostic labels are outside the expected range"}

    # Intersection measures strong extra-ring support; union preserves a pulse
    # in the returned partition if either physical diagnostic supports it.
    strict = np.where((cone == hard) & (path == hard), hard, 0).astype(np.int64)
    background_labels = np.where((cone == hard) | (path == hard), hard, 0).astype(np.int64)
    capped = np.minimum(charges, 2.0)
    component_cleanup = []
    for label in range(1, expected + 1):
        assigned = hard == label
        retained = strict == label
        assigned_charge = float(np.sum(charges[assigned]))
        retained_charge = float(np.sum(charges[retained]))
        assigned_capped = float(np.sum(capped[assigned]))
        retained_capped = float(np.sum(capped[retained]))
        component_cleanup.append(
            {
                "ring_label": label,
                "assigned_hits": int(np.count_nonzero(assigned)),
                "retained_hits": int(np.count_nonzero(retained)),
                "retained_unique_pmts": int(len(np.unique(ids[retained]))),
                "assigned_charge_pe": assigned_charge,
                "retained_charge_pe": retained_charge,
                "retained_assigned_charge_fraction": (
                    retained_charge / assigned_charge if assigned_charge > 0.0 else 0.0
                ),
                "assigned_capped_charge_pe": assigned_capped,
                "retained_capped_charge_pe": retained_capped,
                "retained_assigned_capped_charge_fraction": (
                    retained_capped / assigned_capped if assigned_capped > 0.0 else 0.0
                ),
            }
        )
    strict_background = strict == 0
    background = background_labels == 0
    total_charge = float(np.sum(charges))
    total_capped = float(np.sum(capped))
    return {
        "valid": True,
        "expected_ring_count": expected,
        # ``combined_partition_labels`` is retained as the v2.4 guard-facing
        # compatibility name; it is intentionally the strict intersection.
        "combined_partition_labels": strict,
        "strict_intersection_partition_labels": strict,
        "background_partition_labels": background_labels,
        "component_cleanup": component_cleanup,
        "minimum_retained_charge_pe": float(min(item["retained_charge_pe"] for item in component_cleanup)),
        "minimum_retained_capped_charge_pe": float(
            min(item["retained_capped_charge_pe"] for item in component_cleanup)
        ),
        "minimum_retained_unique_pmts": int(min(item["retained_unique_pmts"] for item in component_cleanup)),
        "background_hits": int(np.count_nonzero(background)),
        "background_charge_pe": float(np.sum(charges[background])),
        "background_charge_fraction": (
            float(np.sum(charges[background])) / total_charge if total_charge > 0.0 else None
        ),
        "background_capped_charge_pe": float(np.sum(capped[background])),
        "background_capped_charge_fraction": (
            float(np.sum(capped[background])) / total_capped if total_capped > 0.0 else None
        ),
        "strict_background_hits": int(np.count_nonzero(strict_background)),
        "strict_background_charge_pe": float(np.sum(charges[strict_background])),
        "strict_background_charge_fraction": (
            float(np.sum(charges[strict_background])) / total_charge if total_charge > 0.0 else None
        ),
        "decision_rule": (
            "use the intersection of the 18-degree cone view and local path "
            "continuity to test extra-ring support; return label zero only "
            "when a pulse fails both views, preserving connected tail light "
            "and sparse on-cone light"
        ),
    }
