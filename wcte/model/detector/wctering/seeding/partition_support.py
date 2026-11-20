"""Observable component support and conservative background assignment.

A hit is removed by the background policy only when both cone compatibility
and local space/time continuity reject it. These are seed diagnostics; the
later physical light fit supplies the final fractional charge partition.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from wctering.seeding.config import RingFinderConfig
from wctering.seeding.cone_seeds import fit_dominant_ring
from wctering.seeding.partition_quality import (
    partition_background_cleanup_diagnostics,
    partition_cone_support_diagnostics,
    partition_spatiotemporal_continuity_diagnostics,
)

GROUP_INDEX = 1.388


def _two_ring_observable_support_gate(
    hypotheses: Mapping[str, Any] | None,
    prediction_before_support: int,
    config: RingFinderConfig,
) -> dict[str, Any]:
    """Require full-event reconstructed PMT support for an H2 claim.

    The ten-PMT partition floor remains an optimization/fit-validity guard and
    is deliberately retained in the module-fold refits.  This separate final
    gate applies the configured observable-ring floor only to the complete
    detector event, so withholding one module fold cannot redefine whether a
    physical component is observable.
    """

    prediction = int(prediction_before_support)
    if prediction not in (1, 2):
        return {
            "criterion_applied": False,
            "support_passed": prediction != 2,
            "two_ring_claim_vetoed": False,
            "prediction_before_support": prediction,
            "prediction_after_support": prediction,
            "minimum_required_unique_pmts_per_component": int(config.minimum_observable_ring_pmts),
            "reason": "the H2 support gate applies only to class-one/two events",
        }
    value = None
    if isinstance(hypotheses, Mapping):
        value = hypotheses.get("minimum_partition_unique_pmts")
    try:
        minimum_pmts = int(value)
    except (TypeError, ValueError):
        minimum_pmts = None
    if prediction != 2:
        return {
            "criterion_applied": False,
            "support_passed": True,
            "two_ring_claim_vetoed": False,
            "prediction_before_support": prediction,
            "prediction_after_support": prediction,
            "minimum_partition_unique_pmts": minimum_pmts,
            "minimum_required_unique_pmts_per_component": int(config.minimum_observable_ring_pmts),
            "reason": "no two-ring claim survived the H1/H2 decision",
        }
    if minimum_pmts is None:
        return {
            "criterion_applied": False,
            "support_passed": True,
            "two_ring_claim_vetoed": False,
            "prediction_before_support": 2,
            "prediction_after_support": 2,
            "minimum_partition_unique_pmts": None,
            "minimum_required_unique_pmts_per_component": int(config.minimum_observable_ring_pmts),
            "reason": "full-event reconstructed component support is unavailable",
        }
    passed = bool(minimum_pmts >= int(config.minimum_observable_ring_pmts))
    return {
        "criterion_applied": True,
        "support_passed": passed,
        "two_ring_claim_vetoed": bool(not passed),
        "prediction_before_support": 2,
        "prediction_after_support": 2 if passed else 1,
        "minimum_partition_unique_pmts": minimum_pmts,
        "minimum_required_unique_pmts_per_component": int(config.minimum_observable_ring_pmts),
        "decision_rule": (
            "retain a two-ring claim only when each full-event reconstructed "
            "component contains at least minimum_observable_ring_pmts distinct PMTs"
        ),
    }


def _final_partition_background_assignment(
    payload: Mapping[str, Any], final_prediction: int
) -> dict[str, Any]:
    """Select the aligned background-cleaned partition for a final class."""

    final_prediction = int(final_prediction)
    full_hypotheses = payload.get("full_event_hypotheses")
    final_cleanup: Mapping[str, Any] | None = None
    cleanup_source = None
    if final_prediction == 1 and isinstance(full_hypotheses, Mapping):
        candidate = full_hypotheses.get("h1_partition_background_cleanup")
        if isinstance(candidate, Mapping):
            final_cleanup = candidate
            cleanup_source = "H1 dominant-cone plus local continuity"
    elif final_prediction == 2 and isinstance(full_hypotheses, Mapping):
        candidate = full_hypotheses.get("partition_background_cleanup")
        if isinstance(candidate, Mapping):
            final_cleanup = candidate
            cleanup_source = "H2 forced cones plus local continuity"
    elif final_prediction == 3:
        three_test = payload.get("three_ring_hypothesis_test")
        three_full = three_test.get("full_event_hypotheses") if isinstance(three_test, Mapping) else None
        candidate = (
            three_full.get("partition_background_cleanup") if isinstance(three_full, Mapping) else None
        )
        if isinstance(candidate, Mapping):
            final_cleanup = candidate
            cleanup_source = "H3 forced cones plus local continuity"

    raw_final_labels = (
        final_cleanup.get("background_partition_labels", [])
        if isinstance(final_cleanup, Mapping) and final_cleanup.get("valid", False)
        else []
    )
    final_labels = np.asarray(raw_final_labels, dtype=np.int64)
    expected_length = int(payload.get("usable_hits", len(final_labels)))
    expected_positive = list(range(1, final_prediction + 1))
    apply_background_assignment = bool(
        final_prediction > 0
        and final_labels.shape == (expected_length,)
        and np.all(np.isin(final_labels, range(final_prediction + 1)))
        and sorted(set(map(int, final_labels[final_labels > 0]))) == expected_positive
    )
    background_hits = int(np.count_nonzero(final_labels == 0)) if apply_background_assignment else None
    return {
        "criterion_applied": bool(final_prediction > 0),
        "background_assignment_applied": apply_background_assignment,
        "predicted_ring_class": final_prediction,
        "cleanup_source": cleanup_source,
        "partition_labels": final_labels if apply_background_assignment else [],
        "background_hits": background_hits,
        "background_charge_pe": (
            final_cleanup.get("background_charge_pe")
            if apply_background_assignment and isinstance(final_cleanup, Mapping)
            else None
        ),
        "background_charge_fraction": (
            final_cleanup.get("background_charge_fraction")
            if apply_background_assignment and isinstance(final_cleanup, Mapping)
            else None
        ),
        "minimum_retained_charge_pe": (
            final_cleanup.get("minimum_retained_charge_pe")
            if apply_background_assignment and isinstance(final_cleanup, Mapping)
            else None
        ),
        "minimum_retained_unique_pmts": (
            final_cleanup.get("minimum_retained_unique_pmts")
            if apply_background_assignment and isinstance(final_cleanup, Mapping)
            else None
        ),
        "decision_rule": (
            "for every accepted H1, H2, or H3 result, return the component "
            "label unless the pulse fails both broad-cone compatibility and "
            "same-label local spatial-and-fitted-time continuity; label those "
            "joint failures zero as background/unassigned"
        ),
    }


def _attach_h1_background_diagnostics(
    hypotheses: dict[str, Any],
    positions: np.ndarray,
    times: np.ndarray,
    charges: np.ndarray,
    pmt_ids: np.ndarray,
) -> None:
    """Attach a detector-only one-cone/background view to the full event."""

    h1_fit = hypotheses.get("h1_fit")
    if not isinstance(h1_fit, Mapping) or not bool(h1_fit.get("valid", False)):
        hypotheses["h1_partition_background_cleanup"] = {
            "valid": False,
            "reason": "a valid H1 point-source fit is unavailable",
        }
        return
    try:
        ring = fit_dominant_ring(
            positions,
            times,
            np.minimum(charges, 2.0),
        )
        labels = np.ones(len(charges), dtype=np.int64)
        cone = partition_cone_support_diagnostics(
            positions,
            charges,
            labels,
            [ring],
            1,
        )
        continuity = partition_spatiotemporal_continuity_diagnostics(
            positions,
            times,
            charges,
            pmt_ids,
            labels,
            [h1_fit],
            1,
            group_index=GROUP_INDEX,
        )
        cleanup = partition_background_cleanup_diagnostics(
            charges,
            pmt_ids,
            labels,
            cone,
            continuity,
            1,
        )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as error:
        hypotheses["h1_partition_background_cleanup"] = {
            "valid": False,
            "reason": f"one-cone background diagnostic failed: {error}",
        }
        return
    hypotheses["h1_partition_ring"] = ring
    hypotheses["h1_partition_cone_support"] = cone
    hypotheses["h1_partition_spatiotemporal_continuity"] = continuity
    hypotheses["h1_partition_background_cleanup"] = cleanup


def _zero_ring_support_gate(
    pmt_ids: np.ndarray,
    charges: np.ndarray,
    config: RingFinderConfig,
) -> dict[str, Any]:
    """Define zero as insufficient support for an observable ring."""

    n_hits = int(len(charges))
    n_unique_pmts = int(len(np.unique(pmt_ids)))
    enough_hits = bool(n_hits >= 12)
    enough_pmts = bool(n_unique_pmts >= config.minimum_observable_ring_pmts)
    return {
        "zero_ring": bool(not (enough_hits and enough_pmts)),
        "usable_hits": n_hits,
        "usable_unique_pmts": n_unique_pmts,
        "minimum_usable_hits": 12,
        "minimum_observable_ring_unique_pmts": int(config.minimum_observable_ring_pmts),
        "hit_support_passed": enough_hits,
        "pmt_support_passed": enough_pmts,
        "decision_rule": (
            "zero rings iff fewer than 12 usable hits or fewer than the "
            "configured minimum distinct PMTs remain after input preparation"
        ),
    }
