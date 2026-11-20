"""Fit the one-, two-, and three-ring seed hypotheses and their support diagnostics.

These routines supply the frozen multiplicity model with the same feature
schema it was trained on. The discarded historical decision cascade is absent."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from wctering.seeding.config import RingFinderConfig
from wctering.seeding.information import (
    incremental_axis_score,
    shared_axis_score,
)
from wctering.seeding.multilateration import PointSourceFitter
from wctering.seeding.three_ring_proposals import partition_three_rings
from wctering.seeding.two_ring_proposals import partition_two_rings
from wctering.seeding.partition_quality import (
    partition_background_cleanup_diagnostics,
    partition_cone_support_diagnostics,
    partition_spatiotemporal_continuity_diagnostics,
)


SCHEMA_VERSION = "wcte-axis-incompatibility-zero-one-two-three-plus-v2.16"
PHASE_INDEX = 1.346
GROUP_INDEX = 1.388
SIGMA_T_NS = 1.0
CHARGE_WEIGHT_POWER = 1.0


from wctering.seeding.partition_fit import _fit_partition, _undirected_angle_deg


def _run_hypotheses(
    positions: np.ndarray,
    times: np.ndarray,
    charges: np.ndarray,
    pmt_ids: np.ndarray,
    config: RingFinderConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    total_charge = float(np.sum(charges))
    fitter = PointSourceFitter(
        positions,
        times,
        charges,
        pmt_ids,
        refraction_index=PHASE_INDEX,
        group_index=GROUP_INDEX,
        sigma_t_ns=SIGMA_T_NS,
        charge_weight_power=CHARGE_WEIGHT_POWER,
    )
    # H1 fits all retained hits. H2 is forced independently, then each proposed
    # component is timed separately; its existence alone does not select K=2.
    h1_fit = _fit_partition(fitter, np.ones(len(charges), dtype=bool), total_charge_pe=total_charge)
    partition_started = time.perf_counter()
    labels, rings = partition_two_rings(positions, times, charges)
    partition_runtime = time.perf_counter() - partition_started
    labels = np.asarray(labels, dtype=np.int64)
    positive_labels = sorted(set(map(int, labels[labels > 0])))
    partition_support = {
        int(label): {
            "n_hits": int(np.count_nonzero(labels == label)),
            "n_unique_pmts": int(len(np.unique(pmt_ids[labels == label]))),
        }
        for label in positive_labels
    }
    partition_cone_support = partition_cone_support_diagnostics(positions, charges, labels, rings, 2)
    result: dict[str, Any] = {
        "valid": False,
        "returned_n_rings": int(len(positive_labels)),
        "h1_fit": h1_fit,
        "partition_labels": labels,
        "partition_rings": rings,
        "partition_runtime_s": float(partition_runtime),
        "partition_support": partition_support,
        "partition_cone_support": partition_cone_support,
        "minimum_required_partition_unique_pmts": int(config.minimum_partition_pmts),
    }
    if positive_labels != [1, 2]:
        result["reason"] = "forced partitioner did not form two rings"
        result["total_runtime_s"] = float(time.perf_counter() - started)
        return result
    minimum_support = min(int(partition_support[label]["n_unique_pmts"]) for label in (1, 2))
    if config.minimum_partition_pmts > 0 and minimum_support < config.minimum_partition_pmts:
        result["reason"] = (
            f"forced partition has fewer than {config.minimum_partition_pmts} distinct PMTs in one component"
        )
        result["minimum_partition_unique_pmts"] = int(minimum_support)
        result["total_runtime_s"] = float(time.perf_counter() - started)
        return result
    # Apply the fit-validity support floor before optimizing two timing vertices.
    # The stronger observable-ring floor is a separate full-event decision gate.
    h2_fits = [_fit_partition(fitter, labels == label, total_charge_pe=total_charge) for label in (1, 2)]
    if not all(bool(fit.get("valid", False)) for fit in h2_fits):
        result.update(
            h2_fits=h2_fits,
            reason="at least one partition point-source fit was invalid",
            total_runtime_s=float(time.perf_counter() - started),
        )
        return result
    # Test whether both components can share one weak timing-information axis.
    # The classifier combines this incompatibility with light-pattern evidence.
    axis_score = shared_axis_score([fit["profiled_information_matrix_per_mm2"] for fit in h2_fits])
    axis_score["covariance_axis_separation_deg"] = _undirected_angle_deg(
        h2_fits[0]["covariance_major_axis"], h2_fits[1]["covariance_major_axis"]
    )
    partition_continuity = partition_spatiotemporal_continuity_diagnostics(
        positions,
        times,
        charges,
        pmt_ids,
        labels,
        h2_fits,
        2,
        group_index=GROUP_INDEX,
    )
    result.update(
        valid=True,
        h2_fits=h2_fits,
        partition_spatiotemporal_continuity=partition_continuity,
        partition_background_cleanup=(
            partition_background_cleanup_diagnostics(
                charges,
                pmt_ids,
                labels,
                partition_cone_support,
                partition_continuity,
                2,
            )
        ),
        axis_incompatibility=axis_score,
        minimum_partition_charge_fraction=float(min(fit["charge_fraction"] for fit in h2_fits)),
        minimum_partition_information_fraction=float(axis_score["minimum_partition_information_fraction"]),
        minimum_partition_unique_pmts=int(min(fit["n_unique_pmts"] for fit in h2_fits)),
        minimum_partition_unique_modules=int(min(fit["n_unique_modules"] for fit in h2_fits)),
        total_runtime_s=float(time.perf_counter() - started),
    )
    return result


def _run_three_ring_hypotheses(
    positions: np.ndarray,
    times: np.ndarray,
    charges: np.ndarray,
    pmt_ids: np.ndarray,
    config: RingFinderConfig,
    *,
    protect_raw_charge_timing: bool = True,
) -> dict[str, Any]:
    """Fit the guarded direct/hierarchical H3 proposal and incremental score."""

    started = time.perf_counter()
    total_charge = float(np.sum(charges))
    fitter = PointSourceFitter(
        positions,
        times,
        charges,
        pmt_ids,
        refraction_index=PHASE_INDEX,
        group_index=GROUP_INDEX,
        sigma_t_ns=SIGMA_T_NS,
        charge_weight_power=CHARGE_WEIGHT_POWER,
    )
    partition_started = time.perf_counter()
    partition_result = partition_three_rings(
        positions,
        times,
        charges,
        return_diagnostics=True,
        protect_raw_charge_timing=protect_raw_charge_timing,
    )
    if len(partition_result) == 3:
        labels, rings, proposal_diagnostics = partition_result
    else:  # Retains compatibility with patched/external legacy partitioners.
        labels, rings = partition_result
        proposal_diagnostics = {
            "enabled": False,
            "hierarchical_proposal_adopted": False,
            "reason": "legacy two-value partitioner result",
        }
    partition_runtime = time.perf_counter() - partition_started
    labels = np.asarray(labels, dtype=np.int64)
    positive_labels = sorted(set(map(int, labels[labels > 0])))
    partition_support = {
        int(label): {
            "n_hits": int(np.count_nonzero(labels == label)),
            "n_unique_pmts": int(len(np.unique(pmt_ids[labels == label]))),
        }
        for label in positive_labels
    }
    result: dict[str, Any] = {
        "valid": False,
        "returned_n_rings": int(len(positive_labels)),
        "partition_labels": labels,
        "partition_rings": rings,
        "partition_runtime_s": float(partition_runtime),
        "partition_support": partition_support,
        "partition_cone_support": partition_cone_support_diagnostics(positions, charges, labels, rings, 3),
        "partition_proposal_diagnostics": proposal_diagnostics,
        "minimum_required_partition_unique_pmts": int(config.minimum_three_ring_pmts),
    }
    if positive_labels != [1, 2, 3]:
        result["reason"] = "forced partitioner did not form three rings"
        result["total_runtime_s"] = float(time.perf_counter() - started)
        return result

    minimum_support = min(int(partition_support[label]["n_unique_pmts"]) for label in (1, 2, 3))
    if config.minimum_three_ring_pmts > 0 and minimum_support < config.minimum_three_ring_pmts:
        result["reason"] = (
            "forced three-ring partition has fewer than "
            f"{config.minimum_three_ring_pmts} distinct PMTs in one component"
        )
        result["minimum_partition_unique_pmts"] = int(minimum_support)
        result["total_runtime_s"] = float(time.perf_counter() - started)
        return result

    h3_fits = [_fit_partition(fitter, labels == label, total_charge_pe=total_charge) for label in (1, 2, 3)]
    if not all(bool(fit.get("valid", False)) for fit in h3_fits):
        result.update(
            h3_fits=h3_fits,
            reason="at least one three-ring partition point fit was invalid",
            total_runtime_s=float(time.perf_counter() - started),
        )
        return result

    # H3 asks for the extra information-axis incompatibility beyond two rings;
    # this keeps a displaced but redundant third partition from counting twice.
    axis_score = incremental_axis_score([fit["profiled_information_matrix_per_mm2"] for fit in h3_fits])
    candidate_r3 = float(axis_score["trace_normalized_incremental_incompatibility"])
    proposal_adopted = bool(proposal_diagnostics.get("hierarchical_proposal_adopted", False))
    threshold_guard: dict[str, Any] = {
        "criterion": (
            "a legacy configuration at or above the configured R3 threshold "
            "must remain at or above that threshold after proposal replacement"
        ),
        "threshold": float(config.three_ring_decision_threshold),
        "hierarchical_proposal_was_adopted": proposal_adopted,
        "candidate_r3": candidate_r3,
        "legacy_partition_evaluated": False,
        "legacy_r3": None,
        "legacy_partition_valid": None,
        "passed": True,
        "legacy_partition_restored": False,
    }
    # A new hierarchical proposal must not erase legacy evidence that crossed
    # the frozen R3 threshold. Refit the old partition only when this guard can
    # matter; restore it if its evidence remains above that threshold.
    if proposal_adopted and candidate_r3 < config.three_ring_decision_threshold:
        legacy_partition_started = time.perf_counter()
        legacy_labels, legacy_rings = partition_three_rings(
            positions,
            times,
            charges,
            protect_raw_charge_timing=protect_raw_charge_timing,
            enable_hierarchical_proposal=False,
        )
        partition_runtime += time.perf_counter() - legacy_partition_started
        legacy_labels = np.asarray(legacy_labels, dtype=np.int64)
        legacy_positive_labels = sorted(set(map(int, legacy_labels[legacy_labels > 0])))
        legacy_support = {
            int(label): {
                "n_hits": int(np.count_nonzero(legacy_labels == label)),
                "n_unique_pmts": int(len(np.unique(pmt_ids[legacy_labels == label]))),
            }
            for label in legacy_positive_labels
        }
        legacy_minimum_support = (
            min(int(legacy_support[label]["n_unique_pmts"]) for label in (1, 2, 3))
            if legacy_positive_labels == [1, 2, 3]
            else 0
        )
        legacy_partition_valid = bool(
            legacy_positive_labels == [1, 2, 3]
            and (
                config.minimum_three_ring_pmts <= 0
                or legacy_minimum_support >= config.minimum_three_ring_pmts
            )
        )
        legacy_fits: list[dict[str, Any]] = []
        legacy_axis_score: dict[str, Any] | None = None
        if legacy_partition_valid:
            legacy_fits = [
                _fit_partition(
                    fitter,
                    legacy_labels == label,
                    total_charge_pe=total_charge,
                )
                for label in (1, 2, 3)
            ]
            legacy_partition_valid = bool(all(bool(fit.get("valid", False)) for fit in legacy_fits))
        if legacy_partition_valid:
            legacy_axis_score = incremental_axis_score(
                [fit["profiled_information_matrix_per_mm2"] for fit in legacy_fits]
            )
        legacy_r3 = (
            float(legacy_axis_score["trace_normalized_incremental_incompatibility"])
            if legacy_axis_score is not None
            else None
        )
        restore_legacy = bool(legacy_r3 is not None and legacy_r3 >= config.three_ring_decision_threshold)
        threshold_guard.update(
            legacy_partition_evaluated=True,
            legacy_r3=legacy_r3,
            legacy_partition_valid=legacy_partition_valid,
            passed=not restore_legacy,
            legacy_partition_restored=restore_legacy,
        )
        if restore_legacy:
            labels = legacy_labels
            rings = legacy_rings
            positive_labels = legacy_positive_labels
            partition_support = legacy_support
            h3_fits = legacy_fits
            axis_score = legacy_axis_score
            result.update(
                returned_n_rings=3,
                partition_labels=labels,
                partition_rings=rings,
                partition_runtime_s=float(partition_runtime),
                partition_support=partition_support,
            )
            proposal_diagnostics["hierarchical_proposal_adopted"] = False
            proposal_diagnostics["returned_source"] = proposal_diagnostics.get(
                "baseline_source", "direct_peel"
            )
            proposal_diagnostics["hierarchical_candidate_rejected_by_axis_threshold"] = True
            adoption_checks = proposal_diagnostics.get("adoption_checks")
            if isinstance(adoption_checks, dict):
                adoption_checks["axis_evidence_threshold_preserved"] = False
    elif proposal_adopted:
        adoption_checks = proposal_diagnostics.get("adoption_checks")
        if isinstance(adoption_checks, dict):
            adoption_checks["axis_evidence_threshold_preserved"] = True
    proposal_diagnostics["axis_evidence_threshold_guard"] = threshold_guard
    # Keep pairwise angles for interpretation. The decision feature above uses
    # full information matrices, not just these pairwise angular summaries.
    covariance_pairwise = []
    for first in range(3):
        for second in range(first + 1, 3):
            covariance_pairwise.append(
                {
                    "partition_labels": [first + 1, second + 1],
                    "separation_deg": _undirected_angle_deg(
                        h3_fits[first]["covariance_major_axis"],
                        h3_fits[second]["covariance_major_axis"],
                    ),
                }
            )
    partition_cone_support = partition_cone_support_diagnostics(positions, charges, labels, rings, 3)
    partition_continuity = partition_spatiotemporal_continuity_diagnostics(
        positions,
        times,
        charges,
        pmt_ids,
        labels,
        h3_fits,
        3,
        group_index=GROUP_INDEX,
    )
    result.update(
        valid=True,
        h3_fits=h3_fits,
        partition_spatiotemporal_continuity=partition_continuity,
        partition_background_cleanup=(
            partition_background_cleanup_diagnostics(
                charges,
                pmt_ids,
                labels,
                partition_cone_support,
                partition_continuity,
                3,
            )
        ),
        incremental_axis_incompatibility=axis_score,
        covariance_axis_pairwise_separations=covariance_pairwise,
        minimum_partition_charge_fraction=float(min(fit["charge_fraction"] for fit in h3_fits)),
        minimum_partition_information_fraction=float(axis_score["minimum_partition_information_fraction"]),
        minimum_partition_unique_pmts=int(min(fit["n_unique_pmts"] for fit in h3_fits)),
        minimum_partition_unique_modules=int(min(fit["n_unique_modules"] for fit in h3_fits)),
        partition_cone_support=partition_cone_support,
        total_runtime_s=float(time.perf_counter() - started),
    )
    return result
