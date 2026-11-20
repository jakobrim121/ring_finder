"""Construct and rank three-ring proposals from the two-ring partition.

The hierarchy recursively splits each H2 parent and checks timing, angular
evidence, separation and weakest-ring support before choosing an H3 proposal."""

from __future__ import annotations


import numpy as np


from wctering.seeding.cone_seeds import (
    _best_localized_cone,
    _fit_vertex_t0,
    _ring_params,
)
from wctering.seeding.seed_constants import (
    _H3_HIER_BACKGROUND,
    _H3_HIER_FEATURES,
    _H3_HIER_OPENING_TOLERANCE_DEG,
    _H3_HIER_PHASE_INDEX,
    _H3_HIER_WEAK_FRACTION_TOLERANCE,
    _PEEL_MARGIN,
    _QUAL_BAND,
    _SIGMA_THETA,
    _THETA_C_MAX,
    _VG,
)
from wctering.seeding.two_ring_proposals import (
    _coherence_loglik,
    _coherence_max_gap,
    _competitive_shared_vertex_k,
    _partition_quality,
    _peel_sharp_labels,
    _ring_params_shared,
    _shared_vertex_loglik,
    partition_two_rings,
)


def _positive_partition_labels(labels):
    labels = np.asarray(labels, dtype=int)
    return sorted(set(labels[labels > 0].tolist()))


def _average_ranks(values):
    """Ascending one-based ranks with deterministic average handling of ties."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _point_timing_summary(pos, t, weights):
    vertex, t0 = _fit_vertex_t0(pos, t, weights)
    distances = np.linalg.norm(pos - vertex, axis=1)
    residuals = t - t0 - distances / _VG
    return {
        "vertex": np.asarray(vertex, dtype=float),
        "t0_ns": float(t0),
        "chi2_ns2": float(np.sum(weights * residuals**2)),
        "rms_ns": float(np.sqrt(np.average(residuals**2, weights=weights))),
    }


def _one_cone_background_loglik(
    weights,
    unit_directions,
    background=_H3_HIER_BACKGROUND,
    sigma=_SIGMA_THETA,
):
    axis = np.average(unit_directions, 0, weights=weights)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    angles = np.degrees(np.arccos(np.clip(unit_directions @ axis, -1.0, 1.0)))
    opening = float(np.average(angles, weights=weights))
    density = np.exp(-0.5 * ((angles - opening) / sigma) ** 2) + background
    return float(np.sum(weights * np.log(density)))


def _two_cone_geometry_summary(unit_directions, weights, labels):
    openings = []
    axes = []
    coverage = []
    for label in (1, 2):
        selected = labels == label
        axis = np.average(unit_directions[selected], 0, weights=weights[selected])
        axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
        angles = np.degrees(np.arccos(np.clip(unit_directions @ axis, -1.0, 1.0)))
        opening = float(np.average(angles[selected], weights=weights[selected]))
        on_cone = selected & (np.abs(angles - opening) < _QUAL_BAND)
        if np.count_nonzero(on_cone) >= 3:
            anchor = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
            basis_one = np.cross(axis, anchor)
            basis_one /= max(float(np.linalg.norm(basis_one)), 1.0e-30)
            basis_two = np.cross(axis, basis_one)
            azimuth = (
                np.degrees(
                    np.arctan2(
                        unit_directions[on_cone] @ basis_two,
                        unit_directions[on_cone] @ basis_one,
                    )
                )
                % 360.0
            )
            occupied = sum(
                bool(np.any((azimuth >= sector * 30.0) & (azimuth < (sector + 1) * 30.0)))
                for sector in range(12)
            )
        else:
            occupied = 0
        axes.append(axis)
        openings.append(opening)
        coverage.append(int(occupied))
    separation = float(np.degrees(np.arccos(np.clip(float(np.dot(axes[0], axes[1])), -1.0, 1.0))))
    nominal_angle = float(np.degrees(np.arccos(1.0 / _H3_HIER_PHASE_INDEX)))
    required = min(nominal_angle, max(openings[0], 0.0), max(openings[1], 0.0))
    return {
        "opening_angles_deg": [float(value) for value in openings],
        "axis_separation_deg": separation,
        "resolved_margin_deg": float(separation - required),
        "coverage_sectors": coverage,
        "minimum_coverage_sectors": int(min(coverage)),
        "maximum_azimuthal_gap_deg": float(_coherence_max_gap(unit_directions, weights, labels)),
        "physical_completeness_quality": float(_partition_quality(unit_directions, weights, labels)),
    }


def _hierarchical_split_metrics(pos, t, q, split_labels):
    capped = np.minimum(q, 2.0)
    parent = _point_timing_summary(pos, t, capped)
    children = [
        _point_timing_summary(
            pos[split_labels == label],
            t[split_labels == label],
            capped[split_labels == label],
        )
        for label in (1, 2)
    ]
    child_chi2 = float(sum(child["chi2_ns2"] for child in children))
    count = int(len(split_labels))
    timing_bic_gain = float(parent["chi2_ns2"] - child_chi2 - 4.0 * np.log(max(count, 1)))
    vectors = pos - parent["vertex"]
    directions = vectors / np.maximum(np.linalg.norm(vectors, axis=1), 1.0e-9)[:, None]
    one_loglik = _one_cone_background_loglik(capped, directions)
    two_loglik = float(_coherence_loglik(capped, directions, split_labels))
    fractions = [
        float(np.sum(capped[split_labels == label]) / max(float(np.sum(capped)), 1.0e-30)) for label in (1, 2)
    ]
    return {
        "parent_records": count,
        "timing_bic_gain": timing_bic_gain,
        "timing_bic_gain_per_record": float(timing_bic_gain / max(count, 1)),
        "child_vertex_separation_mm": float(np.linalg.norm(children[0]["vertex"] - children[1]["vertex"])),
        "angular_loglik_gain": float(two_loglik - one_loglik),
        "angular_loglik_gain_per_capped_charge": float(
            (two_loglik - one_loglik) / max(float(np.sum(capped)), 1.0e-30)
        ),
        "capped_charge_fractions": fractions,
        "minimum_capped_charge_fraction": float(min(fractions)),
        **_two_cone_geometry_summary(directions, capped, split_labels),
    }


def _hierarchical_h3_candidates(pos, t, q, h2_labels):
    """Recursively apply the established H2 proposal to both H2 parents."""

    if _positive_partition_labels(h2_labels) != [1, 2]:
        return []
    candidates = []
    for parent in (1, 2):
        selected = h2_labels == parent
        if np.count_nonzero(selected) < 10:
            continue
        recursive_labels, _recursive_rings = partition_two_rings(pos[selected], t[selected], q[selected])
        methods = (("recursive_h2", np.asarray(recursive_labels, dtype=int)),)
        for method, split_labels in methods:
            split_labels = np.asarray(split_labels, dtype=int)
            if _positive_partition_labels(split_labels) != [1, 2]:
                continue
            labels = np.zeros(len(h2_labels), dtype=int)
            labels[h2_labels != parent] = 1
            selected_indices = np.flatnonzero(selected)
            labels[selected_indices[split_labels == 1]] = 2
            labels[selected_indices[split_labels == 2]] = 3
            if _positive_partition_labels(labels) != [1, 2, 3]:
                continue
            try:
                local = _hierarchical_split_metrics(pos[selected], t[selected], q[selected], split_labels)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            candidates.append(
                {
                    "name": f"split_parent_{parent}_{method}",
                    "parent_label": int(parent),
                    "method": method,
                    "labels": labels,
                    "local_metrics": local,
                }
            )
    if not candidates:
        return candidates
    for feature in _H3_HIER_FEATURES:
        ranks = _average_ranks([candidate["local_metrics"][feature] for candidate in candidates])
        for candidate, rank in zip(candidates, ranks):
            candidate.setdefault("feature_ranks", {})[feature] = float(rank)
            candidate["rank_score"] = float(candidate.get("rank_score", 0.0) + rank)
    return candidates


def _partition_timing_chi2(pos, t, weights, labels, expected):
    weights = np.asarray(weights, dtype=float)
    total = 0.0
    for label in expected:
        selected = labels == label
        if np.count_nonzero(selected) < 5 or float(np.sum(weights[selected])) <= 0:
            return float("inf")
        total += _point_timing_summary(pos[selected], t[selected], weights[selected])["chi2_ns2"]
    return float(total)


def _h3_resolution_summary(unit_directions, q, labels):
    axes = []
    openings = []
    for label in (1, 2, 3):
        selected = labels == label
        axis = np.average(unit_directions[selected], 0, weights=q[selected])
        axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
        angles = np.degrees(np.arccos(np.clip(unit_directions @ axis, -1.0, 1.0)))
        opening = float(np.average(angles[selected], weights=q[selected]))
        if opening > _THETA_C_MAX:
            axis, opening = _best_localized_cone(unit_directions[selected], q[selected], _THETA_C_MAX)
        axes.append(axis)
        openings.append(opening)
    nominal_angle = float(np.degrees(np.arccos(1.0 / _H3_HIER_PHASE_INDEX)))
    pairwise = []
    for first in range(3):
        for second in range(first + 1, 3):
            separation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(axes[first], axes[second])),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            required = min(nominal_angle, openings[first], openings[second])
            pairwise.append(
                {
                    "labels": [first + 1, second + 1],
                    "separation_deg": separation,
                    "required_separation_deg": float(required),
                    "margin_deg": float(separation - required),
                }
            )
    minimum_margin = float(min(item["margin_deg"] for item in pairwise))
    return {
        "opening_angles_deg": [float(value) for value in openings],
        "pairwise_resolution": pairwise,
        "minimum_pairwise_resolution_margin_deg": minimum_margin,
        "all_pairs_resolved": bool(minimum_margin >= 0.0),
    }


def _h3_global_proposal_metrics(pos, t, q, unit_directions, labels):
    capped = np.minimum(q, 2.0)
    fractions = [
        float(np.sum(capped[labels == label]) / max(float(np.sum(capped)), 1.0e-30)) for label in (1, 2, 3)
    ]
    return {
        "independent_point_timing_chi2": _partition_timing_chi2(pos, t, capped, labels, (1, 2, 3)),
        "independent_point_timing_chi2_raw_charge": _partition_timing_chi2(pos, t, q, labels, (1, 2, 3)),
        "capped_background_angular_loglik": float(_coherence_loglik(capped, unit_directions, labels)),
        "capped_charge_fractions": fractions,
        "minimum_capped_charge_fraction": float(min(fractions)),
        "maximum_azimuthal_gap_deg": float(_coherence_max_gap(unit_directions, capped, labels)),
        **_h3_resolution_summary(unit_directions, q, labels),
    }


def _jsonable_h3_candidate(candidate):
    return {
        "name": candidate["name"],
        "parent_label": int(candidate["parent_label"]),
        "method": candidate["method"],
        "rank_score": float(candidate["rank_score"]),
        "feature_ranks": dict(candidate["feature_ranks"]),
        "local_metrics": dict(candidate["local_metrics"]),
        "global_metrics": dict(candidate.get("global_metrics", {})),
    }


def partition_three_rings_with_diagnostics(
    pos,
    t,
    q,
    margin=_PEEL_MARGIN,
    *,
    protect_raw_charge_timing=True,
    enable_hierarchical_proposal=True,
):
    """Return the forced H3 proposal, rings, and deterministic proposal audit.

    ``protect_raw_charge_timing`` is enabled for the complete detector event,
    where the raw-charge point-fit chi2 enters the H2-to-H3 timing evidence.
    Module-withholding partitions may disable that guard: their role is to test
    whether capped-charge geometry and S3 survive a detector perturbation, not
    to redefine the full-event timing likelihood.
    """

    pos = np.asarray(pos, float)
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    n_hits = len(pos)
    labels = _peel_sharp_labels(pos, t, q, 3, margin=margin)
    positive = _positive_partition_labels(labels)
    shared_vertex = None
    baseline_source = "direct_peel"
    if positive == [1, 2, 3]:
        shared_vertex, _t0 = _fit_vertex_t0(pos, t, q)
        vectors = pos - shared_vertex
        distances = np.maximum(np.linalg.norm(vectors, axis=1), 1.0e-9)
        unit_directions = vectors / distances[:, None]
        competitive = _competitive_shared_vertex_k(
            pos,
            q,
            unit_directions,
            labels,
            3,
        )
        if _positive_partition_labels(competitive) == [
            1,
            2,
            3,
        ] and _shared_vertex_loglik(pos, q, unit_directions, competitive) > _shared_vertex_loglik(
            pos, q, unit_directions, labels
        ):
            labels = competitive
            baseline_source = "direct_competitive"

        baseline_metrics = _h3_global_proposal_metrics(pos, t, q, unit_directions, labels)
        if enable_hierarchical_proposal:
            h2_labels, _h2_rings = partition_two_rings(pos, t, q, margin=margin)
            candidates = _hierarchical_h3_candidates(pos, t, q, np.asarray(h2_labels, dtype=int))
        else:
            candidates = []
        selected = max(candidates, key=lambda candidate: candidate["rank_score"]) if candidates else None
        adopted = False
        adoption_checks = {
            "capped_angular_loglik_not_worse": False,
            "independent_timing_chi2_not_worse": False,
            "raw_charge_timing_or_newly_resolved_topology": False,
            "weak_component_fraction_protected": False,
            "pairwise_resolution_margin_not_worse": False,
            "local_split_openings_physical": False,
        }
        if selected is not None:
            selected["global_metrics"] = _h3_global_proposal_metrics(
                pos, t, q, unit_directions, selected["labels"]
            )
            proposal_metrics = selected["global_metrics"]
            adoption_checks = {
                "capped_angular_loglik_not_worse": bool(
                    proposal_metrics["capped_background_angular_loglik"]
                    >= baseline_metrics["capped_background_angular_loglik"]
                ),
                "independent_timing_chi2_not_worse": bool(
                    proposal_metrics["independent_point_timing_chi2"]
                    <= baseline_metrics["independent_point_timing_chi2"]
                ),
                "raw_charge_timing_or_newly_resolved_topology": bool(
                    not protect_raw_charge_timing
                    or proposal_metrics["independent_point_timing_chi2_raw_charge"]
                    <= baseline_metrics["independent_point_timing_chi2_raw_charge"]
                    or (not baseline_metrics["all_pairs_resolved"] and proposal_metrics["all_pairs_resolved"])
                ),
                "weak_component_fraction_protected": bool(
                    proposal_metrics["minimum_capped_charge_fraction"]
                    >= baseline_metrics["minimum_capped_charge_fraction"] - _H3_HIER_WEAK_FRACTION_TOLERANCE
                ),
                "pairwise_resolution_margin_not_worse": bool(
                    proposal_metrics["minimum_pairwise_resolution_margin_deg"]
                    >= baseline_metrics["minimum_pairwise_resolution_margin_deg"]
                ),
                "local_split_openings_physical": bool(
                    max(selected["local_metrics"]["opening_angles_deg"])
                    <= _THETA_C_MAX + _H3_HIER_OPENING_TOLERANCE_DEG
                ),
            }
            adopted = bool(all(adoption_checks.values()))
            if adopted:
                labels = np.asarray(selected["labels"], dtype=int)
        diagnostics = {
            "enabled": True,
            "hierarchical_proposal_enabled": bool(enable_hierarchical_proposal),
            "baseline_source": baseline_source,
            "baseline_metrics": baseline_metrics,
            "candidate_count": int(len(candidates)),
            "selection_features": list(_H3_HIER_FEATURES),
            "selection_rule": (
                "maximize the sum of average within-event ranks over local "
                "timing BIC gain, capped angular gain, cone-resolution margin, "
                "azimuthal coverage, physical completeness, and weak-component "
                "support"
            ),
            "selected_candidate": (selected["name"] if selected is not None else None),
            "adoption_checks": adoption_checks,
            "weak_fraction_tolerance": float(_H3_HIER_WEAK_FRACTION_TOLERANCE),
            "maximum_local_split_opening_deg": float(_THETA_C_MAX + _H3_HIER_OPENING_TOLERANCE_DEG),
            "raw_charge_timing_protection_enabled": bool(protect_raw_charge_timing),
            "hierarchical_proposal_adopted": adopted,
            "returned_source": (selected["name"] if adopted and selected is not None else baseline_source),
            "candidates": [_jsonable_h3_candidate(candidate) for candidate in candidates],
        }
    else:
        diagnostics = {
            "enabled": True,
            "hierarchical_proposal_enabled": bool(enable_hierarchical_proposal),
            "baseline_source": "direct_peel",
            "candidate_count": 0,
            "raw_charge_timing_protection_enabled": bool(protect_raw_charge_timing),
            "hierarchical_proposal_adopted": False,
            "returned_source": "direct_peel",
            "reason": "direct forced proposal did not form three components",
        }

    unique_labels = _positive_partition_labels(labels)
    relabelled = np.zeros(n_hits, dtype=int)
    rings = []
    for new_label, old_label in enumerate(unique_labels, 1):
        mask = labels == old_label
        relabelled[mask] = new_label
        if shared_vertex is not None:
            rings.append(_ring_params_shared(pos[mask], t[mask], q[mask], shared_vertex))
        else:
            rings.append(_ring_params(pos[mask], t[mask], q[mask], np.ones(int(mask.sum()))))
    if not rings:
        relabelled[:] = 1
        rings = [_ring_params(pos, t, q, np.ones(n_hits))]
    return relabelled, rings, diagnostics


def partition_three_rings(
    pos,
    t,
    q,
    margin=_PEEL_MARGIN,
    *,
    return_diagnostics=False,
    protect_raw_charge_timing=True,
    enable_hierarchical_proposal=True,
):
    """Return the forced H3 proposal; preserve the legacy two-value default."""

    labels, rings, diagnostics = partition_three_rings_with_diagnostics(
        pos,
        t,
        q,
        margin=margin,
        protect_raw_charge_timing=protect_raw_charge_timing,
        enable_hierarchical_proposal=enable_hierarchical_proposal,
    )
    if return_diagnostics:
        return labels, rings, diagnostics
    return labels, rings
