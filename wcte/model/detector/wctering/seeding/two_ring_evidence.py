"""Conservative detector-only rescue for two-ring false negatives.

The frozen S_new and fold-resolved-cone decisions remain the primary model.
This module recognizes complementary failure classes using only quantities
already produced by the forced H1/H2 fit:

* a timing-supported two-source improvement whose geometry is either nested
  in every fold or displaced beyond the beta=1 shared-vertex cone limit; and
* a near-resolved two-cone candidate whose shared-vertex opening angle is
  measurably above the beta=1 limit, the signature of a displaced cone;
* a faint, fold-stable detached cone; and
* a localized non-collinear interaction kink.

Neither branch uses truth, particle identity, sample name, or event number.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


CHARGE_CAP_PE = 2.0
CORE_BAND_DEG = 10.0
AZIMUTHAL_SECTORS = 12
SECTOR_CHARGE_FRACTION = 0.01

LOCAL_MIN_BIC_GAIN_PER_HIT = 0.25
LOCAL_MIN_CAPPED_CHARGE_FRACTION = 0.15
LOCAL_MIN_AZIMUTHAL_COVERAGE = 5.0 / 12.0
LOCAL_MAX_VERTEX_SEPARATION_MM = 750.0
LOCAL_MIN_SEPARATION_MARGIN_DEG = -10.0
LOCAL_MAX_RESOLVED_VOTES_FOR_NESTED_PAIR = 0
LOCAL_MIN_BOTH_OPENINGS_EXCESS_DEG = 0.0

DISPLACED_MIN_SEPARATION_MARGIN_DEG = -5.0
DISPLACED_MAX_SEPARATION_MARGIN_DEG = 5.0
DISPLACED_MIN_OPENING_EXCESS_DEG = 3.0
DISPLACED_MIN_BIC_GAIN_PER_HIT = 0.0
DISPLACED_MIN_CAPPED_CHARGE_FRACTION = 0.20
DISPLACED_MIN_AZIMUTHAL_COVERAGE = 0.50

# A faint, detached secondary can be real even when its small information
# fraction keeps S_new below threshold.  Requiring a high directed separation,
# four of five resolved-cone votes, a complete minority cone, and a contained
# local two-source timing improvement makes this distinct from diffuse noise.
DETACHED_MIN_BIC_GAIN_PER_HIT_EXCLUSIVE = 0.0
DETACHED_MIN_CAPPED_CHARGE_FRACTION = 0.15
DETACHED_MIN_AZIMUTHAL_COVERAGE = 10.0 / 12.0
DETACHED_MAX_VERTEX_SEPARATION_MM = 750.0
DETACHED_MIN_AXIS_SEPARATION_DEG = 60.0
DETACHED_MIN_RESOLVED_VOTES = 4

# A localized pion interaction kink has two complete, comparably populated
# cones whose independent timing vertices remain close to the interaction.
# Filled single tracks can produce a similar radial split, but their fitted
# sources separate along the long track and their cone axes remain nearly
# collinear.  The compressed opening is the expected shared-vertex distortion
# for the short post-interaction segment.
KINK_MIN_BIC_GAIN_PER_HIT = 0.10
KINK_MIN_CAPPED_CHARGE_FRACTION = 0.45
KINK_MIN_AZIMUTHAL_COVERAGE = 1.0
KINK_MAX_VERTEX_SEPARATION_MM = 350.0
KINK_MIN_AXIS_SEPARATION_DEG = 12.0
KINK_MAX_MINIMUM_OPENING_EXCESS_DEG = -8.0
KINK_MAX_RESOLVED_VOTES = 2

# Cheap screen for the more expensive forced-H3 diagnostic.  This is not a
# decision by itself.  It identifies the characteristic weak-overlap case:
# two balanced and complete H2 components, appreciably different axes, and
# nearly coincident point-source vertices despite a rejected H2 score.
OVERLAP_SCREEN_MIN_CAPPED_CHARGE_FRACTION = 0.40
OVERLAP_SCREEN_MIN_AZIMUTHAL_COVERAGE = 11.0 / 12.0
OVERLAP_SCREEN_MAX_VERTEX_SEPARATION_MM = 150.0
OVERLAP_SCREEN_MIN_AXIS_SEPARATION_DEG = 25.0

# A second physical source can be too information-poor for S_new when the
# independent point-source timing vertices land far apart.  This is distinct
# from the local-interaction branches above: demand a positive penalized timing
# gain, two balanced/complete cones, and four-of-five resolved geometry.  The
# cone widths must also show the expected source contrast: either a materially
# narrowed slow/short secondary or a widened displaced cone.  Two ordinary
# beta~1 openings are otherwise consistent with splitting one long trajectory.
LONG_BASELINE_MIN_BIC_GAIN_PER_HIT = 0.25
LONG_BASELINE_MIN_CAPPED_CHARGE_FRACTION = 0.15
LONG_BASELINE_MIN_AZIMUTHAL_COVERAGE = 0.75
LONG_BASELINE_MIN_VERTEX_SEPARATION_MM = 750.0
LONG_BASELINE_MIN_SEPARATION_MARGIN_DEG = 0.0
LONG_BASELINE_MIN_RESOLVED_VOTES = 4
LONG_BASELINE_MIN_WIDE_OPENING_EXCESS_DEG = 4.0
LONG_BASELINE_MAX_NARROW_OPENING_EXCESS_DEG = -10.0

# Cheap screen for a forced-H3 fit when H2 has likely mixed three physical
# sources into two balanced partitions.  It is intentionally only a compute
# screen; the H3 decision still requires its independent timing, charge,
# coverage, fold-stability, and cone-geometry evidence.
BALANCED_H3_SCREEN_MIN_BIC_GAIN_PER_HIT = 0.05
BALANCED_H3_SCREEN_MIN_CAPPED_CHARGE_FRACTION = 0.40


def _unit_axis(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    axis = np.average(vectors, axis=0, weights=weights)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1.0e-30:
        raise ValueError("ring axis is undefined")
    return axis / norm


def _angles_deg(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(vectors @ axis, -1.0, 1.0)))


def _azimuthal_coverage(
    vectors: np.ndarray,
    axis: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    if np.count_nonzero(mask) < 3 or float(np.sum(weights[mask])) <= 0.0:
        return 0.0
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1.0e-30)
    second = np.cross(axis, first)
    phi = np.degrees(np.arctan2(vectors[mask] @ second, vectors[mask] @ first)) % 360.0
    sectors = np.minimum(
        (phi // (360.0 / AZIMUTHAL_SECTORS)).astype(int),
        AZIMUTHAL_SECTORS - 1,
    )
    sector_charge = np.zeros(AZIMUTHAL_SECTORS, dtype=float)
    np.add.at(sector_charge, sectors, weights[mask])
    occupied = sector_charge > (SECTOR_CHARGE_FRACTION * float(np.sum(weights[mask])))
    return float(np.count_nonzero(occupied) / AZIMUTHAL_SECTORS)


def two_ring_evidence_diagnostics(
    pmt_positions_mm: np.ndarray,
    charges_pe: np.ndarray,
    hypotheses: Mapping[str, Any],
    resolved_cone_gate: Mapping[str, Any],
    *,
    base_prediction_s_new: int,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return a one-sided H1-to-H2 rescue decision and its measured evidence."""

    base_prediction = int(base_prediction_s_new)
    prediction_before = int(resolved_cone_gate.get("prediction_s_new_resolved", base_prediction))
    common: dict[str, Any] = {
        "enabled": bool(enabled),
        "criterion_applied": False,
        "rescue_applied": False,
        "rescue_branch": None,
        "base_prediction_s_new": base_prediction,
        "prediction_before_rescue": prediction_before,
        "prediction_after_rescue": prediction_before,
    }
    if not enabled:
        return {**common, "reason": "two-ring evidence rescue is disabled"}
    if prediction_before != 1 or base_prediction not in (1, 2):
        return {
            **common,
            "reason": "the primary H1/H2 decision already retained two rings",
        }

    positions = np.asarray(pmt_positions_mm, dtype=float)
    charges = np.asarray(charges_pe, dtype=float)
    labels = np.asarray(hypotheses.get("partition_labels", []), dtype=int)
    rings = hypotheses.get("partition_rings", [])
    h1_fit = hypotheses.get("h1_fit", {})
    h2_fits = hypotheses.get("h2_fits", [])
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or charges.shape != (len(positions),)
        or labels.shape != (len(positions),)
        or sorted(set(labels.tolist())) != [1, 2]
        or len(rings) != 2
        or len(h2_fits) != 2
        or not bool(h1_fit.get("valid", False))
        or not all(bool(item.get("valid", False)) for item in h2_fits)
    ):
        return {**common, "reason": "complete valid H1/H2 evidence is unavailable"}

    weights = np.minimum(charges, CHARGE_CAP_PE)
    if not np.isfinite(positions).all() or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        return {**common, "reason": "H1/H2 evidence arrays are invalid"}
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return {**common, "reason": "H1/H2 evidence has zero charge"}

    try:
        vertex = np.asarray(rings[0]["vertex"], dtype=float)
        vectors = positions - vertex
        distances = np.linalg.norm(vectors, axis=1)
        if vertex.shape != (3,) or np.any(distances <= 1.0e-30):
            raise ValueError("invalid shared cone vertex")
        unit_vectors = vectors / distances[:, None]
        capped_charge_fractions = []
        coverages = []
        for label in (1, 2):
            assigned = labels == label
            assigned_weight = float(np.sum(weights[assigned]))
            if np.count_nonzero(assigned) < 3 or assigned_weight <= 0.0:
                raise ValueError("a cone component has insufficient support")
            axis = _unit_axis(unit_vectors[assigned], weights[assigned])
            angles = _angles_deg(unit_vectors, axis)
            opening = float(np.average(angles[assigned], weights=weights[assigned]))
            core = assigned & (np.abs(angles - opening) <= CORE_BAND_DEG)
            capped_charge_fractions.append(assigned_weight / total_weight)
            coverages.append(_azimuthal_coverage(unit_vectors, axis, weights, core))

        h1_chi2 = float(h1_fit["chi2"])
        h2_chi2 = float(sum(float(item["chi2"]) for item in h2_fits))
        n_hits = int(len(positions))
        bic_gain_per_hit = float((h1_chi2 - h2_chi2 - 4.0 * math.log(n_hits)) / n_hits)
        h2_vertices = [np.asarray(item["vertex_mm"], dtype=float) for item in h2_fits]
        vertex_separation = float(np.linalg.norm(h2_vertices[0] - h2_vertices[1]))
        axis_separation = float(resolved_cone_gate["cone_axis_separation_deg"])
        separation_margin = float(resolved_cone_gate["separation_margin_deg"])
        nominal_angle = float(resolved_cone_gate["nominal_beta1_cherenkov_angle_deg"])
        opening_angles = [float(value) for value in resolved_cone_gate["partition_opening_angles_deg"]]
        minimum_opening_excess = float(min(opening_angles) - nominal_angle)
        opening_excess = float(max(opening_angles) - nominal_angle)
        resolved_votes = int(resolved_cone_gate["resolved_configuration_votes"])
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return {**common, "reason": f"invalid H1/H2 evidence: {error}"}

    metrics = {
        "h2_bic_gain_per_hit": bic_gain_per_hit,
        "h2_extra_continuous_parameters": 4,
        "minimum_capped_charge_fraction": float(min(capped_charge_fractions)),
        "minimum_ring_azimuthal_coverage": float(min(coverages)),
        "h2_vertex_separation_mm": vertex_separation,
        "cone_axis_separation_deg": axis_separation,
        "separation_margin_deg": separation_margin,
        "minimum_opening_angle_excess_deg": minimum_opening_excess,
        "maximum_opening_angle_excess_deg": opening_excess,
        "resolved_configuration_votes": resolved_votes,
        "charge_cap_pe": CHARGE_CAP_PE,
        "core_band_deg": CORE_BAND_DEG,
        "azimuthal_sector_count": AZIMUTHAL_SECTORS,
    }
    local_geometry_supported = bool(
        resolved_votes <= LOCAL_MAX_RESOLVED_VOTES_FOR_NESTED_PAIR
        or minimum_opening_excess >= LOCAL_MIN_BOTH_OPENINGS_EXCESS_DEG
    )
    timing_supported_source = bool(
        base_prediction == 1
        and bic_gain_per_hit >= LOCAL_MIN_BIC_GAIN_PER_HIT
        and min(capped_charge_fractions) >= LOCAL_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= LOCAL_MIN_AZIMUTHAL_COVERAGE
        and vertex_separation <= LOCAL_MAX_VERTEX_SEPARATION_MM
        and separation_margin >= LOCAL_MIN_SEPARATION_MARGIN_DEG
        and local_geometry_supported
    )
    displaced_near_resolved = bool(
        base_prediction == 2
        and separation_margin >= DISPLACED_MIN_SEPARATION_MARGIN_DEG
        and separation_margin <= DISPLACED_MAX_SEPARATION_MARGIN_DEG
        and opening_excess >= DISPLACED_MIN_OPENING_EXCESS_DEG
        and bic_gain_per_hit > DISPLACED_MIN_BIC_GAIN_PER_HIT
        and min(capped_charge_fractions) >= DISPLACED_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= DISPLACED_MIN_AZIMUTHAL_COVERAGE
    )
    fold_stable_detached_secondary = bool(
        base_prediction == 1
        and bic_gain_per_hit > DETACHED_MIN_BIC_GAIN_PER_HIT_EXCLUSIVE
        and min(capped_charge_fractions) >= DETACHED_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= DETACHED_MIN_AZIMUTHAL_COVERAGE
        and vertex_separation <= DETACHED_MAX_VERTEX_SEPARATION_MM
        and axis_separation >= DETACHED_MIN_AXIS_SEPARATION_DEG
        and resolved_votes >= DETACHED_MIN_RESOLVED_VOTES
    )
    localized_interaction_kink = bool(
        base_prediction == 1
        and bic_gain_per_hit >= KINK_MIN_BIC_GAIN_PER_HIT
        and min(capped_charge_fractions) >= KINK_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= KINK_MIN_AZIMUTHAL_COVERAGE
        and vertex_separation <= KINK_MAX_VERTEX_SEPARATION_MM
        and axis_separation >= KINK_MIN_AXIS_SEPARATION_DEG
        and minimum_opening_excess <= KINK_MAX_MINIMUM_OPENING_EXCESS_DEG
        and resolved_votes <= KINK_MAX_RESOLVED_VOTES
    )
    long_baseline_opening_contrast_supported = bool(
        opening_excess >= LONG_BASELINE_MIN_WIDE_OPENING_EXCESS_DEG
        or minimum_opening_excess <= LONG_BASELINE_MAX_NARROW_OPENING_EXCESS_DEG
    )
    long_baseline_displaced_source = bool(
        base_prediction == 1
        and bic_gain_per_hit >= LONG_BASELINE_MIN_BIC_GAIN_PER_HIT
        and min(capped_charge_fractions) >= LONG_BASELINE_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= LONG_BASELINE_MIN_AZIMUTHAL_COVERAGE
        and vertex_separation >= LONG_BASELINE_MIN_VERTEX_SEPARATION_MM
        and separation_margin >= LONG_BASELINE_MIN_SEPARATION_MARGIN_DEG
        and resolved_votes >= LONG_BASELINE_MIN_RESOLVED_VOTES
        and long_baseline_opening_contrast_supported
    )
    overlapping_second_axis_h3_diagnostic = bool(
        base_prediction == 1
        and min(capped_charge_fractions) >= OVERLAP_SCREEN_MIN_CAPPED_CHARGE_FRACTION
        and min(coverages) >= OVERLAP_SCREEN_MIN_AZIMUTHAL_COVERAGE
        and vertex_separation <= OVERLAP_SCREEN_MAX_VERTEX_SEPARATION_MM
        and axis_separation >= OVERLAP_SCREEN_MIN_AXIS_SEPARATION_DEG
    )
    balanced_multi_source_h3_diagnostic = bool(
        base_prediction == 1
        and bic_gain_per_hit >= BALANCED_H3_SCREEN_MIN_BIC_GAIN_PER_HIT
        and min(capped_charge_fractions) >= BALANCED_H3_SCREEN_MIN_CAPPED_CHARGE_FRACTION
    )
    branch = (
        "timing_supported_two_source"
        if timing_supported_source
        else (
            "displaced_near_resolved_cone"
            if displaced_near_resolved
            else (
                "fold_stable_detached_secondary"
                if fold_stable_detached_secondary
                else (
                    "localized_interaction_kink"
                    if localized_interaction_kink
                    else ("long_baseline_displaced_source" if long_baseline_displaced_source else None)
                )
            )
        )
    )
    rescued = branch is not None
    return {
        **common,
        **metrics,
        "criterion_applied": True,
        "rescue_applied": rescued,
        "rescue_branch": branch,
        "prediction_after_rescue": 2 if rescued else 1,
        "timing_supported_two_source_passed": timing_supported_source,
        "timing_supported_geometry_passed": local_geometry_supported,
        "displaced_near_resolved_cone_passed": displaced_near_resolved,
        "fold_stable_detached_secondary_passed": (fold_stable_detached_secondary),
        "localized_interaction_kink_passed": localized_interaction_kink,
        "long_baseline_displaced_source_passed": (long_baseline_displaced_source),
        "long_baseline_opening_contrast_supported": (long_baseline_opening_contrast_supported),
        "overlapping_second_axis_h3_diagnostic_recommended": (overlapping_second_axis_h3_diagnostic),
        "balanced_multi_source_h3_diagnostic_recommended": (balanced_multi_source_h3_diagnostic),
        "thresholds": {
            "timing_supported_two_source": {
                "minimum_h2_bic_gain_per_hit": LOCAL_MIN_BIC_GAIN_PER_HIT,
                "minimum_capped_charge_fraction": LOCAL_MIN_CAPPED_CHARGE_FRACTION,
                "minimum_ring_azimuthal_coverage": LOCAL_MIN_AZIMUTHAL_COVERAGE,
                "maximum_h2_vertex_separation_mm": LOCAL_MAX_VERTEX_SEPARATION_MM,
                "minimum_separation_margin_deg": LOCAL_MIN_SEPARATION_MARGIN_DEG,
                "geometry_alternative": {
                    "maximum_resolved_configuration_votes_for_nested_pair": (
                        LOCAL_MAX_RESOLVED_VOTES_FOR_NESTED_PAIR
                    ),
                    "minimum_both_openings_excess_deg": (LOCAL_MIN_BOTH_OPENINGS_EXCESS_DEG),
                },
            },
            "displaced_near_resolved_cone": {
                "minimum_separation_margin_deg": DISPLACED_MIN_SEPARATION_MARGIN_DEG,
                "maximum_separation_margin_deg": DISPLACED_MAX_SEPARATION_MARGIN_DEG,
                "minimum_opening_angle_excess_deg": DISPLACED_MIN_OPENING_EXCESS_DEG,
                "minimum_h2_bic_gain_per_hit_exclusive": DISPLACED_MIN_BIC_GAIN_PER_HIT,
                "minimum_capped_charge_fraction": DISPLACED_MIN_CAPPED_CHARGE_FRACTION,
                "minimum_ring_azimuthal_coverage": DISPLACED_MIN_AZIMUTHAL_COVERAGE,
            },
            "fold_stable_detached_secondary": {
                "minimum_h2_bic_gain_per_hit_exclusive": (DETACHED_MIN_BIC_GAIN_PER_HIT_EXCLUSIVE),
                "minimum_capped_charge_fraction": (DETACHED_MIN_CAPPED_CHARGE_FRACTION),
                "minimum_ring_azimuthal_coverage": (DETACHED_MIN_AZIMUTHAL_COVERAGE),
                "maximum_h2_vertex_separation_mm": (DETACHED_MAX_VERTEX_SEPARATION_MM),
                "minimum_cone_axis_separation_deg": (DETACHED_MIN_AXIS_SEPARATION_DEG),
                "minimum_resolved_configuration_votes": (DETACHED_MIN_RESOLVED_VOTES),
            },
            "localized_interaction_kink": {
                "minimum_h2_bic_gain_per_hit": KINK_MIN_BIC_GAIN_PER_HIT,
                "minimum_capped_charge_fraction": (KINK_MIN_CAPPED_CHARGE_FRACTION),
                "minimum_ring_azimuthal_coverage": (KINK_MIN_AZIMUTHAL_COVERAGE),
                "maximum_h2_vertex_separation_mm": (KINK_MAX_VERTEX_SEPARATION_MM),
                "minimum_cone_axis_separation_deg": (KINK_MIN_AXIS_SEPARATION_DEG),
                "maximum_minimum_opening_angle_excess_deg": (KINK_MAX_MINIMUM_OPENING_EXCESS_DEG),
                "maximum_resolved_configuration_votes": (KINK_MAX_RESOLVED_VOTES),
            },
            "long_baseline_displaced_source": {
                "minimum_h2_bic_gain_per_hit": (LONG_BASELINE_MIN_BIC_GAIN_PER_HIT),
                "minimum_capped_charge_fraction": (LONG_BASELINE_MIN_CAPPED_CHARGE_FRACTION),
                "minimum_ring_azimuthal_coverage": (LONG_BASELINE_MIN_AZIMUTHAL_COVERAGE),
                "minimum_h2_vertex_separation_mm": (LONG_BASELINE_MIN_VERTEX_SEPARATION_MM),
                "minimum_separation_margin_deg": (LONG_BASELINE_MIN_SEPARATION_MARGIN_DEG),
                "minimum_resolved_configuration_votes": (LONG_BASELINE_MIN_RESOLVED_VOTES),
                "opening_contrast_alternative": {
                    "minimum_wide_opening_angle_excess_deg": (LONG_BASELINE_MIN_WIDE_OPENING_EXCESS_DEG),
                    "maximum_narrow_opening_angle_excess_deg": (LONG_BASELINE_MAX_NARROW_OPENING_EXCESS_DEG),
                },
            },
            "balanced_multi_source_h3_screen": {
                "minimum_h2_bic_gain_per_hit": (BALANCED_H3_SCREEN_MIN_BIC_GAIN_PER_HIT),
                "minimum_capped_charge_fraction": (BALANCED_H3_SCREEN_MIN_CAPPED_CHARGE_FRACTION),
            },
            "overlapping_second_axis_h3_diagnostic": {
                "minimum_capped_charge_fraction": (OVERLAP_SCREEN_MIN_CAPPED_CHARGE_FRACTION),
                "minimum_ring_azimuthal_coverage": (OVERLAP_SCREEN_MIN_AZIMUTHAL_COVERAGE),
                "maximum_h2_vertex_separation_mm": (OVERLAP_SCREEN_MAX_VERTEX_SEPARATION_MM),
                "minimum_cone_axis_separation_deg": (OVERLAP_SCREEN_MIN_AXIS_SEPARATION_DEG),
            },
        },
        "decision_rule": (
            "promote only a one-ring result that passes a declared detector-only "
            "two-source branch; the overlap screen merely requests a forced-H3 "
            "diagnostic and is not itself a promotion"
        ),
    }
