"""Fold-stable cone-resolvability gate for S_new multiple-ring claims.

The gate is deliberately one-sided: it may merge or retain an S_new
multiple-ring candidate, but it can never promote an S_new one-ring decision.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_PHASE_INDEX = 1.346
EXPECTED_MODULE_FOLDS = 4


def nominal_cherenkov_angle_deg(
    phase_index: float = DEFAULT_PHASE_INDEX,
) -> float:
    """Return the beta=1 Cherenkov angle for the supplied phase index."""

    phase_index = float(phase_index)
    if not math.isfinite(phase_index) or phase_index <= 1.0:
        raise ValueError("phase_index must be finite and exceed one")
    return float(math.degrees(math.acos(1.0 / phase_index)))


def directed_axis_separation_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the directed angle between two propagation axes in degrees.

    Unlike the weak information axes used by S_new, reconstructed cone axes
    are directed particle-propagation estimates.  The dot product therefore
    is not made absolute.
    """

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("each propagation axis must have shape (3,)")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("propagation axes must be finite")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-30 or b_norm <= 1.0e-30:
        raise ValueError("propagation axes must be nonzero")
    cosine = float((a / a_norm) @ (b / b_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def resolved_cone_diagnostics(
    result_or_hypotheses: Mapping[str, Any],
    *,
    phase_index: float = DEFAULT_PHASE_INDEX,
) -> dict[str, Any]:
    """Evaluate the adaptive angular-separation criterion.

    ``result_or_hypotheses`` may be a complete counter result containing
    ``full_event_hypotheses`` or that hypotheses mapping itself.
    """

    hypotheses = result_or_hypotheses.get("full_event_hypotheses", result_or_hypotheses)
    rings = hypotheses.get("partition_rings", [])
    if len(rings) != 2:
        return {
            "valid": False,
            "resolved": False,
            "reason": "full-event hypothesis does not contain exactly two cones",
        }

    try:
        theta_values = np.asarray([ring["theta_c"] for ring in rings], dtype=float)
        if not np.isfinite(theta_values).all() or np.any(theta_values <= 0.0):
            raise ValueError("opening angles must be finite and positive")
        separation = directed_axis_separation_deg(rings[0]["direction"], rings[1]["direction"])
        nominal = nominal_cherenkov_angle_deg(phase_index)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "valid": False,
            "resolved": False,
            "reason": f"invalid reconstructed cone parameters: {error}",
        }

    required = float(min(nominal, *theta_values.tolist()))
    return {
        "valid": True,
        "resolved": bool(separation >= required),
        "cone_axis_separation_deg": float(separation),
        "minimum_resolved_separation_deg": required,
        "nominal_beta1_cherenkov_angle_deg": nominal,
        "partition_opening_angles_deg": theta_values.tolist(),
        "separation_margin_deg": float(separation - required),
        "criterion": (
            "cone_axis_separation_deg >= min(nominal_beta1_cherenkov_angle_deg, theta_C1, theta_C2)"
        ),
    }


def classify_with_resolved_cone_gate(
    result: Mapping[str, Any],
    *,
    phase_index: float = DEFAULT_PHASE_INDEX,
) -> dict[str, Any]:
    """Apply the fold-stable separation criterion after the S_new decision.

    The original geometric criterion is evaluated independently on the full
    event and on each of the four complete-mPMT deletion refits.  A base
    multiple-ring claim is retained when the evidence satisfies one of three
    pre-defined branches:

    * at least four of the five configurations are geometrically resolved;
    * exactly three are resolved and ``S_new >= 2 * threshold``; or
    * exactly two are resolved, the full event is one of them, and
      ``S_new >= 4 * threshold``.

    Missing or malformed fold refits count as unresolved.  This function
    remains one-sided and never promotes a base one-ring decision.
    """

    if "base_prediction_s_new" in result:
        base_prediction = int(result["base_prediction_s_new"])
    elif "predicted_ring_class" in result:
        base_prediction = int(result["predicted_ring_class"])
    else:
        raise KeyError("result has no S_new base prediction")
    if base_prediction not in (1, 2):
        raise ValueError("S_new base prediction must be 1 or 2")

    full_diagnostics = resolved_cone_diagnostics(result, phase_index=phase_index)
    full_resolved = bool(full_diagnostics.get("resolved", False))

    raw_folds = result.get("module_fold_refits", [])
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, (str, bytes)):
        raw_folds = []
    folds_by_id: dict[int, Mapping[str, Any]] = {}
    for index, fold in enumerate(raw_folds):
        if not isinstance(fold, Mapping):
            continue
        try:
            fold_id = int(fold.get("withheld_fold", index))
        except (TypeError, ValueError):
            continue
        if 0 <= fold_id < EXPECTED_MODULE_FOLDS and fold_id not in folds_by_id:
            folds_by_id[fold_id] = fold

    fold_diagnostics: list[dict[str, Any]] = []
    for fold_id in range(EXPECTED_MODULE_FOLDS):
        fold = folds_by_id.get(fold_id)
        if fold is None:
            diagnostics = {
                "valid": False,
                "resolved": False,
                "reason": "module-fold refit is missing",
            }
            r_fold = None
        else:
            hypotheses = fold.get("hypotheses", {})
            if not isinstance(hypotheses, Mapping):
                hypotheses = {}
            diagnostics = resolved_cone_diagnostics(hypotheses, phase_index=phase_index)
            raw_score = fold.get("r_fold")
            try:
                r_fold = float(raw_score)
            except (TypeError, ValueError):
                r_fold = None
            if r_fold is not None and not math.isfinite(r_fold):
                r_fold = None
        fold_diagnostics.append(
            {
                **diagnostics,
                "withheld_fold": fold_id,
                "r_fold": r_fold,
            }
        )

    resolved_votes = int(full_resolved) + sum(
        int(bool(item.get("resolved", False))) for item in fold_diagnostics
    )
    try:
        s_new = float(result.get("decision_score_s_new", 0.0))
        threshold = float(result.get("threshold", 0.01))
    except (TypeError, ValueError) as error:
        raise ValueError("S_new and threshold must be numeric") from error
    if not math.isfinite(s_new) or s_new < 0.0:
        raise ValueError("S_new must be finite and non-negative")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    required_multiplier: int | None
    if resolved_votes >= 4:
        robust_resolved = True
        required_multiplier = 1
        decision_branch = "at_least_four_resolved_configurations"
    elif resolved_votes == 3:
        required_multiplier = 2
        robust_resolved = bool(s_new >= required_multiplier * threshold)
        decision_branch = "three_votes_requires_two_times_threshold"
    elif resolved_votes == 2 and full_resolved:
        required_multiplier = 4
        robust_resolved = bool(s_new >= required_multiplier * threshold)
        decision_branch = "two_votes_with_full_requires_four_times_threshold"
    else:
        required_multiplier = None
        robust_resolved = False
        decision_branch = "insufficient_fold_stable_resolvability"

    vetoed = bool(base_prediction == 2 and not robust_resolved)
    final_prediction = 1 if vetoed else base_prediction
    return {
        **full_diagnostics,
        "resolved": bool(robust_resolved),
        "full_event_resolved": full_resolved,
        "full_event_diagnostics": full_diagnostics,
        "module_fold_resolvability": fold_diagnostics,
        "configuration_resolved_flags": [
            full_resolved,
            *[bool(item.get("resolved", False)) for item in fold_diagnostics],
        ],
        "resolved_configuration_votes": int(resolved_votes),
        "expected_configuration_count": 1 + EXPECTED_MODULE_FOLDS,
        "available_configuration_count": 1 + len(folds_by_id),
        "robust_resolved": bool(robust_resolved),
        "decision_score_s_new": s_new,
        "threshold": threshold,
        "required_threshold_multiplier": required_multiplier,
        "minimum_required_decision_score_s_new": (
            None if required_multiplier is None else float(required_multiplier * threshold)
        ),
        "decision_branch": decision_branch,
        "close_angle_rescue_applied": bool(base_prediction == 2 and robust_resolved and not full_resolved),
        "criterion_applied": bool(base_prediction == 2),
        "base_prediction_s_new": base_prediction,
        "prediction_s_new_resolved": int(final_prediction),
        "multiple_ring_claim_vetoed": vetoed,
        "robust_criterion": (
            "retain a base multiple-ring claim when resolved votes across the "
            "full event plus four module folds are >=4, or are 3 with "
            "S_new>=2*threshold, or are 2 with the full event resolved and "
            "S_new>=4*threshold"
        ),
        "decision_rule": (
            "retain an S_new multiple-ring prediction only when the fixed "
            "five-configuration fold-stable cone-resolvability rule passes"
        ),
    }
