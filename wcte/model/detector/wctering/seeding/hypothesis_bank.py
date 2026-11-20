"""Direct physical hypothesis bank for a unified multiplicity model.

This bypasses the historical cascade of learned decisions. The same forced
H1/H2 fits, four module refits, H3 proposal, and existing background cleanup
are retained. No additional ROOT or event metadata are consumed.
"""

import numpy as np
from wctering.seeding.config import coerce_config
from wctering.seeding.folds import fold_ids_for_hits, load_module_fold_plan


def fit_hypothesis_bank(pmt_ids, positions, charges, times, config, event_id=None):
    from wctering.seeding.hypotheses import (
        _run_hypotheses,
        _run_three_ring_hypotheses,
        PHASE_INDEX,
        SCHEMA_VERSION,
    )
    from wctering.seeding.partition_support import (
        _attach_h1_background_diagnostics,
        _zero_ring_support_gate,
        _two_ring_observable_support_gate,
    )
    from wctering.seeding.partition_fit import _axis_set_alignment
    from wctering.serialization import seed_json as _json_safe
    from wctering.seeding.resolvability import classify_with_resolved_cone_gate
    from wctering.seeding.two_ring_evidence import two_ring_evidence_diagnostics

    cfg = coerce_config(config)
    ids = np.asarray(pmt_ids, np.int64)
    positions = np.asarray(positions, float)
    q = np.asarray(charges, float)
    t = np.asarray(times, float)
    if positions.shape != (len(ids), 3) or q.shape != (len(ids),) or t.shape != (len(ids),):
        raise ValueError("Detector hit arrays are not aligned")
    if not (np.isfinite(positions).all() and np.isfinite(q).all() and np.isfinite(t).all() and np.all(q > 0)):
        raise ValueError("Hypothesis bank requires prepared finite positive detector hits")
    zero = _zero_ring_support_gate(ids, q, cfg)
    d = dict(
        schema_version=SCHEMA_VERSION,
        event_id=event_id,
        zero_ring_support_gate=zero,
        base_prediction_s_new=1,
        decision_score_s_new=0.0,
        r_full=0.0,
        r_robust_q25=0.0,
        threshold=cfg.decision_threshold,
    )
    if zero["zero_ring"]:
        d.update(
            usable_hits=len(q),
            usable_unique_pmts=len(set(ids.tolist())),
            usable_charge_pe=float(q.sum()),
        )
        return d
    full = _run_hypotheses(positions, t, q, ids, cfg)
    _attach_h1_background_diagnostics(full, positions, t, q, ids)
    d["full_event_hypotheses"] = full
    if not full.get("valid", False):
        # Retain the established early-return feature schema. Observable
        # support is separately available in zero_ring_support_gate.
        return _json_safe(d)
    hitfold = fold_ids_for_hits(ids)
    full_score = float(full["axis_incompatibility"]["trace_normalized_incompatibility"])
    full_axes = [x["information_weak_axis"] for x in full["h2_fits"]]
    folds = []
    scores = []
    for f in range(4):
        withheld = hitfold == f
        tr = ~withheld
        h = _run_hypotheses(positions[tr], t[tr], q[tr], ids[tr], cfg)
        score = (
            float(h["axis_incompatibility"]["trace_normalized_incompatibility"])
            if h.get("valid", False)
            else 0.0
        )
        alignment = (
            _axis_set_alignment([x["information_weak_axis"] for x in h["h2_fits"]], full_axes)
            if h.get("valid", False)
            else dict(valid=False)
        )
        scores.append(score)
        folds.append(
            dict(
                withheld_fold=f,
                withheld_hits=int(withheld.sum()),
                withheld_unique_modules=len(set((ids[withheld] // 100).tolist())),
                withheld_charge_fraction=float(q[withheld].sum() / q.sum()),
                r_fold=score,
                hypotheses=h,
                alignment_to_full=alignment,
            )
        )
    robust = float(np.quantile([full_score, *scores], 0.25, method="linear"))
    snew = min(full_score, robust)
    base = 2 if snew >= cfg.decision_threshold else 1
    d.update(
        r_full=full_score,
        r_fold=scores,
        r_robust_q25=robust,
        decision_score_s_new=snew,
        base_prediction_s_new=base,
        module_fold_refits=folds,
        module_fold_plan=load_module_fold_plan(),
        usable_hits=len(q),
        usable_unique_pmts=len(set(ids.tolist())),
        usable_charge_pe=float(q.sum()),
    )
    # Reproduce the rare sub-60-PMT legacy H3 requests as well as the unified
    # >=60-PMT bank, so feature availability is identical to its training data.
    make_three = d["usable_unique_pmts"] >= 60
    if not make_three:
        gate = classify_with_resolved_cone_gate(d, phase_index=PHASE_INDEX)
        rescue = two_ring_evidence_diagnostics(
            positions,
            q,
            full,
            gate,
            base_prediction_s_new=base,
            enabled=cfg.enable_two_ring_evidence_rescue,
        )
        support = _two_ring_observable_support_gate(full, int(rescue["prediction_after_rescue"]), cfg)
        make_three = bool(
            support["prediction_after_support"] == 2
            or (
                cfg.enable_three_ring_evidence_rescue
                and (
                    base == 2
                    or rescue.get("overlapping_second_axis_h3_diagnostic_recommended", False)
                    or rescue.get("balanced_multi_source_h3_diagnostic_recommended", False)
                )
            )
        )
    if make_three:
        d["three_ring_hypothesis_test"] = {
            "full_event_hypotheses": _run_three_ring_hypotheses(positions, t, q, ids, cfg)
        }
    return _json_safe(d)
