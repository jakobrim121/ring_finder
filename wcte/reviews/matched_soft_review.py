"""Compare ring counts with the same calibrated allowance for soft light."""

import numpy as np

from aperture_multiplicity_review import lower_count_evidence
from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from shared_scoring import match_charge_shares
from sparse_soft_sources import compare


def review(captured, prediction, original, particle, diagnostics):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    count = prediction["fractions"].shape[1] - 1
    report = dict(eligible=False, applied=False)
    confidence = lower_count_evidence(original, count)
    protected = (
        "continuation",
        "primary_repair",
        "lineage_turn",
        "primary_multiring",
        "beam_reallocation",
        "aperture_multiplicity",
        "count_review",
        "third_preservation",
        "retained_residual",
        "residual_extension",
        "supported_second",
    )
    if (
        count not in (2, 3)
        or confidence < (0.35 if count == 2 else 0.55)
        or any(diagnostics.get(k, {}).get("applied") for k in protected)
        or any(
            diagnostics.get("aperture_multiplicity", {}).get(n)
            for n in ("protected_faint_section", "protected_old_core")
        )
        or diagnostics.get("count_review", {}).get("protected_old_core")
    ):
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True, primary_models=captured.get("primary_models"))
    prior, optics, _, _ = measured_bank(choice, prepared, bank, original, count, deep=True)
    penalty = np.log(len(receiver.active))

    def score(trial):
        return 2.0 * np.asarray([trial["charge"], trial["joint"]]) - penalty * trial["dimension"]

    def mixed(trial):
        return score(trial) @ [0.65, 0.35]

    prior_higher = max((mixed(r) for r in prior if r["count"] == count), default=-np.inf)
    prior_lower = max((mixed(r) for r in prior if r["count"] == count - 1), default=-np.inf)
    if not np.isfinite(prior_higher) or prior_lower < prior_higher - 50.0:
        return prediction, report
    alternatives, searched = compare(choice, optics, count, matched_rivals=True)
    report.update(eligible=True, searched_sources=searched)
    high = [r for r in (*prior, *alternatives) if r["count"] == count]
    benchmark = np.max([score(r) for r in high], axis=0)
    benchmark_mixed = max(map(mixed, high))
    old = prediction["fractions"]
    q = np.minimum(prediction["charges"], 2.0)
    kink = diagnostics.get("kink", {})
    may_join = (
        particle == "proton"
        and count == 2
        and confidence >= 0.98
        and kink.get("applied")
        and kink.get("turn_deg", 180.0) <= 20.0
        and diagnostics.get("fast_primary", {}).get("applied")
    )
    candidates = [
        r
        for r in alternatives
        if r["count"] == count - 1
        and all(
            n["gate_pmt_support"] >= 35.0 and n["gate_leave_one_module"] >= 30.0 for n in r["neighborhoods"]
        )
    ]
    for trial in sorted(candidates, key=mixed, reverse=True):
        gain = score(trial) - benchmark
        soft_model = trial["label"].startswith("sparse_soft_")
        joint_gain = mixed(trial) - benchmark_mixed
        balanced = soft_model and confidence >= (0.90 if count == 2 else 0.55) and np.min(gain) >= 6.6349
        timed = soft_model and count == 2 and gain[0] >= -1.0 and gain[1] >= 45.0 and joint_gain >= 25.0
        joined = may_join and trial["label"].startswith("refined_no_soft_optical_1_") and np.min(gain) >= 20.0
        if not (balanced or timed or joined):
            continue
        shares = trial["fractions"]["prompt"]
        left, right, dice = match_charge_shares(old, shares[:, 1:].T, q)
        if (
            len(left) != count - 1
            or np.min(dice) < 0.78
            or not supported_shares(prediction, shares, particle)
        ):
            continue
        lost = (set(range(count)) - set(left)).pop() + 1
        weighted = q * old[:, lost]
        mass = max(weighted.sum(), 1e-12)
        covered = weighted @ trial["modeled_charge_share"] / mass
        soft = weighted @ trial["uncounted_source_share"] / mass
        protected_reduction = (
            (balanced and covered >= 0.90 and soft >= 0.84)
            or (timed and covered >= 0.90 and soft >= 0.70 and min(dice) >= 0.95)
            or (joined and covered >= 0.90 and min(dice) >= 0.90)
        )
        if not protected_reduction:
            continue
        report.update(
            applied=True,
            label=trial["label"],
            detector_gain=gain.tolist(),
            omitted_rate_covered=float(covered),
            omitted_soft_covered=float(soft),
            retained_dice=dice.tolist(),
            continuous_rejoin=bool(joined),
            strong_timing=bool(timed),
        )
        return dict(prediction, fractions=shares[:, np.r_[0, right[np.argsort(left)] + 1]]), report
    return prediction, report
