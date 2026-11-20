"""Preserve an independently illuminated third ring before accepting a reduction."""

import numpy as np

from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from shared_scoring import match_charge_shares


def review(captured, original_three, reduced, original, particle, diagnostics=None):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(applied=False)
    diagnostics = diagnostics or {}
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return reduced, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True, primary_models=captured.get("primary_models"))
    trials, _, _, _ = measured_bank(choice, prepared, bank, original, 3, deep=True)
    penalty = np.log(len(receiver.active))

    def evidence(trial):
        charge, joint = trial["charge"], trial["joint"]
        return 2.0 * np.asarray([charge, 0.65 * charge + 0.35 * joint, joint]) - penalty * trial["dimension"]

    two = [evidence(r) for r in trials if r["count"] == 2]
    if not two:
        return reduced, report
    benchmark = np.max(two, axis=0)
    alternative = reduced["fractions"]
    before = original_three["fractions"]
    q = np.minimum(reduced["charges"], 2.0)
    modules = reduced["pmt_ids"] // 100
    contenders = [
        r
        for r in trials
        if r["count"] == 3
        and all(
            n["gate_pmt_support"] >= 40.0 and n["gate_leave_one_module"] >= 30.0 for n in r["neighborhoods"]
        )
    ]
    report["tested_alternatives"] = len(contenders)
    aperture_margin = diagnostics.get("aperture_multiplicity", {}).get("margin")
    count_only = (
        diagnostics.get("count_review", {}).get("applied")
        and aperture_margin is not None
        and aperture_margin <= -75.0
    )
    for trial in sorted(contenders, key=lambda r: evidence(r)[1], reverse=True):
        gain = evidence(trial) - benchmark
        if np.min(gain) < 0.0 or gain[1] < 25.0:
            continue
        shares = trial["fractions"]["prompt"]
        left, right, _ = match_charge_shares(alternative, shares[:, 1:].T, q)
        retained = q @ np.minimum(alternative[:, left + 1], shares[:, right + 1])
        retained /= np.maximum(q @ alternative[:, left + 1], 1e-12)
        if len(right) != 2:
            continue
        extra = (set(range(3)) - set(right)).pop() + 1
        independent = q * shares[:, extra] * alternative[:, 0]
        distributed = independent.sum() - np.bincount(modules, weights=independent).max(initial=0.0)
        independently_supported = distributed >= 45.0 and np.min(retained) >= 0.48
        contradicted_count_only = count_only and np.min(gain) >= 45.0
        if not (independently_supported or contradicted_count_only) or not supported_shares(
            original_three, shares, particle
        ):
            continue
        i, j, persistence = match_charge_shares(before, shares[:, 1:].T, q)
        if len(i) != 3 or np.min(persistence) < 0.75:
            continue
        report.update(
            applied=True,
            label=trial["label"],
            detector_gain=gain.tolist(),
            independent_pe=float(distributed),
            original_dice=persistence.tolist(),
            contrasted_count_only=bool(contradicted_count_only),
        )
        return dict(original_three, fractions=shares[:, np.r_[0, 1 + j[np.argsort(i)]]]), report
    return reduced, report
