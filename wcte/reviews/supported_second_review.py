"""Restore a measured second proton-beam ring when both detector laws support it."""

import numpy as np

from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def original_preference(original):
    """Use the already fitted charged alternatives to scope the costly comparison."""
    supported = [
        r for r in original.get("candidates", ()) if r.get("supported") and r.get("gain") is not None
    ]
    two = [r["gain"] for r in supported if r["count"] == 2 and r.get("particles") and r.get("parameters")]
    one = [r["gain"] for r in supported if r["count"] == 1]
    return max(two, default=-np.inf) - max(one, default=0.0)


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=False, applied=False)
    if particle != "proton" or prediction["fractions"].shape[1] != 2:
        return prediction, report
    preference = original_preference(original)
    if preference < 70.0:
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True)
    trials, _, _, _ = measured_bank(choice, prepared, bank, original, 1, deep=True)
    penalty = np.log(len(receiver.active))

    def evidence(trial):
        return 2.0 * np.asarray([trial["charge"], trial["joint"]]) - penalty * trial["dimension"]

    one = [evidence(r) for r in trials if r["count"] == 1]
    two = [
        r
        for r in trials
        if r["count"] == 2
        and all(
            n["gate_pmt_support"] >= track_counting.get().min_track_pmts
            and n["gate_leave_one_module"] >= track_counting.get().scaled_support(25.0)
            for n in r["neighborhoods"]
        )
    ]
    report.update(eligible=True, original_preference=float(preference), tested_alternatives=len(two))
    if not one or not two:
        return prediction, report
    benchmark = np.max(one, axis=0)
    current = prediction["fractions"]
    q = np.minimum(prediction["charges"], 2.0)
    modules = prediction["pmt_ids"] // 100
    for trial in sorted(two, key=lambda r: evidence(r) @ [0.65, 0.35], reverse=True):
        gain = evidence(trial) - benchmark
        if np.min(gain) < 75.0:
            continue
        shares = trial["fractions"]["prompt"]
        _, matched, _ = match_charge_shares(current, shares[:, 1:].T, q)
        if len(matched) != 1:
            continue
        retained_column = int(matched[0]) + 1
        added_column = 3 - retained_column
        old = current[:, 1]
        retained = (q @ np.minimum(old, shares[:, retained_column])) / max(q @ old, 1e-12)
        remaining = q * shares[:, added_column] * current[:, 0]
        distributed = remaining.sum() - np.bincount(modules, weights=remaining).max(initial=0.0)
        if retained < 0.70 or distributed < 12.0 or not supported_shares(prediction, shares, particle):
            continue
        report.update(
            applied=True,
            label=trial["label"],
            detector_gain=gain.tolist(),
            retained=float(retained),
            independent_pe=float(distributed),
        )
        return dict(prediction, fractions=shares[:, [0, retained_column, added_column]]), report
    return prediction, report
