"""Test a third ring while retaining the accepted beam and outgoing track."""

import numpy as np

from light_competition import Competition
from light_model_bank import measured_bank
from residual_cores import propose
from retained_history import compare_on_history, restore_record
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def review(captured, prediction, detail, continuation):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=False, applied=False)
    if prediction["fractions"].shape[1] != 3:
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True)
    base = restore_record(choice, continuation, prediction)
    if base is None:
        return prediction, report
    report["eligible"] = True
    trials, optics, optical_results, _ = measured_bank(choice, prepared, bank, detail, 2, deep=True)
    seed = next(((geometry, t) for _, geometry, t in optics if len(t.core_groups) == 2), None)
    if seed is None:
        return prediction, report
    geometry, optical = seed
    others = propose(choice, bank, geometry, optical, optical_results[2], thorough=True, steps=1)
    exact = compare_on_history(choice, others, base)
    rivals = [r for r in (*trials, *others, *exact) if r["count"] == 2]
    if not rivals:
        return prediction, report
    penalty = np.log(len(receiver.active))

    def measure(r):
        return 2.0 * (0.65 * r["charge"] + 0.35 * r["joint"]) - penalty * r["dimension"]

    old = max(rivals, key=measure)
    report["comparison"] = old["label"]
    q = np.minimum(prediction["charges"], 2.0)
    shares = prediction["fractions"]
    for r in sorted(exact, key=measure, reverse=True):
        if (
            r["count"] != 3
            or (margin := measure(r) - measure(old)) < 6.6349
            or min(part["gate_pmt_support"] for part in r["neighborhoods"]) < track_counting.get().min_track_pmts
        ):
            continue
        candidate = r["fractions"]["prompt"]
        i, j, _ = match_charge_shares(shares, candidate[:, 1:].T, q)
        retained = q @ np.minimum(shares[:, i + 1], candidate[:, j + 1])
        mass = q @ shares[:, i + 1]
        fraction = retained / np.maximum(mass, 1.0e-15)
        if len(i) != 2 or min(fraction) < 0.66:
            continue
        report.update(applied=True, count=3, margin=float(margin), retained_incumbent=fraction.tolist())
        return dict(prediction, fractions=candidate), report
    return prediction, report
