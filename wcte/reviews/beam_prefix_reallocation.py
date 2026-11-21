"""Fit distinct incoming light that a broad incumbent can absorb."""

import numpy as np

from aperture_primary_only import measured_entry_overlap
from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from primary_competition import candidates
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def review(captured, prediction, original, particle, *, chord_retry=False):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(applied=False, eligible=False)
    count = prediction["fractions"].shape[1] - 1
    probabilities = original.get("initial_counter_scores")
    if not probabilities or count not in (1, 2, 3):
        return prediction, report
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(str(i), probabilities.get(i, 0.0)) for i in range(4)]
    if len(probabilities) < 4:
        return prediction, report
    old = prediction["fractions"]
    selected = original.get("selected") or {}
    species = selected.get("particles") or ()
    wrong_incident = (
        count == 1
        and particle == "proton"
        and selected.get("family") == "residual"
        and len(species) >= 2
        and species[0] != particle
    )
    optical_one = (
        particle == "pion" and selected.get("family") == "released" and sum(probabilities[2:]) >= 0.10
    )
    if (
        count == 1
        and not (optical_one or wrong_incident)
        or count == 2
        and probabilities[3] < 0.30
        or count == 3
        and probabilities[3] < 0.90
    ):
        return prediction, report
    ids, charges = prediction["pmt_ids"], prediction["charges"]
    overlap = max(measured_entry_overlap(ids, charges, core, particle) for core in old[:, 1:].T)
    # A well represented aperture footprint already belongs to the incumbent.
    # The exceptional chord comparison has separate longitudinal evidence.
    if (count == 3 and overlap >= 0.65) or (not (chord_retry or wrong_incident) and overlap >= 0.90):
        return prediction, dict(report, represented_entry=True, entry_overlap=overlap)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report

    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True)
    prior, optics, _, _ = measured_bank(choice, prepared, bank, original, count, deep=True, optical_starts=2)
    fitted, starts = candidates(choice, particle, optics, count, second_start=True, reference_starts=2)
    penalty = np.log(len(receiver.active))

    def evidence(r):
        charge = 2.0 * r["charge"] - penalty * r["dimension"]
        timing = 2.0 * r["joint"] - penalty * r["dimension"]
        return np.asarray([charge, 0.65 * charge + 0.35 * timing, timing])

    controls = [r for r in (*prior, *fitted) if r["count"] == count]
    report.update(eligible=True, entry_overlap=overlap, tested_prefixes=len(starts))
    if not controls:
        return prediction, report
    benchmark = np.max([evidence(r) for r in controls], axis=0)
    q = np.minimum(charges, 2.0)
    old_mass = np.maximum(q @ old[:, 1:], 1e-12)
    total_old = old[:, 1:].sum(1)
    for trial in sorted(fitted, key=lambda r: evidence(r)[1], reverse=True):
        if (
            trial["count"] != count + 1
            or min(p["gate_pmt_support"] for p in trial["neighborhoods"]) < track_counting.get().min_track_pmts
        ):
            continue
        gain_q, gain, gain_t = evidence(trial) - benchmark
        new = trial["fractions"]["prompt"]
        incoming = new[:, -1]
        shared_old = (q @ np.minimum(old[:, 1:], incoming[:, None])) / old_mass
        maximum_old_overlap = 0.25 if wrong_incident else 0.24
        if max(shared_old) > maximum_old_overlap or not supported_shares(prediction, new, particle):
            continue
        i, j, _ = match_charge_shares(old, new[:, 1:].T, q)
        retained = (q @ np.minimum(old[:, i + 1], new[:, j + 1])) / old_mass[i]
        all_retained = (q @ np.minimum(total_old, new[:, 1:].sum(1))) / old_mass.sum()
        if len(i) != count or min(retained) < (0.80 if count == 2 else 0.74) or all_retained < 0.92:
            continue
        spread = trial["neighborhoods"][-1]["gate_leave_one_module"]
        unclaimed = (q @ (incoming * old[:, 0])) / max(q @ incoming, 1e-12)
        main = spread >= 30.0 and (
            optical_one
            and count == 1
            and gain >= 30.0
            and min(gain_q, gain_t) >= 20.0
            or count == 2
            and gain >= 6.6349
            and gain_q >= -20.0
            and gain_t >= 15.0
            or count == 3
            and gain >= 60.0
            and min(gain_q, gain_t) >= 30.0
        )
        chord = (
            chord_retry
            and count == 2
            and spread >= 65.0
            and gain >= 6.6349
            and gain_q >= 30.0
            and gain_t >= -60.0
        )
        faint = (
            count == 2
            and probabilities[3] >= 0.35
            and spread >= 25.0
            and unclaimed >= 0.65
            and min(retained) >= 0.90
            and gain >= -6.6349
            and gain_q >= -30.0
            and gain_t >= 18.0
        )
        # Resolve a wrong-species one-ring history only when the supplied beam
        # species and the added source explain distinctly timed light, even
        # after withholding the module that most favors the pair.
        timed = False
        if (
            wrong_incident
            and gain >= 0.0
            and gain_q >= -30.0
            and gain_t >= 55.0
            and spread >= 100.0
            and min(retained) >= 0.78
        ):
            for control in controls:
                change = trial["joint_by_module"] - control["joint_by_module"]
                margin = 2.0 * (change.sum() - change.max(initial=0.0))
                margin -= penalty * (trial["dimension"] - control["dimension"])
                if margin < 35.0:
                    break
            else:
                timed = True
        if not (main or chord or faint or timed):
            continue
        report.update(
            applied=True,
            label=trial["label"],
            detector_gain=[float(gain_q), float(gain), float(gain_t)],
            retained_charge=retained.tolist(),
            primary_overlap=shared_old.tolist(),
            independent_primary_pmts=spread,
            faint_entry=bool(faint),
            chord_retry=bool(chord),
            timed_wrong_incident=timed,
        )
        return dict(prediction, fractions=new), report
    return prediction, report
