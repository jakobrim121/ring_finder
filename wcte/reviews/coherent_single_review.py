"""Count a single source if it explains the distributed light of both cores."""

import numpy as np

from calibrated_count_review import supported_shares
from light_competition import Competition, physical_template
from light_model_bank import measured_bank


def eligible(prediction, original, particle):
    if particle not in ("pion", "proton") or prediction["fractions"].shape[1] != 3:
        return False
    return probability_of_one(original) >= 0.65


def probability_of_one(original):
    prob = original.get("initial_counter_scores") or ()
    if isinstance(prob, dict):
        one = prob.get("1", prob.get(1, 0.0))
    else:
        one = prob[1] if len(prob) > 1 else 0.0
    return one


def linked_records(original, particle):
    selected = original.get("selected") or {}
    return [
        r
        for r in (selected, *original.get("candidates", ()))
        if r.get("count") == 1
        and len(r.get("particles") or ()) == 2
        and r["particles"][0] == particle
        and r.get("supported")
        and r.get("parameters")
        and r.get("order")
    ]


def linked_eligible(prediction, original, particle):
    selected = original.get("selected") or {}
    return (
        particle in ("pion", "proton")
        and prediction["fractions"].shape[1] == 3
        and probability_of_one(original) >= 0.10
        and selected.get("family") != "beam_window_turn"
        and bool(linked_records(original, particle))
    )


def review(captured, prediction, original, particle, continuation):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    ordinary = eligible(prediction, original, particle)
    linked = linked_eligible(prediction, original, particle)
    report = dict(applied=False, eligible=ordinary or linked)
    if not report["eligible"]:
        return (prediction, report)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return (prediction, report)
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    competition = Competition(event, receiver, prompt=True, module_charge=True)
    trials, _, _, _ = measured_bank(competition, prepared, bank, original, 2, deep=True)
    rivals = [
        trial
        for trial in trials
        if trial["count"] == 2
        and all(
            (
                n["gate_pmt_support"] >= 30.0 and n["gate_leave_one_module"] >= 20.0
                for n in trial["neighborhoods"]
            )
        )
    ]
    accepted = captured.get("coherence_reference")
    if accepted is not None:
        rivals.append(accepted)
    if continuation.get("applied"):
        history = continuation.get("fine_chord", continuation)
        qualified = [r for r in history.get("candidates", ()) if r["count"] == 2 and r.get("eligible")]
        incumbent = max(qualified, key=lambda row: row["gain"], default=None)
        candidate = physical_template(event, receiver, incumbent) if incumbent is not None else None
        if candidate is None:
            return (prediction, dict(report, protected_continuation=True))
        from dataclasses import replace

        retained = competition.fit(replace(candidate, dimensions=incumbent["dimension"]), fixed_ratios=True)
        if retained is None:
            return (prediction, dict(report, protected_continuation=True))
        rivals.append(retained)
    report["qualified_two_ring_models"] = len(rivals)
    if not rivals:
        return (prediction, report)
    penalty = np.log(len(receiver.active))
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1:]
    mass = q @ old

    def evidence(trial):
        q_value = 2.0 * trial["charge"] - penalty * trial["dimension"]
        t_value = 2.0 * trial["joint"] - penalty * trial["dimension"]
        return np.array([q_value, 0.65 * q_value + 0.35 * t_value, t_value])

    max_rival = np.max([evidence(r) for r in rivals], axis=0)
    alternatives = []
    report["models"] = alternatives
    singles = [trial for trial in trials if trial["count"] == 1] if ordinary else []
    if linked:
        singles.extend(linked_models(competition, original, particle))
    for single in sorted(singles, key=lambda trial: evidence(trial)[1], reverse=True):
        if min((n["gate_leave_one_module"] for n in single["neighborhoods"])) < 35.0:
            continue
        new = single["fractions"]["prompt"]
        retained = q @ np.minimum(old, new[:, 1:]) / np.maximum(mass, 1e-12)
        margin = evidence(single) - max_rival
        left_out = []
        for rival in rivals:
            charge_delta = single["charge_by_module"] - rival["charge_by_module"]
            joint_delta = single["joint_by_module"] - rival["joint_by_module"]
            mixed = 0.65 * charge_delta + 0.35 * joint_delta
            left_out.append(
                2.0 * (mixed.sum() - mixed.max(initial=0.0))
                - penalty * (single["dimension"] - rival["dimension"])
            )
        independent = min(left_out)
        diagnostic = dict(
            label=single["label"],
            retained=retained.tolist(),
            gains=margin.tolist(),
            gain_without_top_module=float(independent),
        )
        alternatives.append(diagnostic)
        if (
            min(retained) >= 0.88
            and margin[0] >= 60.0
            and (margin[1] >= 50.0)
            and (margin[2] >= 6.6349)
            and (independent >= 25.0)
            and supported_shares(prediction, new, particle)
        ):
            report.update(applied=True, chosen=diagnostic)
            return (dict(prediction, fractions=new), report)
    return (prediction, report)


def linked_models(competition, original, particle):
    """Refit an attached two-section particle history at the surveyed aperture."""
    from matched_turn_split import at_aperture, qualify, seed_turn, turn_degrees

    records = linked_records(original, particle)
    valid = []
    visited = set()
    for record in sorted(records, key=lambda r: -r["gain"]):
        stamp = tuple(record["parameters"])
        if stamp in visited:
            continue
        visited.add(stamp)
        start = at_aperture(np.asarray(stamp, float))
        if start is not None:
            if turn_degrees(start) >= 14.98:
                start = seed_turn(start, 13.0)
            trial = qualify(competition, tuple(record["particles"]), start, separate=False, aperture=True)
            if trial is not None:
                valid.append(dict(trial, label="linked_aperture_" + "_".join(record["particles"])))
        if len(visited) >= 2:
            break
    return valid
