"""Check for one independently supported ring around a retained charged fit."""

import numpy as np

from light_competition import Competition
from light_model_bank import measured_bank
from primary_residual_second import compare as add_to_primary
from residual_cores import propose
from retained_history import compare_on_history, restore_energy_fit, restore_original_physical
from shared_scoring import match_charge_shares


def review(captured, prediction, original, particle, energy_history):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    count = prediction["fractions"].shape[1] - 1
    selected = original.get("selected") or {}
    previous_tracks = len(selected.get("particles") or ())
    possible_missing_second = count == 1 and particle == "proton" and (previous_tracks >= 2)
    tested_pair = count == 2 and (
        bool(energy_history) or (selected.get("count") == 2 and previous_tracks >= 2)
    )
    report = dict(eligible=False, applied=False)
    if not (possible_missing_second or tested_pair):
        return (prediction, report)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return (prediction, report)
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(
        event, model, prompt=True, primary_models=captured.get("primary_models"), module_charge=True
    )
    if count == 1:
        source = None
    elif energy_history:
        source = restore_energy_fit(choice, energy_history)
    else:
        source = restore_original_physical(choice, original, prediction)
    if count == 2 and source is None:
        return (prediction, report)
    report["eligible"] = True
    prior, optics, controls, _ = measured_bank(choice, prepared, bank, original, count, deep=True)
    seed = next(((geometry, t) for _, geometry, t in optics if len(t.core_groups) == count), None)
    if seed is None:
        return (prediction, report)
    geometry, template = seed
    residual = propose(choice, bank, geometry, template, controls[count], thorough=True, steps=1)
    if count == 1:
        alternatives = add_to_primary(choice, particle, residual, original)
    else:
        alternatives = compare_on_history(choice, residual, source)
    penalty = np.log(len(model.active))

    def score(trial):
        return 2.0 * (0.65 * trial["charge"] + 0.35 * trial["joint"]) - penalty * trial["dimension"]

    controls = [trial for trial in (*prior, *residual, *alternatives) if trial["count"] == count]
    if energy_history:
        controls.append(energy_history)
    best = max(
        (
            t
            for t in alternatives
            if t["count"] == count + 1 and min((r["gate_pmt_support"] for r in t["neighborhoods"])) >= 30.0
        ),
        key=score,
        default=None,
    )
    if best is None or not controls:
        return (prediction, report)
    reference = max(controls, key=score)
    gain = float(score(best) - score(reference))
    shares = best["fractions"]["prompt"]
    prior_i, candidate_i, dice = match_charge_shares(
        prediction["fractions"], shares[:, 1:].T, prediction["charges"]
    )
    weights = np.minimum(prediction["charges"], 2.0)
    previous = prediction["fractions"][:, prior_i + 1]
    matched = shares[:, candidate_i + 1]
    retained = weights @ np.minimum(previous, matched) / np.maximum(weights @ previous, 1e-12)
    independent_support = best["neighborhoods"][-1]["gate_leave_one_module"]
    report.update(
        margin=gain,
        retained_dice=dice.tolist(),
        retained_charge=retained.tolist(),
        independent_support=independent_support,
        comparison=reference["label"],
        energy_refit=bool(energy_history),
    )
    charge_delta = [
        2.0 * (best["charge"] - r["charge"]) - penalty * (best["dimension"] - r["dimension"])
        for r in controls
    ]
    timed_delta = [
        2.0 * (best["joint"] - r["joint"]) - penalty * (best["dimension"] - r["dimension"]) for r in controls
    ]
    robust_timed, robust_mixed = ([], [])
    for r in controls:
        dx = best["joint_by_module"] - r["joint_by_module"]
        complexity = penalty * (best["dimension"] - r["dimension"])
        robust_timed.append(2.0 * (dx.sum() - dx.max(initial=0.0)) - complexity)
        dq = best["charge_by_module"] - r["charge_by_module"]
        balanced = 0.65 * dq + 0.35 * dx
        robust_mixed.append(2.0 * (balanced.sum() - balanced.max(initial=0.0)) - complexity)
    module_margin = float(min(robust_mixed))
    report.update(
        charge_margin=float(min(charge_delta)),
        timing_margin=float(min(timed_delta)),
        timing_without_top_module=float(min(robust_timed)),
        mixed_without_top_module=module_margin,
    )
    visible = independent_support >= 30.0 or (
        count == 1
        and gain >= 75.0
        and (independent_support >= 26.0)
        and (best["neighborhoods"][-1]["gate_pmt_support"] >= 35.0)
    )
    required_module_margin = 20.0 if count == 2 and True else 0.0
    if count == 1 and reference["label"] == "primary_residual_known_beam_finite_primary":
        required_module_margin = 6.6349
    if (
        gain < 20.0
        or module_margin <= required_module_margin
        or (not visible)
        or (len(prior_i) != count)
        or (min(dice) < 0.9)
        or (min(retained) < (0.85 if count == 1 else 0.66))
    ):
        return (prediction, report)
    unseen_core = next(iter(set(range(count + 1)) - set(candidate_i)))
    ordered = np.r_[0, 1 + candidate_i[np.argsort(prior_i)], 1 + unseen_core]
    if count == 1 and captured.get("keep_coherence_reference"):
        captured["coherence_reference"] = best
    return (dict(prediction, fractions=shares[:, ordered]), dict(report, applied=True, count=count + 1))
