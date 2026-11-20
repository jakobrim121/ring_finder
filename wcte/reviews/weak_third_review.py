"""Test whether a low-support third ring has an independent wavefront."""

import numpy as np

from aperture_primary_only import measured_entry_overlap
from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from proton_vertex_residual import rank
from residual_cores import propose
from shared_scoring import match_charge_shares


def eligible(prediction, original, particle):
    if particle not in ("pion", "proton") or prediction["fractions"].shape[1] != 4:
        return False
    q = np.minimum(prediction["charges"], 2.0)
    smallest = (q @ prediction["fractions"][:, 1:]).min()
    return bool(15.0 <= smallest <= 48.0)


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    result = dict(eligible=eligible(prediction, original, particle), applied=False)
    if not result["eligible"]:
        return prediction, result
    event, detail, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, result
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"]
    original_mass = q @ old[:, 1:]
    weak = int(np.argmin(original_mass))
    remaining = np.delete(original_mass, weak)
    if remaining.min() < 80.0 or remaining.max() > 3.0 * remaining.min():
        return prediction, dict(result, protected_unequal_light=True)
    aperture = measured_entry_overlap(
        prediction["pmt_ids"], prediction["charges"], old[:, weak + 1], particle
    )
    if aperture >= 0.55:
        return prediction, dict(result, protected_aperture_overlap=float(aperture))
    model = Model(
        event.pmt_ids,
        event.charges_pe,
        common_background(np.asarray(bank["1"]["vertices"][0])),
        ("muon",),
        4,
        2,
    )
    competition = Competition(event, model, prompt=True, module_charge=True)
    banked, geometries, starts, _ = measured_bank(
        competition, detail, bank, original, 4, deep=True, physical_coverage=True
    )
    logn = np.log(len(model.active))
    weak_light = q * old[:, weak + 1]
    module = prediction["pmt_ids"] // 100
    candidates = []
    for trial in banked:
        if trial["count"] != 2 or min(r["gate_leave_one_module"] for r in trial["neighborhoods"]) < 35.0:
            continue
        new = trial["fractions"]["prompt"]
        left, right, _ = match_charge_shares(old, new[:, 1:].T, q)
        if set(left) != (set(range(3)) - {weak}):
            continue
        retained = (q @ np.minimum(old[:, left + 1], new[:, right + 1])) / original_mass[left]
        modeled = np.maximum(trial["modeled_charge_share"], new[:, 1:].sum(1))
        uncounted = trial["uncounted_source_share"]
        missed = weak_light * (1.0 - modeled)
        spread = np.bincount(module, weights=missed)
        if (
            retained.min() < 0.88
            or missed.sum() > 0.12 * original_mass[weak]
            or spread.sum() - spread.max(initial=0.0) > 8.0
            or (weak_light @ uncounted) < 0.60 * original_mass[weak]
            or not supported_shares(prediction, new, particle)
        ):
            continue
        candidates.append((trial, retained, float(missed.sum())))
    if not candidates:
        return prediction, result
    trial, retained, missing = max(candidates, key=lambda r: rank(r[0], logn)[1])
    if "2" not in bank or 2 not in starts:
        return prediction, result
    q3 = [
        t
        for t in banked
        if t["count"] >= 3 and min(n["gate_pmt_support"] for n in t["neighborhoods"]) >= 30.0
    ]
    if not q3:
        return prediction, result
    seed = next(((g, t) for _, g, t in geometries if len(t.core_groups) == 2), None)
    if seed is not None:
        geom, shape = seed
        extra = propose(competition, bank, geom, shape, starts[2], thorough=True, steps=2)
        q3.extend(
            t
            for t in extra
            if t["count"] >= 3 and min(n["gate_pmt_support"] for n in t["neighborhoods"]) >= 30.0
        )
    before = np.max([rank(t, logn) for t in q3], axis=0)
    gains = rank(trial, logn) - before
    held = []
    for rival in q3:
        diff = 0.65 * (trial["charge_by_module"] - rival["charge_by_module"])
        diff += 0.35 * (trial["joint_by_module"] - rival["joint_by_module"])
        held.append(
            2.0 * (diff.sum() - diff.max(initial=0.0)) - logn * (trial["dimension"] - rival["dimension"])
        )
    result.update(
        label=trial["label"],
        lower_gains=gains.tolist(),
        held=float(min(held)),
        residual_charge=missing,
        retained=retained.tolist(),
        qualified_higher=len(q3),
    )
    if gains.min() >= 50.0 and min(held) >= 25.0:
        return dict(prediction, fractions=trial["fractions"]["prompt"]), dict(result, applied=True)
    return prediction, result
