"""Compare a finite entrance section with a weak core in a three-ring fit."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from aperture_primary_only import measured_entry_overlap
from light_competition import Competition
from light_model_bank import measured_bank
from primary_competition import candidates
from wcte.counting import track_counting


def eligibility(prediction, original, particle):
    if particle not in ("pion", "proton") or prediction["fractions"].shape[1] != 4:
        return None
    probabilities = original.get("initial_counter_scores") or ()
    if isinstance(probabilities, dict):
        probability = probabilities.get("3", probabilities.get(3, 0.0))
    else:
        probability = probabilities[3] if len(probabilities) > 3 else 0.0
    if probability < 0.66:
        return None
    ids, charge = prediction["pmt_ids"], prediction["charges"]
    old = prediction["fractions"][:, 1:]
    mass = np.minimum(charge, 2.0) @ old
    if mass.min() > 80.0:
        return None
    overlap = max(measured_entry_overlap(ids, charge, core, particle) for core in old.T)
    return overlap if overlap < 0.69 else None


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    overlap = eligibility(prediction, original, particle)
    report = dict(eligible=overlap is not None, applied=False)
    if overlap is None:
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report

    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    comparison = Competition(event, receiver, prompt=True, module_charge=True)
    prior, optics, _, _ = measured_bank(comparison, prepared, bank, original, 3, deep=True, optical_starts=2)
    controls = [r for r in prior if r["count"] == 3]
    if not controls:
        return prediction, report
    fitted, starts = candidates(comparison, particle, optics, 3, second_start=True, reference_starts=2)
    penalty = np.log(len(receiver.active))

    def evidence(r):
        charge = 2.0 * r["charge"] - penalty * r["dimension"]
        timing = 2.0 * r["joint"] - penalty * r["dimension"]
        return np.array([charge, 0.65 * charge + 0.35 * timing, timing])

    maximum = np.max([evidence(r) for r in controls], axis=0)
    minimum = track_counting.get().min_track_pmts
    resolved = [r for r in controls if min(n["gate_leave_one_module"] for n in r["neighborhoods"]) >= minimum]
    supported = np.max([evidence(r) for r in resolved], axis=0) if resolved else maximum
    ids, charge = prediction["pmt_ids"], prediction["charges"]
    q = np.minimum(charge, 2.0)
    old = prediction["fractions"][:, 1:]
    mass = q @ old
    evaluated = []
    report.update(entry_overlap=overlap, tested_prefixes=len(starts), alternatives=evaluated)
    for r in sorted(fitted, key=lambda r: evidence(r)[1], reverse=True):
        if r["count"] != 3 or "_replace" not in r["label"]:
            continue
        spread = [n["gate_leave_one_module"] for n in r["neighborhoods"]]
        if min(spread) < 30.0:
            continue
        new = r["fractions"]["prompt"]
        shared = np.einsum("i,ijk->jk", q, np.minimum(old[:, :, None], new[:, None, 1:3]))
        old_index, new_index = linear_sum_assignment(shared, maximize=True)
        retention = shared[old_index, new_index] / np.maximum(mass[old_index], 1e-12)
        omitted = next(i for i in range(3) if i not in old_index)
        all_retained = (q @ np.minimum(old.sum(1), new[:, 1:].sum(1))) / mass.sum()
        aperture = measured_entry_overlap(ids, charge, new[:, -1], particle)
        margins = evidence(r) - maximum
        qualified_margins = evidence(r) - supported
        withheld_timing = []
        withheld_mixed = []
        for control in controls:
            joint = 2.0 * (r["joint_by_module"] - control["joint_by_module"])
            charge_delta = 2.0 * (r["charge_by_module"] - control["charge_by_module"])
            mixed = 0.65 * charge_delta + 0.35 * joint
            correction = penalty * (r["dimension"] - control["dimension"])
            withheld_timing.append(joint.sum() - joint.max(initial=0.0) - correction)
            withheld_mixed.append(mixed.sum() - mixed.max(initial=0.0) - correction)
        held_t, held_mixed = min(withheld_timing), min(withheld_mixed)
        measured = dict(
            label=r["label"],
            entry=float(aperture),
            all_retained=float(all_retained),
            dropped_charge=float(mass[omitted]),
            retention=retention.tolist(),
            distributed=spread,
            margins=margins.tolist(),
            supported_margins=qualified_margins.tolist(),
            withheld_timing=float(held_t),
            withheld_mixed=float(held_mixed),
        )
        evaluated.append(measured)
        if (
            min(retention) < 0.80
            or all_retained < 0.90
            or mass[omitted] > 80.0
            or aperture < 0.55
            or not (aperture - overlap >= 0.075 or overlap < 0.60)
        ):
            continue
        strong = min(margins) >= 30.0 and min(held_mixed, held_t) >= 15.0
        negligible_loss = (
            mass[omitted] <= 20.0 and all_retained >= 0.97 and min(retention) >= 0.88 and held_t > 0.0
        )
        independently_timed = (
            mass[omitted] <= 30.0
            and min(qualified_margins) >= 50.0
            and margins[0] >= -30.0
            and margins[1] >= 0.0
            and margins[2] >= 40.0
            and (held_t >= 15.0 or negligible_loss)
        )
        if strong or independently_timed:
            report.update(applied=True, selected=measured, timed_weak_core=bool(independently_timed))
            return dict(prediction, fractions=new), report
    return prediction, report
