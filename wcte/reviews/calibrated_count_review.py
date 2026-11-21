"""Compare the retained partition with calibrated alternative ring counts."""

from functools import lru_cache
import json

import numpy as np
from scipy.special import softmax

from aperture_primary_only import measured_entry_overlap
from counter_light_features import light_features
from light_competition import Competition
from light_model_bank import measured_bank
from wcte.counting import optical_sources_resolved, track_counting


@lru_cache(maxsize=2)
def read_model(path):
    return json.loads(path.read_text())


def class_probabilities(package, row):
    values = np.asarray([row.get(key, np.nan) for key in package["features"]], dtype=float)
    raw = np.asarray(package["bias"], dtype=float).copy()
    for stage in package["stages"]:
        for i, tree in enumerate(stage):
            vertex = 0
            while isinstance(tree[vertex], list):
                feature, split, absent, left, right = tree[vertex]
                observed = values[feature]
                in_low_branch = absent if np.isnan(observed) else observed <= float(split)
                vertex = left if in_low_branch else right
            raw[i] += tree[vertex]
    return softmax(raw)


def evidence_cost(probabilities, prior, margin, missed_cost):
    categories = np.arange(4)
    delta = categories[:, None] - categories[None, :]
    loss = missed_cost * np.maximum(delta, 0) + np.maximum(-delta, 0) + 0.18 * (delta != 0)
    return probabilities @ loss + margin * (categories != prior)


def supported_shares(prediction, proposal, particle):
    old = prediction["fractions"][:, 1:]
    if not old.size:
        return True
    ids, charge = prediction["pmt_ids"], prediction["charges"]
    strength = [measured_entry_overlap(ids, charge, ring, particle) for ring in old.T]
    aperture = max(strength)
    if aperture < 0.70:
        return True
    primary = old[:, np.argmax(strength)]
    q = np.minimum(charge, 2.0)
    overlap = (q @ np.minimum(primary[:, None], proposal[:, 1:])).max(initial=0.0)
    return overlap >= 0.66 * (q @ primary)


def review(captured, prediction, recent, particle, package, *, allow_reduction=True):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    model = read_model(package)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    exact_count = prediction["fractions"].shape[1] - 1
    if not exact_count or "1" not in bank or event.timing_selection.get("prompt_window_ns") is None:
        return (prediction, dict(applied=False, eligible=False))
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    competition = Competition(event, receiver)
    trials, _, _, notes = measured_bank(
        competition, prepared, bank, recent, exact_count, deep=model["deep_bank"]
    )
    report = dict(eligible=True, applied=False, candidates=len(trials))
    receipt = dict(
        trials=trials,
        n_active=len(receiver.active),
        n_hits=len(event.pmt_ids),
        raw_charge=float(event.charges_pe.sum()),
        initial_class=prepared.get("initial_k"),
    )
    arrays = {
        f"{i}_{name}": t[name] for i, t in enumerate(trials) for name in ("joint_by_module", "module_shape")
    }
    features, choices = light_features(dict(particle=particle), receipt, arrays)
    if model.get("common_trajectory"):
        from common_trajectory_features import describe_times

        arrays.update({f"{i}_timed": t["fractions"]["timed"] for i, t in enumerate(trials)})
        features.update(describe_times(event.times_ns, receipt, arrays, prediction))
    features["incumbent_count"] = min(exact_count, 3)
    if notes:
        report["failed_bank_comparisons"] = notes
    probabilities = class_probabilities(model, features)
    prior = min(exact_count, 3)
    costs = evidence_cost(probabilities, prior, model["review_margin"], model["missed_cost"])
    report.update(probabilities=probabilities.tolist(), policy_cost=costs.tolist())
    minimum = track_counting.get().min_track_pmts
    for count in np.argsort(costs):
        if count == prior:
            break
        if count not in choices or (count < prior and (not allow_reduction)):
            continue
        if count == 1 and exact_count == 2 and (costs[prior] - costs[count] < 0.1):
            report["weak_reduction_evidence"] = True
            continue
        for index, mode in ((choices[count][1], "timed"), (choices[count][0], "charge")):
            trial = trials[index]
            share = trial["fractions"][mode]
            support = [n["pmt_support"] for n in trial["neighborhoods"]]
            if min(support, default=0.0) < minimum or not supported_shares(prediction, share, particle):
                continue
            geometry = trial.get("geometry")
            if geometry and not optical_sources_resolved(geometry["axes"], geometry["vertices"]):
                continue
            if count == 2 and exact_count >= 3:
                from merged_charge_support import lost_ring_signal, preserves_distributed_light

                retained, remaining = lost_ring_signal(prediction, share)
                report.update(old_core_retained=retained, old_core_unexplained_pe=remaining)
                if not preserves_distributed_light(retained, remaining):
                    report["protected_old_core"] = True
                    continue
            report.update(
                applied=True,
                previous_count=exact_count,
                count=trial["count"],
                label=trial["label"],
                share_rule=mode,
            )
            return (dict(prediction, fractions=share), report)
    return (prediction, report)
