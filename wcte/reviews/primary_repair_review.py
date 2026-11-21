"""Restore independently measured incoming light in an ambiguous single fit."""

from dataclasses import replace

import numpy as np

from aperture_primary_only import compare as primary_options, entry_evidence, holds_entry
from light_competition import Competition, optical_template, physical_template
from wcte.counting import track_counting


def measured_bank(competition, prepared, bank, detail):
    from optical_chain_search import geometries

    optical, controls = [], []
    for count in (1, 2, 3):
        choices = geometries(prepared, bank, count)
        if len(choices) > 2:
            choices = [choices[0], choices[-1]]
        viable = []
        for i, g in enumerate(choices):
            t = optical_template(competition.model, g, f"primary_control_{count}_{i}")
            if t is None:
                continue
            scored = competition.fit(t)
            if scored is not None:
                viable.append((scored["joint"], g, replace(t, amplitudes=scored["ring_amplitudes"])))
                if count == 1:
                    controls.append(scored)
        if viable:
            optical.append(max(viable, key=lambda r: r[0]))
    banked = [
        r
        for r in detail.get("candidates", [])
        if r.get("supported")
        and r.get("count") == 1
        and r.get("particles")
        and r.get("parameters")
        and r.get("order")
    ]
    selected = detail.get("selected") or {}
    if selected.get("count") == 1 and selected.get("particles") and selected.get("order"):
        banked.append(selected)
    for r in sorted(banked, key=lambda one: -one.get("gain", -np.inf)):
        t = physical_template(competition.event, competition.model, r)
        if t is not None and (scored := competition.fit(t, fixed_ratios=True)) is not None:
            controls.append(scored)
            break
    return optical, controls


def review(captured, prediction, detail, particle):
    from beam_endpoint_grid import compare as endpoint_trials
    from coupled_model import Model
    from detached_emission import compare as detached_trials
    from wctering.likelihood.geometry import common_background

    event, old, bank = (captured[k] for k in ("event", "detail", "bank"))
    if (
        prediction["fractions"].shape[1] != 2
        or not event.timing_selection.get("prompt_window_ns")
        or "1" not in bank
    ):
        return prediction, dict(applied=False, reason="Outside prepared single-ring scope")
    if old.get("chord") is not None or holds_entry(detail, old):
        return prediction, dict(applied=False, reason="Measured incumbent reaches the entrance aperture")
    prior = prediction["fractions"][:, 1]
    overlap, residual_support = entry_evidence(event.pmt_ids, event.charges_pe, prior, particle)
    result = dict(applied=False, entry_overlap=overlap, unclaimed_entry_support=residual_support)
    if overlap >= 0.70 or residual_support < track_counting.get().scaled_support(22.0):
        return prediction, dict(
            result, reason="The entry light agrees with the incumbent or lacks separate support"
        )
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, model, prompt=True)
    optical, controls = measured_bank(choice, old, bank, detail)
    if not controls:
        return prediction, dict(result, reason="No fitted same-count reference")
    revised, screened = endpoint_trials(choice, particle, optical, receiver_screen=True)
    alternatives = primary_options(choice, revised, prediction, detail, old, independent_support=True)
    if not alternatives:
        return prediction, dict(result, reason="No independently supported incoming ring", screened=screened)
    log_size = np.log(len(model.active))

    def information(option):
        return 2.0 * (0.65 * option["charge"] + 0.35 * option["joint"]) - option["dimension"] * log_size

    best = max(alternatives, key=information)
    reference = max(controls, key=information)
    margin = float(information(best) - information(reference))
    if margin < 6.6349:
        return prediction, dict(
            result,
            reason="The fitted incumbent keeps stronger measured evidence",
            screened=screened,
            primary_evidence_margin=margin,
        )
    selected = best["fractions"]["prompt"]
    status = dict(
        result,
        applied=True,
        screened=screened,
        primary_evidence_margin=margin,
        count=1,
        selected=best["label"],
        matched_single=reference["label"],
    )

    detached, n = detached_trials(choice, particle, optical, triangulate=True)
    control_score = max(information(t) for t in (best, *controls, *(x for x in detached if x["count"] == 1)))
    charge = np.minimum(prediction["charges"], 2.0)
    primary = selected[:, 1]
    for trial in sorted(detached, key=information, reverse=True):
        if trial["count"] != 2 or any(
            x["gate_pmt_support"] < track_counting.get().min_track_pmts for x in trial["neighborhoods"]
        ):
            continue
        shared = trial["fractions"]["prompt"]
        retention = float((charge @ np.minimum(primary, shared[:, 1])) / max(charge @ primary, 1.0e-12))
        improvement = information(trial) - control_score
        if retention >= 0.66 and improvement >= 6.6349:
            selected = shared
            status.update(
                count=2,
                selected=trial["label"],
                retained_incoming_charge=retention,
                second_ring_evidence_margin=float(improvement),
                daughter_template=trial["daughter_species"],
            )
            break
    status["detached_starts"] = n
    return dict(prediction, fractions=selected), status
