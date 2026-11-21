"""Recover an independently supported incoming trajectory beside retained rings."""

from dataclasses import replace

import numpy as np

from aperture_primary_only import entry_evidence
from light_competition import Competition, optical_template, physical_template, short_lepton_templates
from primary_competition import candidates
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def measured_models(competition, prepared, bank, detail, count=2):
    from optical_chain_search import geometries

    optics = []
    controls = []
    for k in range(1, count + 2):
        options = []
        for j, geometry in enumerate(geometries(prepared, bank, k)):
            template = optical_template(competition.model, geometry, f"primary_multiring_optical_{k}_{j}")
            if template is None:
                continue
            found = competition.fit(template)
            if found is not None:
                options.append(
                    (found["joint"], geometry, replace(template, amplitudes=found["ring_amplitudes"]))
                )
                if k == count:
                    controls.append(found)
        if not options:
            continue
        chosen = max(options, key=lambda candidate: candidate[0])
        optics.append(chosen)
        if k not in (count, count + 1):
            continue
        _, geometry, template = chosen
        for ring in range(1, k + 1):
            for alternative in short_lepton_templates(
                competition.model, geometry, template, ring, flex_core=True
            ):
                if (
                    len(alternative.core_groups) == count
                    and (fit := competition.fit(alternative)) is not None
                ):
                    controls.append(fit)
    selected = detail.get("selected") or {}
    records = [
        r
        for r in (*detail.get("candidates", ()), selected)
        if r.get("count") == count
        and r.get("particles")
        and r.get("order")
        and r.get("parameters")
        and (r is selected or r.get("supported"))
    ]
    examined = set()
    kept = 0
    for record in sorted(records, key=lambda r: -r.get("gain", -np.inf)):
        fingerprint = tuple(record["particles"]), tuple(record["parameters"])
        if fingerprint in examined:
            continue
        examined.add(fingerprint)
        template = physical_template(competition.event, competition.model, record)
        if template is not None and (fit := competition.fit(template, fixed_ratios=True)) is not None:
            controls.append(fit)
            kept += 1
            if kept == 2:
                break
    return optics, controls


def review(captured, prediction, detail, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    old_shares = prediction["fractions"]
    count = old_shares.shape[1] - 1
    result = dict(applied=False, eligible=False)
    modeled_particles = (detail.get("selected") or {}).get("particles") or ()
    wrong_incoming_species = count == 1 and bool(modeled_particles) and modeled_particles[0] != particle
    if count not in (2, 3) and not wrong_incoming_species:
        return prediction, result
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, result
    ids, q = prediction["pmt_ids"], prediction["charges"]
    strongest = max(
        entry_evidence(ids, q, core, particle, residual_support=False)[0] for core in old_shares[:, 1:].T
    )
    _, unclaimed = entry_evidence(ids, q, old_shares[:, 1:].sum(1), particle)
    result.update(best_existing_entry_overlap=strongest, unclaimed_entry_support=unclaimed)
    if (wrong_incoming_species and unclaimed < 35.0) or (
        not wrong_incoming_species and (strongest >= 0.70 or unclaimed < 22.0)
    ):
        return prediction, result
    result["eligible"] = True
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    competition = Competition(event, model, prompt=True)
    optics, controls = measured_models(competition, prepared, bank, detail, count)
    evaluated, _ = candidates(competition, particle, optics, count, second_start=True)
    rivals = [*controls, *(trial for trial in evaluated if trial["count"] == count)]
    if not rivals:
        return prediction, result
    size = np.log(len(model.active))

    def score(fit):
        return 2.0 * (0.65 * fit["charge"] + 0.35 * fit["joint"]) - size * fit["dimension"]

    control = max(rivals, key=score)
    weights = np.minimum(q, 2.0)
    for trial in sorted(evaluated, key=score, reverse=True):
        if trial["count"] != count + 1 or (margin := score(trial) - score(control)) < (
            6.6349 if count == 2 else 20.0
        ):
            continue
        support = [core["gate_pmt_support"] for core in trial["neighborhoods"]]
        independent = trial["neighborhoods"][-1]["gate_leave_one_module"]
        if min(support) < track_counting.get().min_track_pmts or independent < track_counting.get().min_track_pmts:
            continue
        new = trial["fractions"]["prompt"]
        beam = weights * new[:, -1]
        background_share = float((beam @ old_shares[:, 0]) / max(beam.sum(), 1.0e-15))
        if background_share < 0.40:
            continue
        current, assigned, _ = match_charge_shares(old_shares, new[:, 1:].T, q)
        if len(current) != count:
            continue
        charge = weights @ old_shares[:, current + 1]
        retained = (
            weights
            @ np.minimum(old_shares[:, current + 1], new[:, assigned + 1])
            / np.maximum(charge, 1.0e-15)
        )
        if min(retained) < (0.66 if count == 2 else 0.75):
            continue
        result.update(
            applied=True,
            count=count + 1,
            comparison=control["label"],
            measured_margin=float(margin),
            previous_background_share=background_share,
            retained_incumbent=retained.tolist(),
            independent_primary_support=float(independent),
            selected=trial["label"],
        )
        return dict(prediction, fractions=new), result
    return prediction, result
