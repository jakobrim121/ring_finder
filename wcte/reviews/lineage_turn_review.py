"""Review a merged, same-species charged trajectory near the lineage cut."""

from dataclasses import replace

import numpy as np

from light_competition import Competition, optical_template, physical_template, short_lepton_templates
from matched_turn_split import compare, turn_degrees
from wcte.counting import track_counting


def controls(competition, prepared, bank, detail):
    from optical_chain_search import geometries

    for k in (1, 2):
        options = geometries(prepared, bank, k)
        if len(options) > 2:
            options = [options[0], options[-1]]
        best = None
        for i, g in enumerate(options):
            template = optical_template(competition.model, g, f"optical_{k}_{i}")
            if template is None:
                continue
            fit = competition.fit(template)
            if fit is None:
                continue
            if k == 1:
                yield fit
            elif best is None or fit["joint"] > best[0]:
                best = fit["joint"], g, replace(template, amplitudes=fit["ring_amplitudes"])
        if best is not None:
            _, g, t = best
            for j in (1, 2):
                for rival in short_lepton_templates(competition.model, g, t, j, flex_core=False):
                    if (fit := competition.fit(rival)) is not None:
                        yield fit
    chosen = detail["selected"]
    candidates = [
        r
        for r in (*detail.get("candidates", ()), chosen)
        if r.get("count") == 1
        and (r is chosen or r.get("supported"))
        and r.get("particles")
        and r.get("parameters")
        and r.get("order")
    ]
    for r in sorted(candidates, key=lambda r: -r.get("gain", -np.inf)):
        physical = physical_template(competition.event, competition.model, r)
        if physical is not None and (fit := competition.fit(physical, fixed_ratios=True)) is not None:
            yield fit
            break


def review(captured, prediction, detail, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    selected = detail.get("selected") or {}
    if (
        prediction["fractions"].shape[1] != 2
        or selected.get("count") != 1
        or list(selected.get("particles", ())) != [particle, particle]
        or not selected.get("order")
    ):
        return prediction, dict(applied=False, eligible=False)
    event, old, bank = (captured[k] for k in ("event", "detail", "bank"))
    rules = track_counting.get()
    minimum = rules.min_track_pmts
    opening = turn_degrees(np.asarray(selected["parameters"], float))
    if (
        opening > rules.track_merge_angle_deg
        or old.get("chord") is not None
        or "1" not in bank
        or not (event.timing_selection.get("prompt_window_ns"))
    ):
        return prediction, dict(applied=False, eligible=False)
    result = dict(applied=False, eligible=True, original_turn_deg=opening)
    bg = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, bg, ("muon",), 4, 2)
    choice = Competition(event, model, prompt=True)
    tried, plan = compare(choice, detail)
    result.update(plan)
    eligible = [
        r
        for r in tried
        if r["count"] == 2
        and min(r["retained_each_incumbent_section"]) >= 0.9
        and r["retained_both_incumbent_sections"] >= 0.9
        and min(v["gate_leave_one_module"] for v in r["neighborhoods"]) >= minimum
    ]
    merged = [r for r in tried if r["count"] == 1 and r["neighborhoods"][0]["gate_pmt_support"] >= minimum]
    if not eligible or not merged:
        return prediction, result
    size = np.log(len(model.active))

    def score(r):
        return 2.0 * (0.65 * r["charge"] + 0.35 * r["joint"]) - r["dimension"] * size

    same = [
        *merged,
        *(
            r
            for r in controls(choice, old, bank, detail)
            if r["count"] == 1 and r["neighborhoods"][0]["gate_pmt_support"] >= minimum
        ),
    ]
    before = max(same, key=score)
    alternative = max(eligible, key=score)
    delta = float(score(alternative) - score(before))
    result.update(matched_gap=delta, comparison=before["label"])
    if delta < -2.0 * np.log(4.0):
        return prediction, result
    split = alternative["fractions"]["prompt"][:, 1:]
    assigned = prediction["fractions"][:, 1]
    ratio = np.divide(split[:, 0], split.sum(1), out=np.ones(len(split)), where=split.sum(1) > 1.0e-15)
    first = assigned * np.clip(ratio, 0.0, 1.0)
    fractions = np.column_stack([prediction["fractions"][:, 0], first, assigned - first])
    if min(fractions[:, 1:].sum(0)) < minimum:
        return prediction, result
    result.update(
        applied=True,
        count=2,
        selected_turn_deg=alternative["turn_deg"],
        retained_sections=alternative["retained_each_incumbent_section"],
        prompt_support_after_largest_module=[
            r["gate_leave_one_module"] for r in alternative["neighborhoods"]
        ],
    )
    return dict(prediction, fractions=fractions), result
