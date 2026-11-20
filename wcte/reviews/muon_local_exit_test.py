"""Test whether hardware-localized excess leaves a separately supported water ring."""

from dataclasses import replace

import numpy as np

from censored_first_pulse import create
from light_competition import Competition
from light_model_bank import measured_bank
from muon_exit_response import templates
from shared_scoring import match_charge_shares


def affected(prediction):
    q = prediction["charges"]
    capped = np.minimum(q, 2.0)
    old = prediction["fractions"][:, 1:]
    modules = prediction["pmt_ids"] // 100
    if old.shape[1] != 3:
        return None
    for i, r in enumerate(old.T):
        light = np.bincount(modules, weights=q * r)
        main = int(light.argmax())
        score = light[main] / max(light.sum(), 1.0e-8)
        rest = (capped * r)[modules != main].sum()
        if score >= 0.88 and rest <= 8.0 and (capped @ r) >= 10.0:
            return i, main, float(score), float(rest)
    return None


def controls(choice, placed, slot, where):
    """Give every hypothesis the same localized response before testing water."""
    active = placed.shapes.copy()
    centers = placed.center
    in_module = active[slot] * where
    beyond = active[slot] - in_module
    whole = placed.amplitudes[slot]
    in_total = in_module.sum()
    out_total = beyond.sum()
    if in_total <= 0.45:
        return None, None
    # Deliberately retain the measured same-module charge pattern and pulse
    # shape. It cannot qualify a water ring within the masked module.
    active[slot] = in_module / in_total
    amplitude = placed.amplitudes.copy()
    amplitude[slot] = whole * in_total
    groups = tuple(g for g in placed.core_groups if slot not in g)
    lower = replace(
        placed,
        label="water_components_and_local_hardware",
        shapes=active,
        amplitudes=amplitude,
        core_groups=groups,
        dimensions=placed.dimensions,
    )
    if out_total <= 1.0e-7:
        return lower, None
    runner = placed.time_groups.max() + 1
    upper = replace(
        lower,
        label=lower.label + "_plus_distributed",
        shapes=np.vstack([active, beyond / out_total]),
        center=np.column_stack([centers, centers[:, slot - 1]]),
        amplitudes=np.r_[amplitude, whole * out_total],
        time_groups=np.r_[placed.time_groups, runner],
        core_groups=groups + ((len(amplitude),),),
        dimensions=lower.dimensions + 2,
    )
    return lower, upper


def single_water(choice, template, pair, rivals, prediction, weak, module, logn):
    """Remove an empty water component and check alternatives with local light."""
    from muon_exit_response import add_exit

    support = [n["gate_pmt_support"] for n in pair["neighborhoods"]]
    if min(support) > 10.0 or max(support) < 100.0:
        return None, None
    drop = template.core_groups[int(np.argmin(support))][0]
    present = np.delete(np.arange(len(template.amplitudes)), drop)
    remain = tuple((j - int(j > drop),) for (j,) in template.core_groups if j != drop)
    clock = template.time_groups[present[1:] - 1]
    _, clock = np.unique(clock, return_inverse=True)
    fixed = replace(
        template,
        label=template.label + "_one_water",
        shapes=template.shapes[present],
        center=template.center[:, present[1:] - 1],
        amplitudes=pair["ring_amplitudes"][present],
        time_groups=clock,
        core_groups=remain,
        dimensions=template.dimensions - 8,
    )
    solution = choice.fit(fixed, iterations=43, physical_coverage=True)
    if solution is None:
        return None, None
    local = next(
        j for j in range(1, len(template.amplitudes)) if j not in {i for g in template.core_groups for i in g}
    )
    hardware = dict(
        module=module,
        shape=template.shapes[local],
        center=template.center[:, local - 1],
        start=pair["ring_amplitudes"][local],
        limit=2.0 * pair["ring_amplitudes"][local],
    )
    comparators = [pair]
    for _, _, base in rivals:
        if len(base.core_groups) != 2:
            continue
        alternative = choice.fit(add_exit(base, hardware, joined=False), iterations=42)
        if alternative is not None:
            comparators.append(alternative)
    q = choice.q
    prior = prediction["fractions"][:, 1:]
    pmt = prediction["pmt_ids"] // 100
    new = solution["fractions"]["prompt"]
    core = new[:, 1]
    new_total = core + solution["uncounted_source_share"]
    hold = (q @ np.minimum(prior, core[:, None])) / np.maximum(q @ prior, 1.0e-12)
    outside = pmt != module
    lost = q[outside] * np.maximum(prior.sum(1)[outside] - new_total[outside], 0.0)
    lost_outside = lost.sum()
    excluded = np.bincount(pmt[outside], weights=lost)
    distributed = lost_outside - excluded.max(initial=0.0)
    total = (q @ np.minimum(prior.sum(1), new_total)) / max(q @ prior.sum(1), 1.0e-12)
    contrasts = []
    for other in comparators:
        difference = solution["joint_by_module"] - other["joint_by_module"]
        penalty = logn * (solution["dimension"] - other["dimension"])
        contrasts.append(
            (
                2.0 * difference.sum() - penalty,
                2.0 * (difference.sum() - difference.max(initial=0.0)) - penalty,
            )
        )
    score, held = np.min(contrasts, axis=0)
    proof = dict(
        one_water_gain=float(score),
        held=float(held),
        water_core_retained=hold.tolist(),
        total_physically_accounted=float(total),
        lost_outside_capped_pe=float(lost_outside),
        lost_beyond_strongest_outside_module_pe=float(distributed),
        one_water_support=solution["neighborhoods"][0]["gate_leave_one_module"],
        comparators=len(comparators),
    )
    if (
        score >= 10.0
        and held >= 3.0
        and total >= 0.96
        and lost_outside <= 13.0
        and distributed <= 8.0
        and min(np.delete(hold, weak)) >= 0.90
        and proof["one_water_support"] >= 90.0
    ):
        return new, proof
    return None, proof


def review(captured, prediction, detail):
    from charge_starts import weighted_quantile
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=False, applied=False)
    mark = affected(prediction)
    if mark is None:
        return prediction, report
    weak, module, concentration, extra = mark
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    line = (prepared.get("rejected_chord") or {}).get("chord")
    if (
        line is None
        or "1" not in bank
        or not event.timing_selection.get("prompt_window_ns")
        or line["chord_m"] < 2.70
        or line["coherent_coverage"] < 0.74
    ):
        return prediction, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, model, prompt=True, module_charge=True)
    choice.selection = create(event, model, 4.0)
    path, local = templates(choice, line)
    if path is None or local is None or local["module"] != module:
        return prediction, report
    report.update(
        eligible=True, module=module, extra_capped_outside=extra, concentration=concentration, options=[]
    )
    water = choice.fit(path, iterations=33)
    if water is None:
        return prediction, report
    original, optics, _, _ = measured_bank(
        choice, prepared, bank, detail, 3, deep=True, compare_count=3, optical_starts=3
    )
    logn = np.log(len(model.active))
    independent = [
        r
        for r in original
        if r["count"] == 3
        and min((n.get("gate_leave_one_module", 0.0) for n in r.get("neighborhoods", [])), default=0.0)
        >= 20.0
    ]
    q = choice.q
    old = prediction["fractions"][:, 1:]
    old_mass = q @ old
    pmts = model.active // 100 == module
    foot = model.ix
    weights = q * old[:, weak] * pmts[foot]
    if weights.sum() <= 8.0:
        return prediction, report
    med = weighted_quantile(
        choice.selection.time - local["center"][foot] - water["clocks"][0], weights, [0.5]
    )[0]
    report["measured_exit_clock_ns"] = float(med)

    def score(t):
        return float(2.0 * t["joint"] - logn * t["dimension"])

    for _, _, shape in optics:
        if len(shape.core_groups) != 3:
            continue
        fit = choice.fit(shape, iterations=35)
        if fit is None:
            continue
        shape = replace(shape, amplitudes=fit["ring_amplitudes"])
        values = fit["fractions"]["prompt"][:, 1:]
        overlap = (
            2.0
            * (q @ np.minimum(old[:, weak, None], values))
            / np.maximum(old_mass[weak] + q @ values, 1.0e-8)
        )
        i = int(np.argmax(overlap))
        if overlap[i] < 0.55:
            continue
        before, upper = controls(choice, shape, i + 1, pmts)
        if before is None:
            continue
        lower = choice.fit(before, iterations=35, physical_coverage=True)
        if lower is None:
            continue
        other = choice.fit(upper, iterations=32) if upper is not None else None
        strongest = max((score(t) for t in independent), default=score(fit))
        shares = lower["fractions"]["prompt"]
        # Trace both independent water regions through the lower count model.
        keep = np.delete(np.arange(old.shape[1]), weak)
        a, b, dice = match_charge_shares(
            np.column_stack([1.0 - old[:, keep].sum(1), old[:, keep]]), shares[:, 1:].T, q
        )
        compare = (q @ np.minimum(old[:, keep[a]], shares[:, b + 1])) / np.maximum(old_mass[keep[a]], 1.0e-8)
        core = lower["uncounted_source_share"]
        light = float((q @ np.minimum(old[:, weak], core)) / old_mass[weak])
        addition = other["neighborhoods"][-1]["gate_leave_one_module"] if other is not None else 0.0
        water_added = score(other) - score(lower) if other is not None else 0.0
        evidence = score(lower) - strongest
        status = dict(
            water_retention=compare.tolist(),
            hardware_explained=light,
            absorbed_only_at_exit_pe=float(q @ core),
            independent_water_support=addition,
            extra_water_evidence=water_added,
            full_three_evidence=evidence,
            retained_core_support=[n["gate_leave_one_module"] for n in lower["neighborhoods"]],
        )
        report["options"].append(status)
        if abs(med) <= 1.3 and light >= 0.65 and addition <= 15.0 and evidence >= -1.0:
            _, water_optics, _, _ = measured_bank(
                choice, prepared, bank, detail, 2, deep=True, compare_count=2, optical_starts=2
            )
            one, diagnostic = single_water(
                choice, before, lower, water_optics, prediction, weak, module, logn
            )
            if diagnostic is not None:
                status["one_water_comparison"] = diagnostic
            if one is not None:
                return dict(prediction, fractions=one), dict(report, applied=True, chosen=status)
    return prediction, report
