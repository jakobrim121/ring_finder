"""Refit a short charged turn beside one independently retained optical core."""

import numpy as np

from calibrated_count_review import supported_shares
from beam_prefix import fit_prefix
from finite_chains import factory, refine
from light_competition import Competition, Template
from short_kink import starts


def aperture_distance(point, axis, aperture):
    if axis[2] < 0.45:
        return np.inf
    crossing = point[:2] + (1000.0 * aperture.z - point[2]) * axis[:2] / axis[2]
    return np.linalg.norm(crossing)


def path_templates(competition, shape, axis_time, amp, model, state, label, groups):
    from physical_pulse import components

    rates, time = components(competition.event, model, state, separate_knockon=True, all_receivers=True)
    total = rates[:, 1:].sum(0)
    sizes = np.r_[max(3.0, 0.04 * competition.model.total), amp, total]
    design = np.vstack([competition.model.bg, shape, rates[:, 1:].T / np.maximum(total[:, None], 1.0e-300)])
    clocks = np.column_stack([axis_time, time])
    cores = ((1,), *(tuple(2 + 2 * j for j in group) for group in groups))
    limits = {2: (0.73 * total.sum(), 1.27 * total.sum())}
    intensity = np.r_[0, 1, np.full(len(total), 2)]
    clock_groups = np.r_[0, np.ones(len(total), dtype=int)]
    return Template(label, design, clocks, sizes, cores, clock_groups, 9 + model.dimension, limits, intensity)


def compare(competition, particle, optics):
    from source import trajectory
    from staged_physics.primary import Aperture
    from chain_sources import paths
    from smooth_beam_core import fit as smooth

    current = next(((g, t) for _, g, t in optics if len(t.core_groups) == 2), None)
    if current is None:
        return []
    geometry, template = current
    vertices = np.asarray(geometry["vertices"])
    directions = np.asarray(geometry["axes"])
    aperture = Aperture()
    ranked = [aperture_distance(p, d, aperture) for p, d in zip(vertices, directions, strict=True)]
    leader = int(np.argmin(ranked))
    if ranked[leader] > 500.0:
        return []
    other = 1 - leader
    profile = template.amplitudes[[0, other + 1]] @ template.shapes[[0, other + 1]]
    found, _ = fit_prefix(competition.event, particle, profile)
    if found is None:
        return []
    beam, state = found
    record = dict(parameters=state["parameters"], particles=beam.particles)
    fixed = Competition(competition.event, beam, prompt=True)
    hypotheses = starts(fixed, particle, record)
    if not hypotheses:
        return []
    informative = hypotheses[:1]
    for s in hypotheses[1:]:
        prior = informative[0]
        if abs(s[-1] - prior[-1]) > 0.14 or s[2][12:15] @ prior[2][12:15] < 0.963:
            informative.append(s)
            break
    shape, clock, amount = (
        template.shapes[other + 1],
        template.center[:, other],
        template.amplitudes[other + 1],
    )
    single = path_templates(
        competition, shape, clock, amount, beam, state, "retained_optical_charged_one", ((0,),)
    )
    result = competition.fit(single, iterations=24, physical_coverage=True)
    output = [result] if result is not None else []
    for start in informative:
        trial = refine(fixed, start, visibility_probability=True)
        if trial is None or trial["count"] != 2:
            continue
        kernel = factory(competition.event, beam.bg, trial["particles"], 2 * trial["order"])
        state2 = kernel(trial["parameters"], True)
        routes = [p for p, _ in paths(trial["parameters"][:-2])]
        tangent = trajectory(routes[0], np.ones(1))[1][0]
        turn = float(np.degrees(np.arccos(np.clip(tangent @ routes[1][3:6], -1.0, 1.0))))
        model = path_templates(
            competition, shape, clock, amount, kernel, state2, "retained_optical_charged_pair", ((0,), (1,))
        )
        proposal = competition.fit(model, iterations=32, physical_coverage=True)
        if proposal is not None:
            proposal["turn_degrees"] = turn
            output.append(proposal)
    if any(r["count"] == 3 for r in output):
        one, fitted, _, _ = smooth(competition.event, beam.bg, record, budget=34)
        if one is not None:
            model = path_templates(
                competition, shape, clock, amount, one, fitted, "retained_optical_charged_smooth", ((0, 1),)
            )
            proposal = competition.fit(model, iterations=24, physical_coverage=True)
            if proposal is not None:
                output.append(proposal)
    return output


def eligible(prediction, original, particle):
    if particle != "pion" or prediction["fractions"].shape[1] != 3 or len(prediction["pmt_ids"]) < 240:
        return False
    values = original.get("initial_counter_scores") or ()
    if isinstance(values, dict):
        return values.get("3", values.get(3, 0.0)) >= 0.28
    return len(values) > 3 and values[3] >= 0.28


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from light_model_bank import measured_bank
    from proton_vertex_residual import rank
    from wctering.likelihood.geometry import common_background

    report = dict(applied=False, eligible=eligible(prediction, original, particle))
    if not report["eligible"]:
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    competition = Competition(event, receiver, prompt=True, module_charge=True)
    prior, optics, _, _ = measured_bank(competition, prepared, bank, original, 2, deep=True)
    candidates = compare(competition, particle, optics)
    controls = [r for r in prior + candidates if r["count"] == 2]
    if not controls:
        return prediction, report
    logn = np.log(len(receiver.active))
    best = np.max([rank(r, logn) for r in controls], axis=0)
    charge = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1:]
    mass = charge @ old
    tested = []
    report["options"] = tested
    for trial in sorted(candidates, key=lambda x: rank(x, logn)[1], reverse=True):
        if trial["count"] != 3 or trial.get("turn_degrees", 0.0) < 17.0:
            continue
        support = [n["gate_leave_one_module"] for n in trial["neighborhoods"]]
        shares = trial["fractions"]["prompt"]
        combined = shares[:, 1:].sum(1)
        retained = (charge @ np.minimum(combined[:, None], old)) / np.maximum(mass, 1.0e-10)
        gains = rank(trial, logn) - best
        withheld = []
        for r in controls:
            module_change = 0.65 * (trial["charge_by_module"] - r["charge_by_module"])
            module_change += 0.35 * (trial["joint_by_module"] - r["joint_by_module"])
            withheld.append(
                2.0 * (module_change.sum() - module_change.max(initial=0.0))
                - logn * (trial["dimension"] - r["dimension"])
            )
        row = dict(
            label=trial["label"],
            turn_degrees=trial["turn_degrees"],
            retained=retained.tolist(),
            gains=gains.tolist(),
            distributed_support=support,
            held_module=float(min(withheld)),
        )
        tested.append(row)
        if (
            min(gains) >= 60.0
            and min(support) >= 38.0
            and min(retained) >= 0.92
            and min(withheld) >= 35.0
            and supported_shares(prediction, shares, particle)
        ):
            return dict(prediction, fractions=shares), dict(report, applied=True, chosen=row)
    return prediction, report
