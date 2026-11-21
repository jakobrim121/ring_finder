"""Model a long beam muon and the PMTs directly adjacent to its fitted exit."""

from dataclasses import replace

import numpy as np

from light_competition import Template
from wcte.counting import track_counting


def chord_geometry(line):
    from staged_physics.primary import Aperture

    anchor, direction = (np.asarray(line[name], dtype=float) for name in ("anchor_m", "axis"))
    if direction[2] < 0.8:
        return None
    aperture = Aperture()
    entry = anchor + ((aperture.z - anchor[2]) / direction[2]) * direction
    if (
        np.linalg.norm(entry[:2]) > aperture.radius + 0.12
        or line["chord_m"] < 2.35
        or line["coherent_longitudinal_span_m"] < 0.62 * line["chord_m"]
        or line["coherent_coverage"] < 0.50
    ):
        return None
    return anchor, direction


def exit_receiver_weights(competition, line):
    """Limit the influence of very bright tubes in the fitted exit region."""
    geometry = chord_geometry(line)
    if geometry is None:
        return None
    anchor, direction = geometry
    exit_distance = max(hi for _, hi in line["emitting_intervals"])
    exit_point = 1000.0 * (anchor + exit_distance * direction)
    model = competition.model
    close = np.linalg.norm(model.pos[model.ix] - exit_point, axis=1) < 525.0
    window = close & (model.q > 4.0)
    if not window.any():
        return None
    weight = np.ones(len(model.active))
    weight[model.ix[window]] = 4.0 / model.q[window]
    return weight


def templates(competition, line):
    from wctering.receiver.angular_panels import finite_integrate
    from wctering.receiver.finite_source import quadrature
    from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE

    coordinates = chord_geometry(line)
    if coordinates is None:
        return None, None
    anchor, direction = coordinates
    model = competition.model
    theta, width = np.radians([line["opening_deg"], line["width_deg"]])
    relative = model.pos - 1000.0 * anchor
    along = relative @ direction
    radial2 = np.maximum((relative * relative).sum(1) - along**2, 0.0)
    emission = along - np.sqrt(radial2) / np.tan(theta)
    beta_inverse = max(1.0, N_PHASE * np.cos(theta))
    fields = []
    times = []
    for left, right in line["emitting_intervals"]:
        count = max(1, int(np.ceil((right - left) / 0.60)))
        edges = np.linspace(left, right, count + 1)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if hi - lo < 0.01:
                continue
            centre = anchor + 0.5 * (lo + hi) * direction
            parameters = np.r_[centre, direction, hi - lo, theta, np.log(width)]
            value, _, _ = finite_integrate(model.pos, model.normals, parameters, *quadrature(16))
            if value.sum() <= 1e-20:
                continue
            u = np.clip(emission, 1000.0 * lo, 1000.0 * hi)
            distance = np.sqrt((along - u) ** 2 + radial2)
            times.append((beta_inverse * u + N_GROUP * distance) / C)
            fields.append((hi - lo) * value)
    if len(fields) < 3:
        return None, None
    physical = np.stack(fields)
    integral = physical.sum(1)
    background, amplitude = line["amplitudes"]
    starts = amplitude * integral / integral.sum()
    caps = {j + 1: (0.4 * p, 2.2 * p) for j, p in enumerate(starts)}
    clock = np.zeros(len(fields), dtype=int)
    path = Template(
        "one_long_muon_chord",
        np.vstack([model.bg, physical / integral[:, None]]),
        np.column_stack(times),
        np.r_[background, starts],
        (tuple(range(1, 1 + len(fields))),),
        clock,
        9 + len(fields),
        caps,
    )

    # Modeled module light is restricted to the hardware actually adjacent
    # to the forward end of the detector-derived trajectory.
    exit_s = max(hi for _, hi in line["emitting_intervals"]) * 1000.0
    exit_point = 1000.0 * anchor + exit_s * direction
    distance = np.linalg.norm(model.pos - exit_point, axis=1)
    plausible = (along >= exit_s - 160.0) & (along <= exit_s + 420.0) & (radial2 < 270.0**2)
    if not plausible.any():
        return path, None
    nearest = np.argmin(np.where(plausible, distance, np.inf))
    module = int(model.active[nearest] // 100)
    module_tubes = model.active // 100 == module
    local = np.exp(-0.5 * radial2 / 95.0**2) * module_tubes
    local /= local.sum()
    observed = float(model.q[module_tubes[model.ix]].sum() / model.obs.gain)
    parent_here = float(np.sum(starts @ path.shapes[1:, module_tubes]))
    excess = max(0.0, observed - parent_here)
    if excess < 12.0 or observed < 1.5 * parent_here:
        return path, None
    clock = (beta_inverse * along + N_GROUP * np.sqrt(radial2)) / C
    return path, dict(
        module=module, shape=local, center=clock, start=excess, limit=min(2.0 * excess, model.total)
    )


def add_exit(template, local, *, joined):
    """Supply the same localized response to each count hypothesis."""
    index = len(template.amplitudes)
    groups = (template.core_groups[0] + (index,),) if joined else template.core_groups
    time_group = 0 if joined else template.time_groups.max() + 1
    return replace(
        template,
        label=template.label + "_exit_light",
        shapes=np.vstack([template.shapes, local["shape"]]),
        center=np.column_stack([template.center, local["center"]]),
        amplitudes=np.r_[template.amplitudes, local["start"]],
        core_groups=groups,
        time_groups=np.r_[template.time_groups, time_group],
        dimensions=template.dimensions + (1 if joined else 2),
        limits=dict(template.limits) | {index: local["limit"]},
    )


def finite_with_exit(competition, line, model, state, local):
    """Attach the same exit response to the slowing muon's shared flight clock."""
    from chain_sources import paths
    from material import material
    from physical_pulse import components
    from source import trajectory
    from wctering.seeding.coherent_timing import C, N_GROUP

    anchor, axis = chord_geometry(line)
    end = 1000.0 * (anchor + max(hi for _, hi in line["emitting_intervals"]) * axis)
    ((muon, _),) = paths(state["parameters"][: model.ng])
    stop, direction, *_ = trajectory(muon, np.ones(1))
    gap = end - stop[0]
    forward = float(gap @ direction[0])
    if (
        abs(forward) > 240.0
        or np.linalg.norm(gap - forward * direction[0]) > 140.0
        or direction[0] @ axis < 0.97
    ):
        return None
    rate, time, elapsed = components(
        competition.event, model, state, separate_knockon=True, all_receivers=True, return_elapsed=True
    )
    table = material("muon")
    rest = float(table["energy_at"](1000.0 * muon[9] * (1.0 - muon[10])))
    beta = np.sqrt(1.0 - (table["mass"] / (table["mass"] + rest)) ** 2)
    if beta < 0.75:
        return None
    dt = elapsed + forward / (beta * C)
    distance = np.linalg.norm(competition.model.pos - end, axis=1)
    adjusted = dict(local, center=dt + N_GROUP * distance / C)
    mass = rate.sum(0)
    shape = rate.T / np.maximum(mass[:, None], 1.0e-300)
    template = Template(
        "one_muon_finite", shape, time, mass, ((1,),), np.zeros(rate.shape[1] - 1, int), model.dimension, {}
    )
    trial = add_exit(template, adjusted, joined=True)
    groups = np.r_[0, np.ones(len(mass) - 1, dtype=int), 2]
    # Keep one amplitude for the finite muon and one for the module light.
    trial = replace(trial, intensity_groups=groups, limits={2: trial.limits[len(mass)]})
    return competition.fit(trial, iterations=42)


def review(
    captured,
    prediction,
    detail,
    *,
    compare_finite_exit=False,
    occupancy=False,
    guarded_occupancy=False,
):
    """Resolve a fragmented long muon only when its measured light is retained."""
    from coupled_model import Model
    from light_competition import Competition
    from light_model_bank import measured_bank
    from short_kink import matched
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=False, applied=False)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    chord = prepared.get("rejected_chord") or {}
    line = chord.get("chord")
    if (
        line is None
        or "1" not in bank
        or not event.timing_selection.get("prompt_window_ns")
        or line["coherent_coverage"] < 0.80
        or line["chord_end_support"][1] < 0.035
    ):
        return prediction, report
    if guarded_occupancy and (prediction["fractions"].shape[1] != 4 or line["chord_m"] < 2.70):
        return prediction, report
    occupancy = occupancy or guarded_occupancy
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, model, prompt=True, after_first_jitter=not occupancy)
    path, local = templates(choice, line)
    if path is None:
        return prediction, report
    weight = exit_receiver_weights(choice, line)
    if weight is not None:
        choice = Competition(
            event, model, prompt=True, after_first_jitter=not occupancy, receiver_weights=weight
        )
    if occupancy:
        from occupancy_pulse import create

        choice.selection = create(event, model)
    report.update(
        eligible=True,
        chord_extent_m=line["chord_m"],
        exit_prompt_fraction=line["chord_end_support"][1],
        bright_exit_receivers=int(np.count_nonzero(weight < 1.0)) if weight is not None else 0,
    )
    alternatives = [choice.fit(path, iterations=35)]
    if local is not None:
        alternatives.append(choice.fit(add_exit(path, local, joined=True), iterations=42))
        report["exit_module"] = local["module"]
    former = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        fitted, state, _, summary = choice.finite_primary("muon")
        if fitted is not None and summary["count"] == 1:
            alternatives.append(matched(choice, fitted, state, ((0,),), "one_muon_finite"))
            if compare_finite_exit and local is not None:
                variant = finite_with_exit(choice, line, fitted, state, local)
                if variant is not None:
                    alternatives.append(variant)
                    report["tried_finite_exit"] = True
    finally:
        Model.delta_scale = former

    compare_count = None if guarded_occupancy else 3
    banked, optics, _, _ = measured_bank(
        choice, prepared, bank, detail, 3, deep=True, compare_count=compare_count
    )
    counts = (2, 3) if guarded_occupancy else (3,)
    incumbent = [r for r in banked if r["count"] in counts]
    if guarded_occupancy:
        qualified = [
            r
            for r in incumbent
            if all(
                n["gate_pmt_support"] >= track_counting.get().min_track_pmts
                and n["gate_leave_one_module"] >= track_counting.get().scaled_support(20.0)
                for n in r["neighborhoods"]
            )
        ]
        if {r["count"] for r in qualified} != set(counts):
            return prediction, dict(report, reason="missing supported two- or three-ring comparison")
    if local is not None:
        for _, _, optical in optics:
            if len(optical.core_groups) in counts:
                adjusted = choice.fit(
                    add_exit(optical, local, joined=False), iterations=42, score_only=not guarded_occupancy
                )
                if adjusted is not None:
                    incumbent.append(adjusted)
    if not incumbent:
        return prediction, report
    penalty = np.log(len(model.active))

    def score(r):
        if occupancy:
            return 2.0 * r["joint"] - penalty * r["dimension"]
        return 2.0 * (0.65 * r["charge"] + 0.35 * r["joint"]) - penalty * r["dimension"]

    comparator = max(map(score, incumbent))
    if guarded_occupancy:
        strongest = {k: max((r for r in incumbent if r["count"] == k), key=score) for k in counts}
        report["competitors"] = {k: r["label"] for k, r in strongest.items()}
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1:]
    for proposal in sorted((r for r in alternatives if r is not None), key=score, reverse=True):
        gain = float(score(proposal) - comparator)
        core = proposal["fractions"]["prompt"][:, 1]
        covered = (q @ np.minimum(old, core[:, None])) / np.maximum(q @ old, 1e-12)
        shared = q @ np.minimum(old.sum(1), core)
        support = proposal["neighborhoods"][0]["gate_pmt_support"]
        same_light = min(shared / max(q @ old.sum(1), 1e-12), shared / max(q @ core, 1e-12))
        audit = dict(
            margin=gain,
            retained_old_cores=covered.tolist(),
            shared_signal=float(same_light),
            gate_support=support,
            hypothesis=proposal["label"],
        )
        if guarded_occupancy:
            audit["margin_by_count"] = {k: float(score(proposal) - score(r)) for k, r in strongest.items()}
            robustness = []
            for rival in strongest.values():
                module_gain = proposal["joint_by_module"] - rival["joint_by_module"]
                complexity = penalty * (proposal["dimension"] - rival["dimension"])
                robustness.append(
                    float(2.0 * (module_gain.sum() - module_gain.max(initial=0.0)) - complexity)
                )
            audit["margin_without_best_module"] = min(robustness)
        if "margin" not in report:
            report.update(audit)
        if guarded_occupancy:
            accept = gain >= 40.0 and min(covered) >= 0.97 and same_light >= 0.95 and min(robustness) >= 0.0
        else:
            accept = gain >= 6.6349 and min(covered) >= 0.90 and same_light >= 0.90
        if accept and support >= 100.0:
            return dict(prediction, fractions=proposal["fractions"]["prompt"]), dict(
                report | audit, applied=True
            )
    return prediction, report
