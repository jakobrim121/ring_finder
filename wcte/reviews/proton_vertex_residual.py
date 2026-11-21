"""Seed a finite proton continuation from angle and residual light at a measured vertex."""

import numpy as np

from finite_chains import factory, refine
from short_kink import matched
from wcte.counting import track_counting


def residual_directions(model, origin, direction, energies, span, score_gradient, axes):
    from material import material
    from timed_charged_light import projected_aperture

    proton = material("proton")
    tangent = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
    tangent /= np.linalg.norm(tangent)
    across = np.cross(direction, tangent)
    angle = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    azimuth = np.cos(angle)[:, None] * tangent + np.sin(angle)[:, None] * across
    rules = track_counting.get()
    turns = np.radians([rules.turn_offset(offset) for offset in (2.0, 9.0, 19.0, 31.0, 44.0)])
    rays = np.concatenate([np.cos(turn) * direction + np.sin(turn) * azimuth for turn in turns])
    measured = [axis for axis in axes if 0.46 < axis @ direction < rules.merge_cosine]
    if measured:
        rays = np.vstack([rays, measured])
    difference = model.pos - origin
    distance = np.maximum(np.linalg.norm(difference, axis=1), 10.0)
    projected = difference / distance[:, None]
    incidence = -np.einsum("ij,ij->i", projected, model.normals)
    area = projected_aperture(incidence[:, None])[:, 0] / (distance**2 + 1.0e4)

    # A cheap point response at the mean emission point is used only for
    # ranking seeds. The finalists use the whole finite path and detector law.
    threshold_range = proton["margin_mm"]
    light_rank = []
    for e, length in zip(energies, span, strict=True):
        range_ = float(proton["range_at"](e))
        bulk_energy = float(proton["energy_at"](range_ - 0.28 * (range_ - threshold_range)))
        speed = np.sqrt(1.0 - (proton["mass"] / (proton["mass"] + bulk_energy)) ** 2)
        opening = np.arccos(np.clip(1.0 / (speed * proton["index"]), 0.0, 1.0))
        along = difference @ rays.T - 0.22 * length
        radial = np.sqrt(np.maximum(distance[:, None] ** 2 - (along + 0.22 * length) ** 2, 0.0))
        observed = np.arctan2(radial, along)
        rate = area[:, None] * np.exp(-0.5 * ((observed - opening) / np.radians(3.2)) ** 2)
        rate /= np.maximum(rate.sum(0), 1.0e-300)
        contrast = -score_gradient @ rate
        for j in np.argsort(contrast)[-5:][::-1]:
            light_rank.append((float(contrast[j]), rays[j], float(e)))
    return sorted(light_rank, key=lambda x: -x[0])[:5]


def starts(competition, record, optical, *, maximum=3):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical, split
    from charge_starts import allowed_chord
    from material import material
    from source import trajectory
    from value_only_receiver import receiver_values, seed_log_likelihood

    event, model = competition.event, competition.model
    proton = material("proton")
    margin = float(proton["energy_at"](proton["margin_mm"]))
    prototype = factory(event, model.bg, ("proton", "proton"), 4)
    x = np.asarray(record["parameters"])
    path = x.copy()[:12]
    extent = 1000.0 * path[9] * path[10]
    available = min(extent, 1000.0 * path[9] - proton["margin_mm"])
    if available < 115.0:
        return []
    directions = [np.asarray(axis) for _, g, _ in optical for axis in g["axes"]]
    response, noise = np.exp(x[-2:])
    screened = []
    for depth in available * np.asarray([0.21, 0.33, 0.45, 0.58, 0.70, 0.81, 0.91]):
        fraction = depth / extent
        divided = split(path, fraction, "proton", "proton", 1.0)
        if divided is None:
            continue
        leading, trailing = divided
        prefix = np.r_[path[:3], leading]
        origin, tangent, *_ = trajectory(prefix, np.ones(1))
        origin, tangent = origin[0], tangent[0]
        rest = float(proton["energy_at"](1e3 * prefix[9] * (1.0 - prefix[10])))
        if rest < margin + 18.0:
            continue
        energies = margin + (rest - margin) * np.array([0.24, 0.53, 0.86])
        span = proton["range_at"](energies)
        rate = receiver_values(model.pos, model.normals, prefix, 5, "proton").sum(0)
        mean = noise * model.bg + response * depth * rate
        _, slope = model.obs.score(mean, model.ix)
        directions_here = residual_directions(model, origin, tangent, energies, span, slope, directions)
        for _, axis, kinetic in directions_here:
            chord = allowed_chord(origin / 1000.0, axis)
            if chord is None or chord[0] > 1.0e-6 or chord[1] < 0.020:
                continue
            candidate = trailing.copy()
            candidate[:3] = axis
            candidate[3:6] = 0.0
            candidate[6] = 0.001 * proton["range_at"](kinetic)
            candidate[7] = min(1.0, 0.996 * chord[1] / candidate[6])
            initial = canonical(prototype, np.r_[path[:3], leading, candidate, x[-2:]])
            if initial is None:
                continue
            initial = repair_energy(prototype, initial)
            if (
                initial is not None
                and feasible(prototype, initial)
                and prototype.hardware_margin(initial) > 1.0e-5
            ):
                screened.append(
                    (seed_log_likelihood(prototype, initial), prototype.particles, initial, fraction)
                )
    chosen = []
    for candidate in sorted(screened, key=lambda row: -row[0]):
        if all(
            abs(r[-1] - candidate[-1]) > 0.11 or r[2][12:15] @ candidate[2][12:15] < 0.970 for r in chosen
        ):
            chosen.append(candidate)
        if len(chosen) >= maximum:
            break
    return chosen


def compare(competition, optical, *, constrained=True):
    from coupled_model import Model
    from matched_turn_split import qualify
    from smooth_beam_core import fit as fit_smooth

    prior = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        beam, state, _, record = competition.finite_primary("proton")
        if beam is None or record["count"] != 1:
            return [], 0
        plans = starts(competition, record, optical)
        if not plans:
            return [], 0
        values = [matched(competition, beam, state, ((0,),), "proton_vertex_control")]
        for plan in plans:
            fit = refine(competition, plan, physical_coverage=True, visibility_probability=True)
            if constrained and (fit is None or fit["count"] != 2):
                fit = qualify(competition, plan[1], plan[2], separate=True, aperture=True)
            if fit is not None and fit["count"] == 2:
                fit["label"] = "proton_vertex_" + fit["label"]
                values.append(fit)
        if len(values) > 1:
            one = fit_smooth(competition.event, competition.model.bg, record, budget=45)
            if one[0] is not None and one[2].shape[1] == 2:
                values.append(matched(competition, one[0], one[1], ((0, 1),), "proton_vertex_smooth"))
        return [value for value in values if value is not None], len(plans)
    finally:
        Model.delta_scale = prior


def rank(trial, logn):
    q, t = trial["charge"], trial["joint"]
    return 2.0 * np.asarray((q, 0.65 * q + 0.35 * t, t)) - logn * trial["dimension"]


def measured_pair(trial, before):
    from calibrated_count_review import supported_shares
    from shared_scoring import match_charge_shares

    previous = before["fractions"]
    shares = trial["fractions"]["prompt"]
    q = np.minimum(before["charges"], 2.0)
    _, indices, similarity = match_charge_shares(previous, shares[:, 1:].T, q)
    leading = 1 + int(indices[0])
    additional = 3 - leading
    foreground = previous[:, 1]
    total = max(q @ foreground, 1.0e-10)
    retained = q @ np.minimum(foreground, shares[:, leading]) / total
    retained_signal = q @ np.minimum(foreground, shares[:, 1:].sum(1)) / total
    independent = q * shares[:, additional] * previous[:, 0]
    modules = np.bincount(before["pmt_ids"] // 100, weights=independent)
    distributed = independent.sum() - modules.max(initial=0.0)
    result = dict(
        incoming_preserved=bool(supported_shares(before, shares, "proton")),
        support=[r["gate_pmt_support"] for r in trial["neighborhoods"]],
        distributed=[r["gate_leave_one_module"] for r in trial["neighborhoods"]],
        retained=float(retained),
        retained_signal=float(retained_signal),
        independent_pe=float(distributed),
        incumbent_dice=float(similarity[0]),
    )
    return result, shares[:, (0, leading, additional)]


def have_multiring_evidence(original):
    probabilities = original.get("initial_counter_scores")
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(str(i), probabilities.get(i, 0.0)) for i in range(4)]
    return probabilities is not None and len(probabilities) == 4 and sum(probabilities[2:]) >= 0.025


def review(captured, before, original):
    """Recover resolved proton continuations under two independent detector scores."""
    from coupled_model import Model
    from light_competition import Competition
    from light_model_bank import measured_bank
    from wctering.likelihood.geometry import common_background

    empty = dict(applied=False, eligible=False)
    if before["fractions"].shape[1] != 2 or not have_multiring_evidence(original):
        return (before, empty)
    event, prepared, bank = (captured[n] for n in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return (before, empty)
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True, primary_models=captured.get("primary_models"))
    existing, optics, _, _ = measured_bank(choice, prepared, bank, original, 1, deep=True, compare_count=1)
    plans, tried = compare(choice, optics, constrained=False)
    result = dict(eligible=True, applied=False, starts=tried)
    logn = np.log(len(receiver.active))
    one = [rank(t, logn) for t in existing + plans if t["count"] == 1]
    if not one:
        return (before, result)
    control = np.max(one, axis=0)
    ranked = sorted((t for t in plans if t["count"] == 2), key=lambda t: rank(t, logn)[1], reverse=True)
    for trial in ranked:
        margin = rank(trial, logn) - control
        audit, shares = measured_pair(trial, before)
        audit.update(label=trial["label"], detector_gain=margin.tolist())
        if "label" not in result:
            result.update(audit)
        if (
            not audit["incoming_preserved"]
            or audit["retained"] < 0.72
            or audit["retained_signal"] < 0.9
            or (min(audit["distributed"]) < 25.0)
            or (min(audit["support"]) < track_counting.get().scaled_support(35.0))
        ):
            continue
        independently_owned = margin.min() >= 30.0 and audit["independent_pe"] >= 15.0
        consistently_split = (
            margin.min() >= 45.0 and min(audit["support"]) >= track_counting.get().scaled_support(45.0)
        )
        if independently_owned or consistently_split:
            folded = folded_proton(choice, before, trial, logn)
            if folded is not None:
                result.setdefault("folded_controls", []).append(folded)
                if folded["holds"]:
                    continue
            return (dict(before, fractions=shares), dict(result | audit, applied=True, count=2))
    from aperture_wrong_mass_prefix import compare as compare_prefix, nominated

    if nominated(original, len(event.pmt_ids)):
        prefixes, examined = compare_prefix(choice, optics, original)
        result["short_prefix_starts"] = examined
        for trial in sorted(
            (t for t in prefixes if t["count"] == 2), key=lambda t: rank(t, logn)[1], reverse=True
        ):
            margin = rank(trial, logn) - control
            audit, shares = measured_pair(trial, before)
            if (
                margin.min() >= 30.0
                and audit["incoming_preserved"]
                and (audit["retained"] >= 0.8)
                and (audit["retained_signal"] >= 0.9)
                and (audit["independent_pe"] >= 18.0)
                and (min(audit["support"]) >= track_counting.get().scaled_support(45.0))
                and (min(audit["distributed"]) >= 35.0)
            ):
                return (
                    dict(before, fractions=shares),
                    dict(
                        result | audit,
                        applied=True,
                        count=2,
                        label=trial["label"],
                        detector_gain=margin.tolist(),
                    ),
                )
    from elastic_proton_recoil import revisit

    result["elastic_recoil"] = []
    for trial in ranked[:1]:
        audit, _ = measured_pair(trial, before)
        if (
            not audit["incoming_preserved"]
            or audit["retained"] < 0.9
            or audit["retained_signal"] < 0.97
            or (min(audit["support"]) < track_counting.get().min_track_pmts)
            or ((rank(trial, logn) - control).min() < 6.6349)
        ):
            continue
        diagnostic = {}
        adjusted = revisit(choice, trial, diagnostic)
        if adjusted is None:
            result["elastic_recoil"].append(diagnostic)
            continue
        margin = rank(adjusted, logn) - control
        audit, shares = measured_pair(adjusted, before)
        audit["detector_gain"] = margin.tolist()
        for key in ("label", "outgoing_kinetic", "energy_residual", "other_kinetic", "outgoing_angle"):
            audit[key] = adjusted[key]
        audit.update(diagnostic)
        result["elastic_recoil"].append(audit)
        if (
            audit["incoming_preserved"]
            and audit["retained"] >= 0.9
            and (audit["retained_signal"] >= 0.97)
            and (audit["independent_pe"] >= 12.0)
            and (min(audit["support"]) >= track_counting.get().scaled_support(40.0))
            and (min(audit["distributed"]) >= 30.0)
            and (margin.min() >= 6.6349)
        ):
            return (dict(before, fractions=shares), dict(result | audit, applied=True, count=2))
    return (before, result)


def folded_proton(choice, before, split, logn):
    """Compare a joined two-section recoil with the same broken-track seed."""
    from matched_turn_split import qualify, seed_turn, turn_degrees

    x = np.asarray(split["parameters"])
    probes = [min(13.8, turn_degrees(x)), 5.0]
    alternatives = [
        r
        for a in probes
        if (r := qualify(choice, ("proton", "proton"), seed_turn(x, a), separate=False, aperture=True))
        is not None
    ]
    if not alternatives:
        return None
    joined = max(alternatives, key=lambda p: rank(p, logn)[1])
    q = np.minimum(before["charges"], 2.0)
    old = split["fractions"]["prompt"][:, 1:]
    light = joined["fractions"]["prompt"][:, 1]
    retained = (q @ np.minimum(old, light[:, None])) / np.maximum(q @ old, 1.0e-12)
    original = before["fractions"][:, 1]
    previous = float(q @ np.minimum(original, light) / max(q @ original, 1.0e-12))
    evidence = rank(joined, logn) - rank(split, logn)
    by_module = joined["joint_by_module"] - split["joint_by_module"]
    held = float(
        2.0 * (by_module.sum() - by_module.max(initial=0.0))
        - logn * (joined["dimension"] - split["dimension"])
    )
    module_ids = before["pmt_ids"] // 100
    left = q[:, None] * np.maximum(old - light[:, None], 0.0)
    outside = [float(x.sum() - np.bincount(module_ids, weights=x).max(initial=0.0)) for x in left.T]
    nearly_collinear = (
        joined["turn_deg"] <= track_counting.get().turn_offset(-6.0)
        and min(evidence) >= 32.0
        and held >= 20.0
        and min(retained) >= 0.86
        and max(outside) <= 14.0
    )
    well_matched = min(retained) >= 0.90 and min(evidence) >= 8.0 and held >= 5.0
    return dict(
        turn_deg=joined["turn_deg"],
        retained=retained.tolist(),
        incumbent=previous,
        evidence=evidence.tolist(),
        held=held,
        unexplained_outside_best_module=outside,
        holds=bool(previous >= 0.94 and (nearly_collinear or well_matched)),
    )
