"""Compare a proton merger with a short chain of successive recoil vertices."""

import numpy as np

from aperture_partition_split import comparative_evidence
from calibrated_count_review import supported_shares
from finite_chains import factory, refine
from proton_vertex_residual import rank
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def nominated(lower):
    return (
        tuple(lower.get("particles") or ()) == ("proton", "proton")
        and tuple(tuple(g) for g in lower.get("lineage_sections", ())) == ((0, 1),)
        and 38.0 <= min(lower.get("individual_prompt_support", (0.0,))) <= 120.0
        and lower.get("order")
        and bool(np.asarray(lower.get("parameters", ())).size)
    )


def starts(competition, lower, optics):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical, split
    from chain_sources import paths
    from value_only_receiver import seed_log_likelihood

    x = np.asarray(lower["parameters"])
    prototype = factory(competition.event, competition.model.bg, ("proton",) * 3, 4)
    physical = [p for p, _ in paths(x[:-2])]
    axis_seeds = [np.asarray(d) for _, g, _ in optics for d in g["axes"] if d[2] >= 0.58]
    axis_seeds = [d for k, d in enumerate(axis_seeds) if all(d @ a < 0.996 for a in axis_seeds[:k])]
    result = []
    for leg, trajectory_seed in enumerate(physical):
        for fraction in (0.25, 0.42, 0.60, 0.76):
            new_sections = split(trajectory_seed, fraction, "proton", "proton", 1.0)
            if new_sections is None:
                continue
            leading, following = new_sections
            tangent = following[:3]
            side = np.cross(tangent, np.eye(3)[np.argmin(abs(tangent))])
            side /= np.linalg.norm(side)
            transverse = np.cross(tangent, side)
            angles = (
                (np.cos(a) * tangent + np.sin(a) * (np.cos(b) * side + np.sin(b) * transverse))
                for a in np.radians((12.0, 20.0, 29.0))
                for b in np.linspace(0, 2 * np.pi, 8, endpoint=False)
            )
            for turn in (*axis_seeds, *angles):
                if not 0.84 <= tangent @ turn < 0.9995:
                    continue
                following[:3], following[3:6] = turn, 0.0
                parts = [p[3:] for p in physical]
                parts[leg : leg + 1] = [leading, following]
                candidate = canonical(prototype, np.r_[x[:3], *parts, x[-2:]])
                if candidate is not None:
                    candidate = repair_energy(prototype, candidate)
                if (
                    candidate is not None
                    and feasible(prototype, candidate)
                    and prototype.hardware_margin(candidate) > 1.0e-5
                ):
                    result.append(
                        (seed_log_likelihood(prototype, candidate), prototype.particles, candidate, leg)
                    )
    choices = []
    for option in sorted(result, key=lambda r: -r[0]):
        source, trial = option[-1], option[2]
        if any(
            source == saved[-1]
            and trial[12:15] @ saved[2][12:15] > 0.989
            and trial[21:24] @ saved[2][21:24] > 0.989
            for saved in choices
        ):
            continue
        choices.append(option)
        if len(choices) >= 4:
            break
    return choices


def turn_geometry(x):
    from chain_sources import paths
    from source import trajectory

    sections = paths(x[:-2])
    values, gradients = [], []
    for (first, a), (second, b) in zip(sections, sections[1:]):
        _, tangent, _, derivative = trajectory(first, np.ones(1))
        values.append(tangent[0] @ second[3:6])
        slope = tangent[0] @ b[3:6] + second[3:6] @ derivative[0] @ a
        gradients.append(np.r_[slope, 0.0, 0.0])
    return np.asarray(values), np.asarray(gradients)


def fixed_direction(start, separated):
    from chain_sources import paths
    from source import trajectory

    rules = track_counting.get()
    x = start.copy()
    for j in range(2):
        a, b = [p for p, _ in paths(x[:-2])][j : j + 2]
        incoming = trajectory(a, np.ones(1))[1][0]
        transverse = b[3:6] - (incoming @ b[3:6]) * incoming
        if np.linalg.norm(transverse) < 1.0e-8:
            transverse = np.cross(incoming, np.eye(3)[np.argmin(abs(incoming))])
        separation = np.degrees(np.arccos(np.clip(incoming @ b[3:6], -1.0, 1.0)))
        desired = (
            max(separation, rules.turn_offset(1.5))
            if j == separated
            else min(separation, rules.turn_offset(-1.6))
        )
        angle = np.radians(desired)
        offset = 12 + 9 * j
        x[offset : offset + 3] = np.cos(angle) * incoming + np.sin(angle) * transverse / np.linalg.norm(
            transverse
        )
        x[offset + 3 : offset + 6] -= (x[offset + 3 : offset + 6] @ x[offset : offset + 3]) * x[
            offset : offset + 3
        ]
    return x


def profile(choice, start, separated):
    """Hold every path parameter free while profiling a fixed component grouping."""
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from lineage_support import compare as measured_lineage
    from track_coordinates import fit

    rules = track_counting.get()
    if separated is not None and rules.track_merge_angle_deg == 180.0:
        return None
    signs = np.asarray([-1.0 if i == separated else 1.0 for i in range(2)])
    limits = np.cos(np.radians(np.where(signs > 0.0, rules.turn_offset(-0.02), rules.turn_offset(0.02))))

    def chart(order):
        model = factory(choice.event, choice.model.bg, ("proton",) * 3, order)
        old, jacobian = model.inequality, model.inequality_jacobian
        model.inequality = lambda x: np.r_[old(x), signs * (turn_geometry(x)[0] - limits)]
        model.inequality_jacobian = lambda x: np.vstack([jacobian(x), signs[:, None] * turn_geometry(x)[1]])
        return model

    x = fixed_direction(start, separated)
    for order in (8, 16, 32):
        model = chart(order)
        x = canonical(model, x)
        if x is None or (x := repair_energy(model, x)) is None:
            return None
        _, x = fit(model, x, 34, 55, full_likelihood=True, check_initial=True)
        safe = chart(2 * order)
        if not feasible(safe, x) or safe.hardware_margin(x) <= 1.0e-5:
            return None
        result = safe(x, True)
        if abs(result["ll"] - model(x, True)["ll"]) > 0.01:
            continue
        evaluated = measured_lineage(choice, safe, result)
        if evaluated is not None and evaluated["count"] == (1 if separated is None else 2):
            evaluated.update(profiled_sections=separated, parameters=x)
            return evaluated
        return None
    return None


def compare(competition, prediction, lower, optics, *, prefer_on_tie=False):
    import light_competition

    audit = dict(eligible=bool(nominated(lower)), applied=False)
    if not audit["eligible"]:
        return None, audit
    # Compare every fit under the same exact, unjittered charge and timing law.
    choice = light_competition.Competition(
        competition.event, competition.model, prompt=True, module_charge=True
    )
    from short_kink import matched

    previous = factory(choice.event, choice.model.bg, lower["particles"], 2 * lower["order"])
    state = previous(lower["parameters"], True)
    base = matched(choice, previous, state, ((0, 1),), "continued_p_proton")
    if base is None:
        return None, audit
    sampled = starts(choice, lower, optics)
    candidates = [r for option in sampled if (r := refine(choice, option, lineage_union=True)) is not None]
    captured = []
    for trial in candidates[:2]:
        for group in (None, 0, 1):
            fitted = profile(choice, trial["parameters"], group)
            if fitted is not None:
                captured.append(fitted)
    candidates.extend(captured)
    hypotheses = [base] + [r for r in candidates if r["count"] == 1]
    out = audit["fits"] = []
    audit["starts"] = len(sampled)
    logn = np.log(len(choice.model.active))
    q = choice.q
    old = prediction["fractions"][:, 1:]
    all_charge = q @ old.sum(1)
    for trial in sorted(candidates, key=lambda t: -rank(t, logn)[1]):
        if trial["count"] != 2:
            continue
        shares = trial["fractions"]["prompt"]
        i, j, dice = match_charge_shares(prediction["fractions"], shares[:, 1:].T, q)
        retained = (q @ np.minimum(old[:, i], shares[:, j + 1])) / np.maximum(q @ old[:, i], 1.0e-12)
        total = q @ np.minimum(old.sum(1), shares[:, 1:].sum(1)) / max(all_charge, 1.0e-12)
        margin, held = comparative_evidence(trial, hypotheses, logn)
        matched_controls = [r for r in captured if r["count"] == 1]
        local_margin, local_held = (
            comparative_evidence(trial, matched_controls, logn) if matched_controls else (None, None)
        )
        gate = [p["gate_pmt_support"] for p in trial["neighborhoods"]]
        diffuse = [p["gate_leave_one_module"] for p in trial["neighborhoods"]]
        item = dict(
            groups=trial["lineage_sections"],
            individual=trial["individual_prompt_support"],
            score=margin.tolist(),
            held=held,
            matched=dice.tolist(),
            retained=retained.tolist(),
            foreground=float(total),
            gate=gate,
            distributed=diffuse,
        )
        if local_margin is not None:
            item["same_complexity_scores"] = local_margin.tolist()
            item["same_complexity_held"] = local_held
        out.append(item)
        strong = (
            min(margin) >= 18.0
            and held >= 12.0
            and total >= 0.88
            and min(gate) >= 40.0
            and min(diffuse) >= 30.0
            and min(retained) >= 0.50
        )
        comparable = (
            prefer_on_tie
            and local_margin is not None
            and "profiled_sections" in trial
            and min(margin) >= -2.0
            and held >= -5.0
            and total >= 0.96
            and min(gate) >= 60.0
            and min(diffuse) >= 45.0
            and min(retained) >= 0.35
            and max(retained) >= 0.90
            and min(trial["individual_prompt_support"]) >= track_counting.get().scaled_support(45.0)
        )
        if (
            (strong or comparable)
            and supported_shares(prediction, shares, "proton")
            and 0 in trial["lineage_sections"][0]
        ):
            audit["tied_preference"] = bool(comparable and not strong)
            return shares, dict(audit, applied=True)
    return None, audit
