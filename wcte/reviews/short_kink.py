"""Find nearby charged continuations hidden inside a single fitted cone."""

import numpy as np

from finite_chains import factory, refine
from light_competition import Template
from wcte.counting import track_counting


def starts(competition, particle, record):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical, split
    from value_only_receiver import seed_log_likelihood

    prototype = factory(competition.event, competition.model.bg, (particle, particle), 4)
    x = np.asarray(record["parameters"])
    rules = track_counting.get()
    rows = []
    for split_fraction in (0.12, 0.28, 0.46, 0.66, 0.84):
        divided = split(x[:12], split_fraction, particle, particle, 1.0)
        if divided is None:
            continue
        incoming, outgoing = divided
        tangent = outgoing[:3]
        side = np.cross(tangent, np.eye(3)[np.argmin(abs(tangent))])
        side /= np.linalg.norm(side)
        ortho = np.cross(tangent, side)
        for angle in np.radians([rules.turn_offset(offset) for offset in (0.0, 11.0, 26.0)]):
            for phi in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                transverse = np.cos(phi) * side + np.sin(phi) * ortho
                turned = outgoing.copy()
                turned[:3] = np.cos(angle) * tangent + np.sin(angle) * transverse
                turned[3:6] = 0.0
                guess = canonical(prototype, np.r_[x[:3], incoming, turned, x[-2:]])
                if guess is None:
                    continue
                guess = repair_energy(prototype, guess)
                if (
                    guess is None
                    or not feasible(prototype, guess)
                    or prototype.hardware_margin(guess) <= 1e-5
                ):
                    continue
                rows.append(
                    (seed_log_likelihood(prototype, guess), prototype.particles, guess, split_fraction)
                )
    return sorted(rows, key=lambda row: -row[0])


def matched(competition, model, state, groups, label, *, physical_coverage=False):
    from physical_pulse import components

    rates, centers = components(competition.event, model, state, separate_knockon=True, all_receivers=True)
    amps = rates.sum(0)
    support = tuple(tuple(1 + 2 * j for j in s) for s in groups)
    basis = Template(
        label,
        rates.T / np.maximum(amps, 1e-300)[:, None],
        centers,
        amps,
        support,
        np.zeros(len(amps) - 1, dtype=int),
        model.dimension,
        {},
    )
    return competition.fit(basis, iterations=22, fixed_ratios=True, physical_coverage=physical_coverage)


def experiments(competition, particle, *, use_smooth=True, visibility_probability=False):
    from coupled_model import Model

    previous = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        return compare(competition, particle, use_smooth, visibility_probability)
    finally:
        Model.delta_scale = previous


def compare(competition, particle, use_smooth, visibility_probability):
    from chain_sources import paths
    from smooth_beam_core import fit as fit_smooth
    from source import trajectory

    event = competition.event
    beam = competition.finite_primary(particle)
    if beam[0] is None or beam[3]["count"] != 1:
        return [], 0
    internal = matched(competition, beam[0], beam[1], ((0,),), "kink_finite_single")
    if internal is None:
        return [], 0
    output = [internal]
    screened = starts(competition, particle, beam[3])
    if not screened:
        return output, 0
    selected = screened[:1]
    for other in screened[1:]:
        if abs(other[3] - selected[0][3]) > 0.16 or other[2][12:15] @ selected[0][2][12:15] < 0.978:
            selected.append(other)
            break
    for candidate in selected:
        result = refine(competition, candidate, visibility_probability=visibility_probability)
        if result is None:
            continue
        p, child = [p for p, _ in paths(result["parameters"][:-2])]
        terminal = trajectory(p, np.ones(1))[1][0]
        turn = np.degrees(np.arccos(np.clip(terminal @ child[3:6], -1.0, 1.0)))
        result["label"] = "kink_" + result["label"]
        result.update(turn_deg=float(turn), seeded_fraction=candidate[3])
        output.append(result)
    if use_smooth and any(r["count"] == 2 for r in output):
        smooth = fit_smooth(event, competition.model.bg, beam[3], budget=48)
        if smooth[0] is not None and smooth[2].shape[1] == 2:
            strong = matched(competition, smooth[0], smooth[1], ((0, 1),), "kink_smooth_single")
            if strong is not None:
                output.append(strong)
    return output, len(screened)
