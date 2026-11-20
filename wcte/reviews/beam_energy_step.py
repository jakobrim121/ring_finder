"""Test abrupt energy changes along finite, connected hadron trajectories."""

import numpy as np

from finite_chains import factory, measured_axes, refine
from short_kink import matched


def proposals(competition, single, particle, optics):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from charge_starts import allowed_chord
    from joined_segments import arc
    from material import material
    from source import trajectory
    from value_only_receiver import seed_log_likelihood

    model = factory(competition.event, competition.model.bg, (particle, particle), 4)
    water = material(particle)
    optical_axes = measured_axes([g for _, g, _ in optics])
    points = [p for p in optical_axes if p[1][2] > -0.75]
    x = single["parameters"]
    chosen = {}
    length = x[9] * x[10]
    for fraction in (0.18, 0.35, 0.53, 0.72, 0.91):
        piece = arc(x[:12], 0.0, fraction)
        if piece is None:
            continue
        pos, direction, *_ = trajectory(piece, np.ones(1))
        vertex, tangent = pos[0] * 0.001, direction[0]
        remaining = 1000.0 * (x[9] - fraction * length)
        energy = water["energy_at"](remaining)
        options = [(tangent, 0.0)]
        for center, direction, opening, span in points:
            point_offset = center - vertex
            miss = np.linalg.norm(point_offset - (point_offset @ direction) * direction)
            if miss < 0.6 and all(direction @ old[0] < 0.994 for old in options):
                options.append((direction, span))
        for direction, span in options[:6]:
            water_path = allowed_chord(vertex, direction)
            if water_path is None or water_path[0] > 1e-5 or water_path[1] < 0.018:
                continue
            angle = float(np.degrees(np.arccos(np.clip(tangent @ direction, -1.0, 1.0))))
            for keep in (0.42, 0.68, 0.88, 0.998):
                e = energy * keep
                if e <= water["energy"][0] + 1.0:
                    continue
                ran = min(float(water["range_at"](e)) / 1000.0, water["maximum_range_mm"] * 0.000999)
                if ran <= 0.025:
                    continue
                for extent in (max(0.04, (0.9 - fraction) * length), span or ran):
                    trail = min(1.0, 0.997 * water_path[1] / ran, extent / ran)
                    child = np.r_[direction, np.zeros(3), ran, trail, x[11]]
                    full = canonical(model, np.r_[piece, child, x[-2:]])
                    if full is None:
                        continue
                    full = repair_energy(model, full)
                    if full is None or not feasible(model, full) or model.hardware_margin(full) <= 1e-5:
                        continue
                    group = angle >= 15.0, ran < 1.15 * water["margin_mm"] * 0.001
                    score = seed_log_likelihood(model, full)
                    if group not in chosen or score > chosen[group][0]:
                        chosen[group] = (score, model.particles, full, fraction)
    return sorted(chosen.values(), key=lambda r: -r[0])


def comparisons(competition, particle, optics):
    from coupled_model import Model

    old = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        beam, state, _, original = competition.finite_primary(particle)
        if beam is None or original["count"] != 1:
            return [], 0
        control = matched(competition, beam, state, ((0,),), "energy_step_continuous_single")
    finally:
        Model.delta_scale = old
    results = [] if control is None else [control]
    starts = proposals(competition, state, particle, optics)
    for seed in starts[:3]:
        result = refine(competition, seed, fine=True, physical_coverage=True)
        if result is not None:
            result["label"] = "energy_step_" + result["label"]
            results.append(result)
    return results, len(starts)
