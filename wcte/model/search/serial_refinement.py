"""Propose a missing charged leg at a local interaction on a fitted path."""

import numpy as np

from bare_model import Model as Bare
from balanced_search import feasible
from bare_residual_search import residual_scores
from charge_starts import allowed_chord, recoil_start
from material import material
from residual_search import axis_grid
from source import raw


def leg_starts(model, point, axis, opening, amp, span, mass):
    chord = allowed_chord(point, axis)
    if chord is None or chord[0] > 1e-5 or chord[1] < 0.012:
        return
    table = material(mass)
    theta = min(opening, np.arccos(1 / table["index"]) - 0.004)
    speed = 1 / (table["index"] * np.cos(theta))
    energy = table["mass"] * (1 / np.sqrt(1 - speed**2) - 1)
    if mass == "electron":
        energy = 25.0 + 0.15 * amp
    energy = np.clip(energy, table["energy"][0] + 1.01, table["energy"][-1])
    middle = float(table["range_at"](energy)) / 1000
    width = np.log(np.deg2rad(1.4))
    thin = np.r_[point, axis, np.zeros(3), middle, 0.0001, width]
    light = model.response * 1000 * raw(model.pos, model.normals, thin, 1, False, mass)[0].sum()
    maximum = table["maximum_range_mm"] / 1000 * (1.0 - 1e-10)
    for extent in (amp / max(light, 1e-8), span):
        length = float(np.clip(extent, 0.01, min(chord[1], maximum * 0.98)))
        ran = min(max(1.002 * length, middle + 0.5 * length), maximum)
        yield np.r_[axis, np.zeros(3), ran, length / ran, width]


def prepend(event, model, state, *, movable=False):
    """Look behind the dominant track for a short incoming charged section."""
    if len(model.particles) >= 4:
        return []
    x = state["parameters"]
    point, outgoing = x[:3], x[3:6]
    model(x, True)
    means = np.exp(x[-2]) * model.primary_cache[1] + np.exp(x[-1]) * model.bg
    _, residual = model.obs.score(means, model.ix)
    nuisance = np.vstack([model.primary_cache[1] / max(model.primary_cache[1].sum(), 1e-30), model.bg])
    axes = axis_grid(512)
    axes = axes[(axes @ outgoing >= -0.65) & (axes @ outgoing < 0.98)]
    candidates = []
    chart = [point]
    if movable:
        # The fitted ring constrains the line well while the start is
        # uncertain. Scan nearby join positions for a first section.
        side = np.cross(outgoing, np.eye(3)[np.argmin(abs(outgoing))])
        side /= np.linalg.norm(side)
        across = np.cross(outgoing, side)
        chart.extend(point + d * outgoing for d in (-0.16, 0.08, 0.20, 0.36))
        chart.extend(point + 0.1 * outgoing + d * axis for d in (-0.16, 0.16) for axis in (side, across))
    for junction in chart:
        if not movable or (allowed_chord(junction, outgoing) is not None):
            rays = model.pos - 1000 * junction
            distance = np.maximum(np.linalg.norm(rays, axis=1), 1e-9)
            rays /= distance[:, None]
            exposure = np.maximum(-np.sum(model.normals * rays, axis=1), 0.02) / (distance**2 + 10000.0)
            polar = np.arccos(np.clip(axes @ rays.T, -1.0, 1.0))
            for opening in np.deg2rad([22.0, 30.0, 36.0, 41.0]):
                shape = exposure * np.exp(-0.5 * ((polar - opening) / np.deg2rad(4.5)) ** 2)
                shape /= np.maximum(shape.sum(1), 1e-300)[:, None]
                slope, information, _ = residual_scores(shape, means, residual, nuisance)
                score = np.where(
                    information > 1e-16, np.maximum(slope, 0.0) ** 2 / np.maximum(information, 1e-16), 0.0
                )
                for j in np.argsort(score)[-3:]:
                    if score[j] > 8.0:
                        candidates.append(
                            (
                                float(score[j]),
                                junction,
                                axes[j],
                                float(opening),
                                float(slope[j] / information[j]),
                            )
                        )
    results = []
    used = []
    for _, junction, axis, opening, photons in sorted(candidates, key=lambda r: r[0], reverse=True):
        if movable and any(np.linalg.norm(junction - v) < 0.12 and axis @ a > 0.95 for v, a in used):
            continue
        used.append((junction, axis))
        if len(used) > 4:
            break
        reverse = allowed_chord(junction, -axis)
        if reverse is None or reverse[0] > 1e-5 or reverse[1] < 0.012:
            continue
        # Restrict only the proposed upstream extent. The fit may change it.
        longest = min(0.32, reverse[1])
        for mass in dict.fromkeys((model.particles[0], "pion", "proton", "muon")):
            if mass == "electron":
                continue
            table = material(mass)
            beta = 1 / (table["index"] * np.cos(min(opening, np.arccos(1 / table["index"]) - 0.004)))
            energy = np.clip(
                table["mass"] * (1 / np.sqrt(1 - beta**2) - 1), table["energy"][0] + 1.01, table["energy"][-1]
            )
            ran = float(table["range_at"](energy)) / 1000
            width = np.log(np.deg2rad(1.4))
            probe = np.r_[junction, axis, np.zeros(3), ran, 0.0001, width]
            response = model.response * 1000 * raw(model.pos, model.normals, probe, 1, False, mass)[0].sum()
            length = float(np.clip(photons / max(response, 1e-8), 0.008, longest))
            ran = max(ran + 0.5 * length, 1.002 * length)
            particles = (mass, *model.particles)
            trial = Bare(event.pmt_ids, event.charges_pe, model.bg, particles, 4, 8)
            start = np.r_[junction - length * axis, axis, np.zeros(3), ran, length / ran, width, x[3:]]
            start = recoil_start(trial, start)
            if start is None or not feasible(trial, start) or trial.hardware_margin(start) <= 1e-5:
                continue
            ll = trial(start, True)["log_joint"]
            results.append((float(2 * ll - trial.dimension * np.log(len(trial.active))), particles, start))
    return results
