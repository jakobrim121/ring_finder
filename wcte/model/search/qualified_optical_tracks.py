"""Seed calibrated charged trajectories from a measured finite cone."""

import numpy as np

from charge_starts import allowed_chord
from geometry import hardware_margin
from material import material
from recoil_competitor import render
from wctering.seeding.coherent_timing import N_PHASE


def paths_for_cone(model, cone, photons):
    """Match finite optical extent and brightness to a material's light yield."""
    point, axis = cone[:3], cone[3:6]
    water = allowed_chord(point, axis)
    if water is None:
        return
    available = water[1] - water[0]
    if available < 0.012:
        return
    width = np.clip(cone[-1], *model.bounds[11])
    for particle in ("electron", "muon", "pion", "proton"):
        m = material(particle)
        low = float(m["energy_at"](m["margin_mm"]))
        max_range = m["maximum_range_mm"] / 1000.0
        angle = min(cone[7], np.arccos(1.0 / m["index"]) - 0.004)
        beta = min(1.0 / (N_PHASE * np.cos(angle)), 0.999)
        apparent = m["mass"] * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
        if particle == "electron":
            apparent = low + 0.13 * photons
        energy = np.clip(max(low + 5.0, apparent), low + 5.0, m["energy"][-1])
        ran = float(m["range_at"](energy)) / 1000.0
        probe = np.r_[point, axis, np.zeros(3), ran, 0.001, width]
        _, light, _ = render(model, probe, particle, 2)
        rate = model.response * light.sum() / (0.001 * ran)
        extent = photons / max(rate, 1e-12)
        for length in np.unique(np.clip([extent, cone[6]], 0.012, min(available, 0.995 * max_range))):
            birth = point + np.clip(-0.5 * length, water[0], water[1] - length) * axis
            r = min(max_range, max(1.001 * length, ran + 0.5 * length))
            x = np.r_[birth, axis, np.zeros(3), r, length / r, width]
            if hardware_margin(x) > 1e-5:
                yield particle, x
