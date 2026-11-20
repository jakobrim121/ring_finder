"""Detector-only interaction proposals from already fitted charged paths.

Split either existing section at two fractional lengths. Every insertion location keeps its own search,
so an almost unchanged later section cannot crowd out a new earlier interaction. Keep the original
incoming mass, and try each allowed outgoing mass. A same-speed proposal and
a longer-range alternative initialize energy loss; conservative H/O recoil
and complete containment must hold before a proposal enters optimization.
The split location, masses' continuous energies, and all directions are then
fitted. No event truth, beam identity, energy label, or clock is consulted.
"""

import numpy as np
from chain_model import material
from source import trajectory
from geometry import TURN, MAX_TAIL_TURN


def canonical(model, x):
    x = np.array([np.clip(v, *bound) for v, bound in zip(x, model.bounds)])
    for j, particle in enumerate(model.particles):
        start = 3 + 9 * j
        axis = x[start : start + 3]
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            return None
        axis /= norm
        bend = x[start + 3 : start + 6]
        bend -= (axis @ bend) * axis
        ran, fraction = x[start + 6 : start + 8]
        core = np.clip((1 - material(particle)["margin_mm"] / (1000 * ran)) / fraction, 0.0, 1.0)
        limit = min(MAX_TAIL_TURN, TURN / max(core, 1e-12))
        bend *= min(1.0, 0.999 * limit / max(np.linalg.norm(bend), 1e-12))
    for start in model.axis_starts():
        norm = np.linalg.norm(x[start : start + 3])
        if norm < 1e-10:
            return None
        x[start : start + 3] /= norm
    return x


def split(path, fraction, incoming, outgoing, range_factor):
    """Cut a constant-curvature path without changing its spatial curve."""
    axis = path[3:6] / np.linalg.norm(path[3:6])
    bend = path[6:9] - (path[6:9] @ axis) * axis
    angle = np.linalg.norm(bend)
    at_split = trajectory(path, np.array([fraction]))[1][0]
    derivative = -angle * np.sin(angle * fraction) * axis + np.cos(angle * fraction) * bend
    first = path[3:].copy()
    first[3:6] = fraction * bend
    first[7] *= fraction
    second = np.r_[at_split, (1 - fraction) * derivative, path[9:]]
    remaining = path[9] * (1 - path[10] * fraction)
    source, target = material(incoming), material(outgoing)
    energy = float(source["energy_at"](1000 * remaining))
    same_speed_energy = energy * target["mass"] / source["mass"]
    same_speed_energy = np.clip(
        same_speed_energy, target["energy"][0] + 1.01, target["energy"][-1] * (1 - 1e-9)
    )
    length = path[9] * path[10] * (1 - fraction)
    ran = max(float(target["range_at"](same_speed_energy)) / 1000, 1.01 * length) * range_factor
    ran = min(ran, target["maximum_range_mm"] / 1000 * (1 - 1e-10))
    if length / ran > 1:
        return None
    second[6:8] = ran, length / ran
    return first, second
