"""Initialize connected charged paths from detector-fitted optical cones.

The cone fit need not have identified the correct interaction order or mass.
We therefore try every ordering, fit their approximate emission centres to
a connected polyline, and try each incoming/outgoing mass pair. The latter
is a compact hypothesis set: after the first interaction, the same outgoing
mass may scatter repeatedly. All continuous parameters are then free.

This is an initialization/search experiment, not a new scoring rule. Truth,
beam labels, event numbers and simulation clocks never enter this module.
"""

import numpy as np
from scipy.optimize import brentq
from chain_sources import paths, interactions
from material import material


def geometries(old, bank, count):
    """Keep distinct optical geometries, regardless of their selected count."""
    candidates = sorted(
        (r for r in old["fits"] if r["k"] == count and r.get("qualified") and r.get("geometry")),
        key=lambda r: r.get("score", -np.inf),
        reverse=True,
    )
    result = []
    for row in candidates:
        g = row["geometry"]
        descriptor = np.r_[np.asarray(g["axes"]).ravel(), np.asarray(g["vertices"]).ravel() / 1000]
        if any(np.linalg.norm(descriptor - d) < 0.05 for _, d in result):
            continue
        result.append((g, descriptor))
        if len(result) == 3:
            break
    if str(count) in bank:
        result.append((bank[str(count)], None))
    return [g for g, _ in result]


def physical_start(vertex, axes, lengths, openings, particles, scale, normalization):
    """Map geometry to physical ranges and satisfy existing recoil bounds.

    An opening angle suggests speed, not particle identity. When that speed
    cannot supply the measured path length or its outgoing particle, raise
    the incoming energy only far enough to provide a feasible starting point.
    No constraint is relaxed, and optimization subsequently refits all ranges.
    """
    sections = []
    for axis, length, opening, particle in zip(axes, lengths, openings, particles):
        m = material(particle)
        maximum = np.rad2deg(np.arccos(1 / m["index"]))
        angle = np.deg2rad(np.clip(opening, 1.0, maximum - 0.5))
        beta = 1 / (m["index"] * np.cos(angle))
        energy = m["mass"] * (1 / np.sqrt(1 - beta * beta) - 1)
        energy = np.clip(scale * energy, m["energy"][0] + 2.0, m["energy"][-1] * (1 - 1e-8))
        ran = max(float(m["range_at"](energy)) / 1000, length / 0.9)
        if ran >= m["maximum_range_mm"] / 1000:
            return None
        sections.append(np.r_[axis, np.zeros(3), ran, length / ran, np.log(np.deg2rad(1.0))])
    primary = np.r_[vertex, *sections]
    for j in reversed(range(len(particles) - 1)):
        rindex = 9 + 9 * j

        def surplus(ran):
            primary[rindex : rindex + 2] = ran, lengths[j] / ran
            return interactions(paths(primary), particles)[0][j]

        initial = primary[rindex]
        if surplus(initial) >= 1e-7:
            continue
        maximum = material(particles[j])["maximum_range_mm"] / 1000 * (1 - 1e-8)
        if surplus(maximum) < 1e-7:
            return None
        root = brentq(lambda r: surplus(r) - 1e-7, initial, maximum, xtol=1e-10)
        surplus(root)
    return np.r_[primary, normalization]
