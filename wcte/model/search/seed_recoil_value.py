"""Evaluate charge-start recoil bounds with scalar work only."""

from functools import lru_cache

import numpy as np

from conservation import ELECTRON, ISOBARS, TARGETS, U
from material import material
from source import trajectory


@lru_cache(maxsize=3)
def masses(baryon_change):
    """Store the allowed target and residual rest masses for each charge change."""
    targets = tuple(U * ISOBARS[a][0] for a in TARGETS)
    residual = tuple(
        max(U * ISOBARS[a + baryon_change][0] - ISOBARS[a + baryon_change][1] * ELECTRON, 0.0) ** 2
        for a in TARGETS
    )
    return targets, residual


def prepare(coordinates, particles, section):
    """Use the same per-section kinematics as the constrained optimizer.

    The recoil condition is invariant under translating the entire arc.
    Use the original endpoint tangent routine on the local incoming chart
    so its normalization arithmetic is the same at the final Brent step.
    """
    j = 3 + 9 * section
    incoming = np.r_[coordinates[:3], coordinates[j : j + 9]]
    a = trajectory(incoming, np.ones(1))[1][0]
    b = coordinates[j + 9 : j + 12].copy()
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine = float(a @ b)
    water_a, water_b = map(material, particles[section : section + 2])
    ma, mb = water_a["mass"], water_b["mass"]
    eb = mb + float(water_b["energy_at"](1000.0 * coordinates[j + 15]))
    pb = np.sqrt(max(eb * eb - mb * mb, 1e-30))
    ea_at = water_a["energy_at"]
    target, residual2 = masses(int(particles[section] == "proton") - int(particles[section + 1] == "proton"))

    def constraint(ran, fraction):
        ea = ma + float(ea_at(1000.0 * ran * (1.0 - fraction)))
        pa = np.sqrt(max(ea * ea - ma * ma, 1e-30))
        missing = max(pa * pa + pb * pb - 2.0 * pa * pb * cosine, 0.0)
        return (
            max((ea + t - eb - np.sqrt(m + missing) + 0.02) / 1000.0 for t, m in zip(target, residual2))
            - 1e-7
        )

    return constraint
