"""Check recoil energy at a fixed geometry without forming derivatives."""

from functools import lru_cache

import numpy as np

from chain_sources import paths
from conservation import ELECTRON, ISOBARS, TARGETS, U
from material import material
from source import trajectory


@lru_cache(maxsize=3)
def masses(baryon_change):
    """Enumerate the same target and residual masses as the full constraint."""
    target = np.array([U * ISOBARS[a][0] for a in TARGETS])
    residual = np.array(
        [
            max(U * ISOBARS[a + baryon_change][0] - ISOBARS[a + baryon_change][1] * ELECTRON, 0.0)
            for a in TARGETS
        ]
    )
    return target, residual**2


def prepare(coordinates, particles, section):
    """Create a scalar recoil test for changing one incoming range."""
    mapped = paths(coordinates)
    incoming, outgoing = [p for p, _ in mapped[section : section + 2]]
    a = trajectory(incoming, np.ones(1))[1][0]
    b = outgoing[3:6]
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(a @ b)
    material_a, material_b = map(material, particles[section : section + 2])
    ma, mb = material_a["mass"], material_b["mass"]
    ea_at = material_a["energy_at"]
    eb = mb + float(material_b["energy_at"](1000.0 * outgoing[9]))
    pb = np.sqrt(max(eb * eb - mb * mb, 1e-30))
    target, residual2 = masses(int(particles[section] == "proton") - int(particles[section + 1] == "proton"))

    def constraint(ran, fraction):
        ea = ma + float(ea_at(1000.0 * ran * (1.0 - fraction)))
        pa = np.sqrt(max(ea * ea - ma * ma, 1e-30))
        missing = max(pa * pa + pb * pb - 2.0 * pa * pb * cosine, 0.0)
        return float(((ea + target - eb - np.sqrt(residual2 + missing) + 0.02) / 1000.0).max()) - 1e-7

    return constraint
