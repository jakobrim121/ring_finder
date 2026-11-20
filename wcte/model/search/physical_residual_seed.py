"""Search for extra optical light around an already fitted charged path."""

import numpy as np

from material import material
from source import trajectory
from wctering.likelihood.charge import MAX_OPENING_RAD


def optical_geometry(path, particle, *, all_emission=False):
    """Use the above-margin emission interval of a fitted charged path."""
    water = material(particle)
    ran = 1000.0 * path[9]
    span = ran * path[10]
    core = span if all_emission else min(span, ran - water["margin_mm"])
    if core < 5.0:
        return None
    sample = np.array([0.0, 0.5, 1.0]) * core / span
    points, directions, _, _ = trajectory(path, sample)
    energy = float(water["energy_at"](ran - 0.5 * core))
    beta = np.sqrt(1.0 - (1.0 + energy / water["mass"]) ** -2)
    opening = np.arccos(np.clip(1.0 / (water["index"] * beta), -1.0, 1.0))
    turn = np.arccos(np.clip(directions[0] @ directions[-1], -1.0, 1.0))
    # A finite straight start represents the average tangent of the arc.
    # The checked physical likelihood still uses the true curved path.
    width = np.hypot(np.exp(path[11]), turn / np.sqrt(12.0))
    cone = np.r_[
        points[1] / 1000.0,
        directions[1],
        core / 1000.0,
        np.clip(opening, 0.12, MAX_OPENING_RAD),
        np.log(np.clip(width, np.deg2rad(1.2), np.deg2rad(16.0))),
    ]
    return cone, points / 1000.0
