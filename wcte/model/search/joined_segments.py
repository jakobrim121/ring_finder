"""Propose a missed charged section from a timed finite-track footprint."""

import numpy as np

from chain_sources import paths
from source import trajectory


def separated_directions(model, state):
    """Leave uncertainty around the fixed 15-degree lineage boundary."""
    incoming, outgoing = [p for p, _ in paths(state["parameters"][: model.ng])]
    tangent = trajectory(incoming, np.array([1.0]))[1][0]
    width = np.hypot(np.exp(incoming[11]), np.exp(outgoing[11]))
    angle = np.arccos(np.clip(tangent @ outgoing[3:6], -1.0, 1.0))
    return float(angle), bool(angle > np.deg2rad(15.0) + 2.0 * width)


def arc(path, left, right):
    """Reparameterize the same physical trajectory between two arclength fractions."""
    length = path[9] * path[10]
    span = (right - left) * length
    remaining = path[9] - left * length
    if not 0.008 <= span <= remaining:
        return None
    points, tangents, _, _ = trajectory(path, np.array([left, right]))
    axis = tangents[0]
    change = tangents[1] - (axis @ tangents[1]) * axis
    sine = np.linalg.norm(change)
    turn = change * (np.arctan2(sine, axis @ tangents[1]) / max(sine, 1e-12))
    return np.r_[points[0] / 1000.0, axis, turn, remaining, span / remaining, path[11]]
