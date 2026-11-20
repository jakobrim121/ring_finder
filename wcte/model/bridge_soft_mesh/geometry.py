"""Water containment and bend limits for finite charged paths."""

import numpy as np
from material import material
from source import trajectory
from wctering.likelihood.geometry import NORMALS, UPPER
from wctering.likelihood.water_boundary import clearance

RESPONSE = 145166.41320117615
TURN = np.deg2rad(15.0)
MAX_TAIL_TURN = np.deg2rad(120.0)


def core_fraction_of_path(x, particle="electron"):
    return float(np.clip((1 - material(particle)["margin_mm"] / (1000 * x[9])) / x[10], 0, 1))


def equalities(x):
    return np.array([x[3:6] @ x[3:6] - 1.0, x[3:6] @ x[6:9]])


def inequalities(x, particle="electron"):
    points = trajectory(x, np.linspace(0, 1, 33))[0] / 1000
    a = x[3:6] / np.linalg.norm(x[3:6])
    bend = x[6:9] - (a @ x[6:9]) * a
    angle = np.linalg.norm(bend)
    # Every arc lies within this distance of its chord. The subtraction is a
    # conservative whole-path prism certificate, not a point sampling claim.
    bound = x[9] * x[10] * angle / (8 * 32**2)
    # Below-margin light may turn strongly without becoming another counted
    # ring. Only the above-margin part must admit the one-core interpretation.
    core = core_fraction_of_path(x, particle)
    return np.r_[
        (UPPER - points @ NORMALS.T - bound).ravel(),
        TURN * TURN - core * core * (bend @ bend),
        MAX_TAIL_TURN**2 - bend @ bend,
    ]


def hardware_margin(x):
    a = x[3:6] / np.linalg.norm(x[3:6])
    angle = np.linalg.norm(x[6:9] - (a @ x[6:9]) * a)
    length = x[9] * x[10]
    pending = [(0.0, 1.0, 0)]
    margins = []
    while pending:
        low, high, depth = pending.pop()
        ends = trajectory(x, np.array([low, high]))[0] / 1000
        chord = ends[1] - ends[0]
        span = np.linalg.norm(chord)
        axis = chord / span if span > 0 else a
        lower = float(clearance(ends.mean(0), axis, span)[0].min()) - length * angle * (high - low) ** 2 / 8
        if lower >= 1e-5:
            margins.append(lower)
            continue
        if depth >= 14:
            return lower
        middle = 0.5 * (low + high)
        point = trajectory(x, np.array([middle]))[0][0] / 1000
        if float(clearance(point, a, 0.0)[0].min()) < 1e-5:
            return -1.0
        pending.extend([(low, middle, depth + 1), (middle, high, depth + 1)])
    return min(margins)
