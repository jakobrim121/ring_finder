"""Uniform emission over the complete water intersection of an oriented line.

The anchor is a coordinate for the line, not a physical emission vertex. Its
location along the line is a redundant coordinate. All emitting intervals are
inside the prism and outside the union of the actual truncated mPMT domes.
Moving endpoint terms are retained analytically before normalization.
"""

import numpy as np
from wctering.likelihood.geometry import NORMALS as FACES, UPPER
from wctering.likelihood.water import CENTRES, NORMALS, RADIUS, CUT
from wctering.receiver.finite_source import raw


def endpoint(t, normal, axis, label):
    """Implicit boundary derivative for the line position c + t*a."""
    denominator = float(normal @ axis)
    if abs(denominator) < 1e-13:
        raise ArithmeticError("Nondifferentiable grazing boundary")
    derivative = -np.asarray(normal) / denominator
    return dict(t=float(t), dc=derivative, da=t * derivative, label=label)


def intervals(c, a, centres=CENTRES, axes=NORMALS, radius=RADIUS, cut=CUT):
    """Segment endpoints and their derivatives in metres, unit axis chart."""
    c = np.asarray(c, float)
    a = np.asarray(a, float)
    velocity = FACES @ a
    margin = UPPER - FACES @ c
    if np.any(margin[abs(velocity) < 1e-13] < -1e-7):
        raise ValueError("Line misses prism")
    lower = [
        endpoint(margin[i] / velocity[i], FACES[i], a, f"prism{i}") for i in np.flatnonzero(velocity < -1e-13)
    ]
    upper = [
        endpoint(margin[i] / velocity[i], FACES[i], a, f"prism{i}") for i in np.flatnonzero(velocity > 1e-13)
    ]
    lo = max(lower, key=lambda r: r["t"])
    hi = min(upper, key=lambda r: r["t"])
    if hi["t"] <= lo["t"]:
        raise ValueError("No positive water chord")
    blocked = []
    for j, (center, normal) in enumerate(zip(centres, axes)):
        d = c - center
        b = float(d @ a)
        disc = radius * radius - float(d @ d) + b * b
        if disc <= 0:
            continue
        half = np.sqrt(disc)
        left = -b - half
        right = -b + half
        x = endpoint(left, d + left * a, a, f"dome{j}:sphere")
        y = endpoint(right, d + right * a, a, f"dome{j}:sphere")
        vel = float(normal @ a)
        offset = float(normal @ d)
        if abs(vel) < 1e-13:
            if offset <= cut:
                continue
        else:
            plane = endpoint((cut - offset) / vel, normal, a, f"dome{j}:plane")
            if vel > 0:
                x = max((x, plane), key=lambda r: r["t"])
            else:
                y = min((y, plane), key=lambda r: r["t"])
        x = max((x, lo), key=lambda r: r["t"])
        y = min((y, hi), key=lambda r: r["t"])
        if y["t"] > x["t"]:
            blocked.append((x, y))
    blocked.sort(key=lambda pair: pair[0]["t"])
    # Overlapping dome exclusions must be removed once, as their union.
    union = []
    for x, y in blocked:
        if union and x["t"] <= union[-1][1]["t"]:
            union[-1] = (union[-1][0], max((union[-1][1], y), key=lambda r: r["t"]))
        else:
            union.append((x, y))
    water = []
    # Subtract that union from the prism chord, retaining each endpoint's origin.
    start = lo
    for x, y in union:
        if x["t"] > start["t"]:
            water.append((start, x))
        start = max((start, y), key=lambda r: r["t"])
    if hi["t"] > start["t"]:
        water.append((start, hi))
    if not water:
        raise ValueError("No active water on line")
    return water


def source(positions, normals, x, order=16, detail=False, domain=None, *, scalar=False):
    """Normalized light and derivatives for a line clipped to its water domain.

    x is [anchor_m[3], axis[3], opening_rad, log(width_rad)]. A segment's
    centre and length both move when the line moves. mc/ma and lc/la below
    carry those endpoint derivatives into the finite-source Jacobian.
    """
    x = np.asarray(x, float)
    norm = np.linalg.norm(x[3:6])
    a = x[3:6] / norm
    c = x[:3]
    projection = (np.eye(3) - np.outer(a, a)) / norm
    pieces = intervals(c, a, **(domain or {}))
    value = np.zeros(len(positions))
    jac = np.zeros((len(positions), 8))
    for lower, upper in pieces:
        length = upper["t"] - lower["t"]
        middle = 0.5 * (lower["t"] + upper["t"])
        mc = 0.5 * (lower["dc"] + upper["dc"])
        ma = 0.5 * (lower["da"] + upper["da"])
        lc = upper["dc"] - lower["dc"]
        la = upper["da"] - lower["da"]
        p = np.concatenate((c + middle * a, a, [length], x[6:8]))
        v, d, count = raw(positions, normals, p, order, scalar=scalar)
        value += length * v
        # raw() is the segment-average response: length*v is its integral.
        motion = length * (d[:, :3] @ a)
        extent = v + length * d[:, 6]
        jac[:, :3] += length * d[:, :3] + motion[:, None] * mc + extent[:, None] * lc
        jac[:, 3:6] += (
            length * (middle * d[:, :3] + d[:, 3:6]) + motion[:, None] * ma + extent[:, None] * la
        ) @ projection
        jac[:, 6:8] += length * d[:, 7:9]
    total = value.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("No finite optical support")
    shape = value / total
    derivative = (jac - shape[:, None] * jac.sum(0)) / total
    if detail:
        return shape, derivative, pieces, value, jac
    return shape, derivative
