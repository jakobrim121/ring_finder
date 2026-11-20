"""Differentiate a whole-arc certificate for staying outside PMT domes.

Each chord uses the existing exact truncated-sphere clearance. Subtracting
the circular sag bound certifies the entire arc, including points between
chords. The most restrictive chord/dome gives an envelope subgradient.
Unlike the old final-only check, this can steer an optimizer away from glass.
"""

import numpy as np
from source import trajectory
from wctering.likelihood.water_boundary import clearance


def certificate(x, segments=16):
    axis = x[3:6] / np.linalg.norm(x[3:6])
    projector = np.eye(3) - np.outer(axis, axis)
    v = x[6:9]
    bend = projector @ v
    angle = np.linalg.norm(bend)
    jb = np.zeros((3, 12))
    jb[:, 3:6] = (-(axis @ v) * np.eye(3) - np.outer(axis, v)) @ projector / np.linalg.norm(x[3:6])
    jb[:, 6:9] = projector
    da = bend @ jb / angle if angle > 1e-12 else np.zeros(12)
    length = x[9] * x[10]
    dl = np.zeros(12)
    dl[9] = x[10]
    dl[10] = x[9]
    bound = length * angle / (8 * segments**2)
    dbound = (angle * dl + length * da) / (8 * segments**2)
    points, _, derivatives, _ = trajectory(x, np.linspace(0.0, 1.0, segments + 1))
    points /= 1000
    derivatives /= 1000
    best, gradient = np.inf, np.zeros(12)
    for j in range(segments):
        center = (points[j] + points[j + 1]) / 2
        center_jac = (derivatives[j] + derivatives[j + 1]) / 2
        delta = points[j + 1] - points[j]
        delta_jac = derivatives[j + 1] - derivatives[j]
        span = np.linalg.norm(delta)
        if span > 1e-14:
            direction = delta / span
            span_jac = direction @ delta_jac
            direction_jac = (np.eye(3) - np.outer(direction, direction)) @ delta_jac / span
        else:
            direction, span_jac, direction_jac = axis, np.zeros(12), np.zeros((3, 12))
        margins, dc, dd, ds = clearance(center, direction, span)
        closest = int(np.argmin(margins))
        value = margins[closest] - bound
        if value < best:
            best = value
            gradient = (
                dc[closest] @ center_jac + dd[closest] @ direction_jac + ds[closest] * span_jac - dbound
            )
    # A 20-micrometre clearance leaves room for the 1-micrometre numerical
    # feasibility tolerance and the unchanged final >10-micrometre guard.
    return np.array([best - 2e-5]), gradient[None, :]
