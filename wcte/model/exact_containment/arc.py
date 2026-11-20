"""Exact prism containment of a circular track, with analytic derivatives.

The projection on each prism normal is sinusoidal along the arc. Its maximum
occurs at an endpoint or a stationary angle. The envelope theorem lets us use
the ordinary fixed-arclength trajectory derivative at that maximum; the
derivative of the maximizing location cancels. At ties the envelope is
nondifferentiable, and either active maximum supplies a valid subgradient.
"""

import math
import numpy as np
from numba import njit
from source import trajectory
from material import material
from geometry import TURN, MAX_TAIL_TURN
from wctering.likelihood.geometry import NORMALS, UPPER


@njit(cache=True)
def maximizing_locations(x, normals):
    axis = x[3:6] / np.linalg.norm(x[3:6])
    bend = x[6:9] - (axis @ x[6:9]) * axis
    power = bend @ bend
    angle = math.sqrt(power)
    result = np.empty(len(normals))
    for j in range(len(normals)):
        a = normals[j] @ axis
        b = normals[j] @ bend
        best, location = 0.0, 0.0
        candidates = np.zeros(7)
        candidates[1] = 1.0
        count = 2
        if angle > 1e-14 and (a != 0.0 or b != 0.0):
            root = math.atan2(-a, b / angle)
            first = math.ceil(-root / math.pi)
            for turn in range(first, first + 5):
                u = (root + turn * math.pi) / angle
                if 0.0 < u < 1.0:
                    candidates[count] = u
                    count += 1
        for k in range(count):
            u = candidates[k]
            if power < 1e-6:
                sine = u - power * u**3 / 6 + power * power * u**5 / 120 - power**3 * u**7 / 5040
                cosine = u * u / 2 - power * u**4 / 24 + power * power * u**6 / 720 - power**3 * u**8 / 40320
            else:
                sine = math.sin(angle * u) / angle
                cosine = (1 - math.cos(angle * u)) / power
            projection = sine * a + cosine * b
            if projection > best:
                best, location = projection, u
        result[j] = location
    return result


def prism(x):
    locations = maximizing_locations(x, NORMALS)
    points, _, derivative, _ = trajectory(x, locations)
    values = UPPER - np.einsum("ij,ij->i", NORMALS, points / 1000)
    jac = -np.einsum("ij,ijp->ip", NORMALS, derivative / 1000)
    return values, jac


def bend_limits(x, particle):
    axis = x[3:6] / np.linalg.norm(x[3:6])
    projector = np.eye(3) - np.outer(axis, axis)
    v = x[6:9]
    bend = projector @ v
    ja = projector / np.linalg.norm(x[3:6])
    jb = np.zeros((3, 12))
    jb[:, 3:6] = (-(axis @ v) * np.eye(3) - np.outer(axis, v)) @ ja
    jb[:, 6:9] = projector
    power = bend @ bend
    dpower = 2 * bend @ jb
    ran, fraction = x[9:11]
    margin = material(particle)["margin_mm"] / 1000
    uncut = (1 - margin / ran) / fraction
    core = np.clip(uncut, 0.0, 1.0)
    dcore = np.zeros(12)
    if 0 < uncut < 1:
        dcore[9] = margin / (ran * ran * fraction)
        dcore[10] = -core / fraction
    return (
        np.array([TURN * TURN - core * core * power, MAX_TAIL_TURN**2 - power]),
        np.vstack([-core * core * dpower - 2 * core * power * dcore, -dpower]),
    )


def primary(x, particle):
    water, water_jac = prism(x)
    bend, bend_jac = bend_limits(x, particle)
    return np.r_[water, bend], np.vstack([water_jac, bend_jac])
