"""Evaluate the existing sixteen-chord dome certificate in one native loop."""

import math

import numpy as np
from numba import njit

from source import trajectory
from wctering.likelihood.water_boundary import CENTRES, CUT, NORMALS, RADIUS


@njit(cache=True, inline="always")
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def closest_chord(points, centres, normals, radius, cut):
    """Find the same active dome and chord as the vectorized certificate.

    The minimum radial and planar margins bound the combined margin from
    below. Most domes are far away, so their full two-branch minimization
    cannot affect the certificate. Keep original chord and dome tie order.
    """
    best = math.inf
    best_chord = 0
    best_dome = 0
    for j in range(len(points) - 1):
        centre = (points[j] + points[j + 1]) / 2
        delta = points[j + 1] - points[j]
        length = math.sqrt(dot(delta, delta))
        if length <= 1e-14:
            return -1, -1
        axis = delta / length
        half = 0.5 * length
        for i in range(len(centres)):
            d = centre - centres[i]
            normal = normals[i]
            b = dot(d, axis)
            e = dot(normal, axis)
            d2 = dot(d, d)
            plane = cut - dot(d, normal)
            closest = min(max(-b, -half), half)
            radial_min = math.sqrt(max(0.0, d2 + closest * (2 * b + closest))) - radius
            if max(radial_min, plane - half * abs(e)) > best + 1e-10:
                continue

            # The convex maximum of two functions can minimize at a segment
            # endpoint, the radial minimum, or a root where the branches meet.
            h = radius + plane
            aa, bb, cc = max(0.0, 1 - e * e), 2 * (b + h * e), d2 - h * h
            disc = bb * bb - 4 * aa * cc
            first, second = math.nan, math.nan
            if disc >= 0.0:
                if aa > 1e-12:
                    root = math.sqrt(disc)
                    q = -0.5 * (bb + (-root if bb < 0.0 else root))
                    first = q / aa
                    if abs(q) > 1e-30:
                        second = cc / q
                elif abs(bb) > 1e-14:
                    first = -cc / bb
            for k, s in enumerate((-half, half, closest, first, second)):
                if not (-half <= s <= half) or (k >= 3 and h - e * s < 0.0):
                    continue
                v = d + s * axis
                value = max(math.sqrt(dot(v, v)) - radius, cut - dot(v, normal))
                if value < best:
                    best, best_chord, best_dome = value, j, i
    return best_chord, best_dome


def certificate(x, segments=16):
    """Use the original analytic subgradient for the active chord and dome."""
    from domes import certificate as original
    from wctering.likelihood.water_boundary import clearance

    fractions = np.linspace(0.0, 1.0, segments + 1)
    points, _, derivatives, _ = trajectory(x, fractions)
    points /= 1000
    derivatives /= 1000
    j, i = closest_chord(points, CENTRES, NORMALS, RADIUS, CUT)
    if j < 0:
        return original(x, segments)
    centre = (points[j] + points[j + 1]) / 2
    centre_jac = (derivatives[j] + derivatives[j + 1]) / 2
    delta = points[j + 1] - points[j]
    delta_jac = derivatives[j + 1] - derivatives[j]
    length = np.linalg.norm(delta)
    axis = delta / length
    projection = np.eye(3) - np.outer(axis, axis)
    value, dc, da, dl = clearance(
        centre, axis, length, centres=CENTRES[i : i + 1], normals=NORMALS[i : i + 1]
    )
    gradient = dc[0] @ centre_jac + da[0] @ (projection @ delta_jac / length) + dl[0] * (axis @ delta_jac)

    # The sag bound and its gradient depend on the arc, not the active chord.
    axis = x[3:6] / np.linalg.norm(x[3:6])
    projection = np.eye(3) - np.outer(axis, axis)
    bend = projection @ x[6:9]
    angle = np.linalg.norm(bend)
    bend_jac = np.zeros((3, 12))
    bend_jac[:, 3:6] = (
        (-(axis @ x[6:9]) * np.eye(3) - np.outer(axis, x[6:9])) @ projection / np.linalg.norm(x[3:6])
    )
    bend_jac[:, 6:9] = projection
    angle_jac = bend @ bend_jac / angle if angle > 1e-12 else np.zeros(12)
    length = x[9] * x[10]
    length_jac = np.zeros(12)
    length_jac[9:11] = x[10], x[9]
    scale = 1 / (8 * segments**2)
    gradient -= scale * (length * angle_jac + angle * length_jac)
    return value - scale * length * angle - 2e-5, gradient[None, :]
