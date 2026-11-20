"""Continuous electron tracks, with separate ring and low-energy light.

Unlike the earlier soft source, these tracks may start above the frozen ring
energy margin. Crossing that margin changes photon ownership, not the light
prediction. Both sides retain the same energy loss. The fitted high-energy core has no
second ensemble-scattering halo; the unresolved low-energy tail retains its
conditional scattering, matching the primary-source convention.
"""

import numpy as np
from electron import legs as original_legs, directions, turning
from mesh import boundaries as original_boundaries
from material import material
from source import raw
from source import trajectory
from prior import prior as original_prior
from geometry import inequalities, hardware_margin


def boundaries(log_range, arcs):
    """Resolve the core/tail transition without a singular zero-length arc.

    For three or more arcs, the first endpoint follows a smooth minimum of
    the usual fractional range and the ring-energy range. The remaining arcs
    use the original material monitor. This puts resolution near the physical
    transition while keeping all intervals positive and differentiable.
    The eighth-power smooth minimum is a fixed numerical mesh choice.
    """
    if arcs < 3:
        return original_boundaries(log_range, arcs)
    ran = 1000 * np.exp(log_range)
    margin = material("electron")["margin_mm"]
    ordinary = ran * (1 - 1 / arcs)
    ratio = (ordinary / margin) ** 8
    first = ordinary / (1 + ratio) ** (1 / 8)
    logarithmic_derivative = 1 / (1 + ratio)
    tail, derivative = original_boundaries(np.log(first / 1000), arcs - 1)
    return np.r_[ran, tail], np.r_[ran, derivative * logarithmic_derivative]


def legs(x):
    if len(x) == 13:
        return original_legs(x)
    axes, axis_jac = directions(x)
    size = len(x)
    ranges, dr = boundaries(x[-1], len(axes) - 1)
    point = x[:3].copy()
    point_jac = np.zeros((3, size))
    point_jac[:, :3] = np.eye(3)
    output = []
    for j, (a, b) in enumerate(zip(axes[:-1], axes[1:])):
        bend, da, db = turning(a, b)
        ran = ranges[j] / 1000
        length = (ranges[j] - ranges[j + 1]) / 1000
        fraction = length / ran
        y = np.r_[point, a, bend, ran, fraction, np.log(np.deg2rad(1.0))]
        jac = np.zeros((12, size))
        jac[:3] = point_jac
        jac[3:6, 3 + 3 * j : 6 + 3 * j] = axis_jac[j]
        jac[6:9, 3 + 3 * j : 6 + 3 * j] = da @ axis_jac[j]
        jac[6:9, 6 + 3 * j : 9 + 3 * j] = db @ axis_jac[j + 1]
        jac[9, -1] = dr[j] / 1000
        jac[10, -1] = (dr[j] - dr[j + 1] - fraction * dr[j]) / ranges[j]
        output.append((y, jac))
        points, _, derivative, _ = trajectory(y, np.array([1.0]))
        point = points[0] / 1000
        point_jac = derivative[0] @ jac / 1000
    return output


def prior(x):
    """The unchanged normalized angular law, evaluated on these mesh knots."""
    if len(x) == 13:
        return original_prior(x)
    axes, jac = directions(x)
    m = material("electron")
    ranges, dr = boundaries(x[-1], len(axes) - 1)
    from stable_vmf import angular, spline_difference

    scattering = m["scatter_at"]
    derivative = scattering.derivative()
    slopes = derivative(ranges)
    value = 0.0
    gradient = np.zeros_like(x)
    for j in range(1, len(axes)):
        q = spline_difference(scattering, ranges[j - 1], ranges[j])
        dq = (
            slopes[j - 1] * (dr[j] - dr[j - 1])
            + spline_difference(derivative, ranges[j - 1], ranges[j]) * dr[j]
        )
        weight, da, db, dvariance = angular(axes[j - 1], axes[j], q)
        value += weight
        gradient[3 + 3 * (j - 1) : 6 + 3 * (j - 1)] += jac[j - 1] @ da
        gradient[3 + 3 * j : 6 + 3 * j] += jac[j] @ db
        gradient[-1] += dvariance * dq
    return float(value), gradient


def containment(x):
    water = [inequalities(y, "electron")[:-2] for y, _ in legs(x)]
    axes, _ = directions(x)
    return np.r_[np.concatenate(water), np.einsum("ij,ij->i", axes[:-1], axes[1:]) + 0.99]


def containment_jacobian(x):
    from single_background import primary_inequality_jacobian

    water = [primary_inequality_jacobian(y, "electron")[:-2] @ jac for y, jac in legs(x)]
    axes, jac = directions(x)
    chart = np.zeros((len(axes) - 1, len(x)))
    for j in range(len(axes) - 1):
        chart[j, 3 + 3 * j : 6 + 3 * j] = jac[j] @ axes[j + 1]
        chart[j, 6 + 3 * j : 9 + 3 * j] = jac[j + 1] @ axes[j]
    return np.vstack([*water, chart])


def clearance(x):
    return min(hardware_margin(y) for y, _ in legs(x))


def source(positions, normals, x, order):
    """Return core/tail intensities and derivatives in full path coordinates."""
    value = np.zeros((2, len(positions)))
    derivative = np.zeros((2, len(positions), len(x)))
    for arc, transform in legs(x):
        parts, jac = raw(positions, normals, arc, order, True, "electron")
        length = 1000 * arc[9] * arc[10]
        dlength = 1000 * (arc[10] * transform[9] + arc[9] * transform[10])
        value += length * parts
        derivative += length * (jac @ transform) + parts[:, :, None] * dlength
    return value, derivative


def core_turn(x):
    """Bound total bending while the track is above the ring energy margin.

    A single declared core must remain coherent. The sum of arc angles is a
    conservative bound on its angular excursion; the low-energy tail has no
    such limit. This uses energy and geometry, never a truth track label.
    """
    margin = material("electron")["margin_mm"] / 1000
    total = 0.0
    gradient = np.zeros(len(x))
    for arc, transform in legs(x):
        ran, fraction = arc[9:11]
        if ran <= margin:
            continue
        uncut = (1 - margin / ran) / fraction
        end = min(1.0, uncut)
        dend = np.zeros(len(x))
        if 0 < uncut < 1:
            dend = margin / (ran * ran * fraction) * transform[9] - uncut / fraction * transform[10]
        bend = arc[6:9]
        angle = np.linalg.norm(bend)
        dangle = bend @ transform[6:9] / angle if angle > 1e-12 else np.zeros(len(x))
        total += end * angle
        gradient += end * dangle + angle * dend
    return np.deg2rad(15.0) ** 2 - total**2, -2 * total * gradient


def paths(z, pieces, pair):
    """Map shared conversion coordinates to individual electron paths.

    A pair has a common birth point and initial tangent. Each electron then
    has its own stopping range and later tangents. The shared tangent is the
    leading collinear conversion approximation, not exact pair kinematics.
    Separate energies are fitted; no beam-energy or simulation prior is used.
    """
    if not pair:
        return [(z, np.eye(len(z)))]
    later = 3 * (pieces - 1)
    output = []
    for particle in range(2):
        indices = np.r_[
            np.arange(6), np.arange(6 + particle * later, 6 + (particle + 1) * later), len(z) - 2 + particle
        ]
        output.append((z[indices], np.eye(len(z))[indices]))
    return output
