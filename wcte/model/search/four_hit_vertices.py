"""Seed compact Cherenkov sources from independent first-pulse receivers."""

import numpy as np

from wctering.likelihood.geometry import NORMALS, UPPER


def intersections(positions, length_times, groups):
    """Intersect four spherical wavefronts with an unknown emission time."""
    x = positions[groups]
    time = length_times[groups]
    # Subtract the squared range of the first PMT from the other three.
    # The position is then affine in its distance to that first PMT.
    delta = x[:, 1:] - x[:, :1]
    dt = time[:, 1:] - time[:, :1]
    valid = abs(np.linalg.det(delta)) > 0.012
    delta, dt, x, time = delta[valid], dt[valid], x[valid], time[valid]
    if not len(x):
        return np.empty((0, 3)), np.empty(0)
    square = np.einsum("ijk,ijk->ij", delta, delta)
    rhs = np.stack([0.5 * (square - dt * dt), -dt], axis=-1)
    origin, slope = np.linalg.solve(delta, rhs).transpose(2, 0, 1)
    a = np.einsum("ij,ij->i", slope, slope) - 1.0
    b = 2.0 * np.einsum("ij,ij->i", origin, slope)
    c = np.einsum("ij,ij->i", origin, origin)
    discriminant = b * b - 4.0 * a * c
    # Use the product of the roots for the second solution. This avoids
    # subtracting almost equal numbers for a nearly linear equation.
    q = -0.5 * (b + np.copysign(np.sqrt(np.maximum(discriminant, 0.0)), b))
    far = np.divide(q, a, out=np.full_like(a, -1.0), where=abs(a) > 1e-12)
    near = np.divide(c, q, out=np.full_like(q, -1.0), where=abs(q) > 1e-12)
    roots = np.column_stack([far, near])
    valid = (discriminant[:, None] >= 0.0) & (roots > np.maximum(0.0, -dt.min(1))[:, None]) & (roots < 4.5)
    hit, side = np.nonzero(valid)
    distance = roots[hit, side]
    points = x[hit, 0] + origin[hit] + distance[:, None] * slope[hit]
    keep = np.min(UPPER - points @ NORMALS.T, axis=1) > 0.006
    return points[keep], (time[hit, 0] - distance)[keep]
