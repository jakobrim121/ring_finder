"""A resolved curved electron path with unresolved scattering between nodes.

Fitted unit directions are continuous path tangents at material-based knots.
Between them the mean follows a circular arc. A conditional angular diffusion
bridge accounts for unresolved scattering, vanishing at both fitted tangents.
This is a local Gaussian transport approximation, not exact spherical or
lateral Brownian transport. All cells share one continuous stopping history.
"""

import numpy as np
from source import trajectory
from mesh import boundaries


def directions(x):
    axes = x[3:-1].reshape(-1, 3)
    norms = np.linalg.norm(axes, axis=1)
    unit = axes / norms[:, None]
    jac = np.array([(np.eye(3) - np.outer(a, a)) / n for a, n in zip(unit, norms)])
    return unit, jac


def turning(a, b):
    """Sphere logarithm and its unconstrained a/b derivatives.

    Multiplication by the unit-vector Jacobians below projects away radial
    directions. A series gives the coincident-tangent limit without 0/0.
    """
    cosine = np.clip(a @ b, -1 + 1e-12, 1.0)
    delta = 1 - cosine
    if delta < 1e-5:
        h = 1 + delta / 3 + 2 * delta**2 / 15 + 2 * delta**3 / 35
        dh = -1 / 3 - 4 * delta / 15 - 6 * delta**2 / 35
    else:
        sine = np.sqrt(1 - cosine * cosine)
        angle = np.arccos(cosine)
        h = angle / sine
        dh = (angle * cosine - sine) / sine**3
    transverse = b - cosine * a
    da = dh * np.outer(transverse, b) - h * (np.outer(a, b) + cosine * np.eye(3))
    db = dh * np.outer(transverse, a) + h * (np.eye(3) - np.outer(a, a))
    return h * transverse, da, db


def legs(x):
    """Each arc's source coordinates and exact derivative in the full path."""
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
