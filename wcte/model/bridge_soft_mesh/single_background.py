"""Water-containment and bend derivatives shared by charged conversion models."""

import numpy as np
from material import material
from source import trajectory
from wctering.likelihood.geometry import NORMALS


def primary_inequality_jacobian(x, particle):
    """Differentiate water containment and the existing coherent-bend limits."""
    _, _, position_jac, _ = trajectory(x, np.linspace(0, 1, 33))
    a = x[3:6] / np.linalg.norm(x[3:6])
    v = x[6:9]
    projector = np.eye(3) - np.outer(a, a)
    bend = projector @ v
    angle = np.linalg.norm(bend)
    dangle = np.zeros(12)
    if angle > 1e-14:
        da = projector / np.linalg.norm(x[3:6])
        db = -((a @ v) * np.eye(3) + np.outer(a, v)) @ da
        dangle[3:6] = (bend / angle) @ db
        dangle[6:9] = (bend / angle) @ projector
    r, f = x[9:11]
    scale = 8 * 32**2
    dbound = r * f * dangle / scale
    dbound[9] += f * angle / scale
    dbound[10] += r * angle / scale
    water = -np.einsum("nc,vcp->vnp", NORMALS, position_jac / 1000).reshape(-1, 12)
    water -= dbound
    margin = material(particle)["margin_mm"] / 1000
    uncut = (1 - margin / r) / f
    core = np.clip(uncut, 0, 1)
    dcore = np.zeros(12)
    if 0 < uncut < 1:
        dcore[9] = margin / (r * r * f)
        dcore[10] = -core / f
    core_jac = -2 * core * angle**2 * dcore - 2 * core**2 * angle * dangle
    tail_jac = -2 * angle * dangle
    return np.vstack([water, core_jac, tail_jac])
