"""Two physical curved tracks meeting at one interaction vertex.

The first 15 coordinates match the straight connected model. Six additional
coordinates give the transverse bend of each track. Incoming momentum at the
interaction follows its END tangent, which matters for the recoil constraint.
Both tracks use exactly the one-track source and containment certificate.
"""

import numpy as np
from source import raw, trajectory
from geometry import equalities, inequalities, hardware_margin as path_margin
from single_background import primary_inequality_jacobian
from conservation import margins


def local_path(x, leg):
    """Return a 12-coordinate path and its derivative in shared coordinates."""
    axis = 3 + 3 * leg
    ran = 9 + 2 * leg
    bend = 15 + 3 * leg
    y = np.r_[x[:3], x[axis : axis + 3], x[bend : bend + 3], x[ran : ran + 2], x[13 + leg]]
    transform = np.zeros((12, 21))
    transform[:3, :3] = np.eye(3)
    transform[3:6, axis : axis + 3] = np.eye(3)
    transform[6:9, bend : bend + 3] = np.eye(3)
    transform[9, ran] = 1.0
    transform[10, ran + 1] = 1.0
    transform[11, 13 + leg] = 1.0
    if leg == 0:
        end, _, position_jac, _ = trajectory(y, np.array([1.0]))
        # Translate the entire arc so its endpoint is the shared vertex.
        end_jac = position_jac[0] @ transform / 1000
        y[:3] -= end[0] / 1000 - x[:3]
        transform[:3] -= end_jac
        transform[:3, :3] += np.eye(3)
    return y, transform


def interaction_coordinates(x):
    """Legacy energy coordinates with the actual incoming endpoint tangent."""
    y, transform = local_path(x, 0)
    _, axes, _, tangent_jac = trajectory(y, np.array([1.0]))
    z = x[:15].copy()
    z[3:6] = axes[0]
    jac = np.zeros((15, 21))
    jac[:, :15] = np.eye(15)
    jac[3:6] = tangent_jac[0] @ transform
    return z, jac


def core_axes(x):
    """Directions at the common interaction, for the existing merge rule."""
    z, _ = interaction_coordinates(x)
    return z[3:6], z[6:9]


def source(positions, normals, x, particles, order, bridge):
    values = np.zeros((2, 2, len(positions)))
    derivative = np.zeros((len(positions), 21))
    for leg, particle in enumerate(particles):
        y, transform = local_path(x, leg)
        parts, jacs = raw(positions, normals, y, order, bridge, particle)
        length = 1000 * y[9] * y[10]
        values[leg] = length * parts
        dlength = 1000 * (y[10] * transform[9] + y[9] * transform[10])
        derivative += length * (jacs.sum(0) @ transform) + parts.sum(0)[:, None] * dlength
    return values, derivative


def equality(x):
    return np.concatenate([equalities(local_path(x, leg)[0]) for leg in (0, 1)])


def equality_jacobian(x):
    rows = []
    for leg in (0, 1):
        y, transform = local_path(x, leg)
        jac = np.zeros((2, 12))
        jac[0, 3:6] = 2 * y[3:6]
        jac[1, 3:6] = y[6:9]
        jac[1, 6:9] = y[3:6]
        rows.append(jac @ transform)
    return np.vstack(rows)


def inequality(x, particles):
    result = [inequalities(local_path(x, leg)[0], particle) for leg, particle in enumerate(particles)]
    z, _ = interaction_coordinates(x)
    return np.r_[np.concatenate(result), margins(z, particles)[0].max()]


def inequality_jacobian(x, particles):
    rows = []
    for leg, particle in enumerate(particles):
        y, transform = local_path(x, leg)
        rows.append(primary_inequality_jacobian(y, particle) @ transform)
    z, transform = interaction_coordinates(x)
    values, jac = margins(z, particles)
    rows.append((jac[np.argmax(values)] @ transform)[None])
    return np.vstack(rows)


def hardware_margin(x):
    return min(path_margin(local_path(x, leg)[0]) for leg in (0, 1))
