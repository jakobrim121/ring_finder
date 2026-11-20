"""Serial charged trajectories with exact endpoint and derivative sharing.

Coordinates are one starting vertex followed by nine numbers per section:
axis (3), transverse bend (3), visible range, traversed fraction, log width.
An interaction changes the mass hypothesis and/or direction. No simulation
ancestry is used: these are candidate physical trajectories for charge fits.
"""

import numpy as np
from source import raw, trajectory
from conservation import margins


def paths(x):
    """Map each local 12-coordinate arc into the common chain coordinates."""
    vertex = x[:3].copy()
    derivative = np.zeros((3, len(x)))
    derivative[:, :3] = np.eye(3)
    result = []
    for start in range(3, len(x), 9):
        path = np.r_[vertex, x[start : start + 9]]
        transform = np.zeros((12, len(x)))
        transform[:3] = derivative
        transform[3:, start : start + 9] = np.eye(9)
        result.append((path, transform))
        position, _, jac, _ = trajectory(path, np.array([1.0]))
        vertex = position[0] / 1000
        derivative = jac[0] @ transform / 1000
    return result


def interactions(mapped, particles):
    """Conservative H/O recoil constraints at every connected endpoint."""
    values, gradients = [], []
    for leg, ((incoming, da), (outgoing, db)) in enumerate(zip(mapped, mapped[1:])):
        _, axis, _, axis_jac = trajectory(incoming, np.array([1.0]))
        z = np.r_[
            outgoing[:3], axis[0], outgoing[3:6], incoming[9:11], outgoing[9:11], incoming[11], outgoing[11]
        ]
        transform = np.vstack([db[:3], axis_jac[0] @ da, db[3:6], da[9:11], db[9:11], da[11], db[11]])
        surplus, jac = margins(z, particles[leg : leg + 2])
        # Isotopic targets remain alternatives, exactly as in the old model.
        best = int(np.argmax(surplus))
        values.append(surplus[best])
        gradients.append(jac[best] @ transform)
    return np.array(values), np.array(gradients)


def light(positions, normals, x, particles, order):
    """Reuse the qualified per-section source and its analytic derivatives."""
    mapped = paths(x)
    pieces = np.zeros((len(mapped), 2, len(positions)))
    derivative = np.zeros((len(positions), len(x)))
    for j, ((path, transform), particle) in enumerate(zip(mapped, particles)):
        value, jac = raw(positions, normals, path, order, False, particle)
        length = 1000 * path[9] * path[10]
        dlength = 1000 * (path[10] * transform[9] + path[9] * transform[10])
        pieces[j] = length * value
        derivative += length * jac.sum(0) @ transform + value.sum(0)[:, None] * dlength
    return pieces, derivative
