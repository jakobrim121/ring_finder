"""Evaluate the calibrated circular path without temporary arrays per node."""

from contextlib import contextmanager
import math

from numba import njit
import numpy as np


@njit(cache=True)
def circular_path(x, nodes):
    """Return position, tangent and their derivatives in source coordinates."""
    norm = np.linalg.norm(x[3:6])
    axis = x[3:6] / norm
    plane = np.eye(3) - np.outer(axis, axis)
    direction_derivative = np.zeros((3, 12))
    direction_derivative[:, 3:6] = plane / norm
    turn = x[6:9]
    bend = plane @ turn
    power = bend @ bend
    bend_derivative = np.zeros((3, 12))
    bend_derivative[:, 3:6] = (-(axis @ turn) * np.eye(3) - np.outer(axis, turn)) @ direction_derivative[
        :, 3:6
    ]
    bend_derivative[:, 6:9] = plane
    power_derivative = 2.0 * bend @ bend_derivative
    length = 1000.0 * x[9] * x[10]
    length_derivative = np.zeros(12)
    length_derivative[9] = 1000.0 * x[10]
    length_derivative[10] = 1000.0 * x[9]
    position = np.empty((len(nodes), 3))
    tangent = np.empty_like(position)
    d_position = np.empty((len(nodes), 3, 12))
    d_tangent = np.empty_like(d_position)
    origin = 1000.0 * x[:3]

    for i, u in enumerate(nodes):
        if power < 1e-6:
            # The series and its derivatives stay finite at zero bend.
            c = 1.0 - power * u * u / 2.0 + power * power * u**4 / 24.0 - power**3 * u**6 / 720.0
            s = u - power * u**3 / 6.0 + power * power * u**5 / 120.0 - power**3 * u**7 / 5040.0
            h = u * u / 2.0 - power * u**4 / 24.0 + power * power * u**6 / 720.0 - power**3 * u**8 / 40320.0
            dc = -u * u / 2.0 + power * u**4 / 12.0 - power * power * u**6 / 240.0
            ds = -(u**3) / 6.0 + power * u**5 / 60.0 - power * power * u**7 / 1680.0
            dh = -(u**4) / 24.0 + power * u**6 / 360.0 - power * power * u**8 / 13440.0
        else:
            speed = math.sqrt(power)
            angle = speed * u
            c, s = math.cos(angle), math.sin(angle) / speed
            h = (1.0 - c) / power
            dc, ds, dh = -0.5 * u * s, (u * c - s) / (2.0 * power), (0.5 * u * s - h) / power
        for j in range(3):
            displacement = s * axis[j] + h * bend[j]
            position[i, j] = origin[j] + length * displacement
            tangent[i, j] = c * axis[j] + s * bend[j]
            d_displacement = ds * axis[j] + dh * bend[j]
            d_direction = dc * axis[j] + ds * bend[j]
            for k in range(12):
                da, db = direction_derivative[j, k], bend_derivative[j, k]
                dp = power_derivative[k]
                d_position[i, j, k] = (
                    length * (s * da + h * db + d_displacement * dp) + displacement * length_derivative[k]
                )
                d_tangent[i, j, k] = c * da + s * db + d_direction * dp
            d_position[i, j, j] += 1000.0
    return position, tangent, d_position, d_tangent


@contextmanager
def scratch_curve(enabled):
    """Switch the charge-integral calls within this worker's trial."""
    if not enabled:
        yield
        return
    import source

    previous = source.trajectory
    source.trajectory = circular_path
    try:
        yield
    finally:
        source.trajectory = previous
