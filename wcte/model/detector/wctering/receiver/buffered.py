"""Allocation-only candidates. No fastmath, approximations or changed quadrature.

Generated mechanically by build_kernel.py from the two frozen source kernels.
"""

import math
import numpy as np
from numba import njit
from wctering.receiver.point_source import cap


@njit(cache=True)
def point_integrate(positions, normals, x, nodes, weights):
    centre = x[:3] * 1000.0
    norm = math.sqrt(np.dot(x[3:6], x[3:6]))
    a = x[3:6] / norm
    length = x[6] * 1000.0
    opening = x[7]
    variance = math.exp(2 * x[8])
    values = np.zeros(len(positions))
    derivatives = np.zeros((len(positions), 9))
    d = np.empty(3)
    u = np.empty(3)
    spatial = np.empty(3)
    axis = np.empty(3)
    for i in range(len(positions)):
        for j in range(len(nodes)):
            fraction = nodes[j]
            for c in range(3):
                d[c] = positions[i, c] - (centre[c] + fraction * length * a[c])
            distance = max(math.sqrt(np.dot(d, d)), 1e-9)
            for c in range(3):
                u[c] = d[c] / distance
            dot = np.dot(normals[i], u)
            acceptance, dacceptance = cap(-dot)
            denominator = distance * distance + 100.0**2
            exposure = max(acceptance / denominator, 1e-30)
            rawcos = np.dot(u, a)
            cosine = min(max(rawcos, -1.0 + 1e-10), 1.0 - 1e-10)
            sine = math.sqrt(1.0 - cosine * cosine)
            residual = math.acos(cosine) - opening
            angular = math.exp(-0.5 * residual * residual / variance)
            value = exposure * angular
            slope = residual / (variance * sine) if -1.0 + 1e-10 < rawcos < 1.0 - 1e-10 else 0.0
            for c in range(3):
                de = 0.0
                if acceptance / denominator > 1e-30:
                    de = (
                        dacceptance * (normals[i, c] - dot * u[c]) / (distance * denominator)
                        + exposure * 2 * d[c] / denominator
                    )
                spatial[c] = angular * de - value * slope * (a[c] - cosine * u[c]) / distance
                axis[c] = fraction * length * spatial[c] + value * slope * u[c]
                derivatives[i, c] += weights[j] * spatial[c] * 1000.0
                derivatives[i, 6] += weights[j] * fraction * spatial[c] * a[c] * 1000.0
            dotaxis = np.dot(axis, a)
            for c in range(3):
                derivatives[i, c + 3] += weights[j] * (axis[c] - dotaxis * a[c]) / norm
            derivatives[i, 7] += weights[j] * value * residual / variance
            derivatives[i, 8] += weights[j] * value * residual * residual / variance
            values[i] += weights[j] * value
    return values, derivatives
