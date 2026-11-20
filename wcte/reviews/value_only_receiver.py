"""Portable receiver values for discrete starting-point scores only.

Same quadrature nodes, cap lookup, angular expression and accumulation as the
delivered cap_lookup/integration.py. No derivative arrays are formed. Full
optimization, qualification, support and ownership keep the original kernels.
"""

import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath={"reassoc", "contract"})
def kernel(positions, normals, x, us, weights, path, emission):
    points, axes, jp, jt = path
    values = np.zeros(len(positions))
    for j in range(len(us)):
        theta, ftz, dftz, ratio, za, zv, variance, vr, vf = emission[j]
        ax, ay, az = axes[j]
        px, py, pz = points[j]
        low_cos = math.cos(min(theta + 12 * math.sqrt(variance), math.pi))
        high_cos = math.cos(max(theta - 12 * math.sqrt(variance), 0.0))
        for i in range(len(positions)):
            dx = positions[i, 0] - px
            dy = positions[i, 1] - py
            dz = positions[i, 2] - pz
            distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)
            rx = dx / distance
            ry = dy / distance
            rz = dz / distance
            rawcos = rx * ax + ry * ay + rz * az
            if rawcos < low_cos or rawcos > high_cos:
                continue
            nx, ny, nz = normals[i]
            dot = nx * rx + ny * ry + nz * rz
            accept, daccept = cap(-dot)
            den = distance * distance + 10000.0
            exposure = max(accept / den, 1e-30)
            cosine = min(max(rawcos, -1 + 1e-10), 1 - 1e-10)
            residual = math.acos(cosine) - theta
            angular = math.exp(-0.5 * residual * residual / variance)
            value = exposure * angular * ftz
            values[i] += weights[j] * value
    return values


def receiver_values(positions, normals, x, order, particle):
    from charge_probe import receiver_values as render

    return render(positions, normals, x, order, particle, integrate=kernel)


def seed_log_likelihood(model, x):
    from charge_probe import log_likelihood

    assert not model.delta_scale
    return log_likelihood(model, x, integrate=kernel)


# Import after the immutable runtime has installed its qualified lookup path.
from lookup import cap
