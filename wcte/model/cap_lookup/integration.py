"""The existing source integrator, with only cap evaluation accelerated."""

import math
import numpy as np
from numba import njit
from lookup import cap


@njit(cache=True, fastmath={"reassoc", "contract"})
def integrate(positions, normals, x, us, weights, path, emission):
    """Same analytic integral without tiny array/BLAS allocations per PMT."""
    points, axes, jp, jt = path
    values = np.zeros(len(positions))
    jac = np.zeros((len(positions), 12))
    v0 = math.exp(2 * x[11])
    for j in range(len(us)):
        theta, ftz, dftz, ratio, za, zv, variance, vr, vf = emission[j]
        ax, ay, az = axes[j]
        px, py, pz = points[j]
        # Outside twelve Gaussian widths, omitted weight is <6e-32.
        # Cull before expensive angular/receiver functions. A zero-total
        # fallback below retains the dense calculation for degenerate trials.
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
            slope = (
                residual / (variance * math.sqrt(1 - cosine * cosine))
                if -1 + 1e-10 < rawcos < 1 - 1e-10
                else 0.0
            )
            de_x = 0.0
            de_y = 0.0
            de_z = 0.0
            if accept / den > 1e-30:
                de_x = daccept * (nx - dot * rx) / (distance * den) + exposure * 2 * dx / den
                de_y = daccept * (ny - dot * ry) / (distance * den) + exposure * 2 * dy / den
                de_z = daccept * (nz - dot * rz) / (distance * den) + exposure * 2 * dz / den
            gx = angular * ftz * de_x - value * slope * (ax - cosine * rx) / distance
            gy = angular * ftz * de_y - value * slope * (ay - cosine * ry) / distance
            gz = angular * ftz * de_z - value * slope * (az - cosine * rz) / distance
            directional = value * slope
            for k in range(12):
                gradient = gx * jp[j, 0, k] + gy * jp[j, 1, k] + gz * jp[j, 2, k]
                gradient += directional * (rx * jt[j, 0, k] + ry * jt[j, 1, k] + rz * jt[j, 2, k])
                jac[i, k] += weights[j] * gradient
            energy = exposure * angular * dftz * (1 + ratio * (residual / variance - za))
            scatter = value * (residual * residual / (2 * variance * variance) - zv)
            jac[i, 9] += weights[j] * (1000 * (1 - x[10] * us[j]) * energy + scatter * vr)
            jac[i, 10] += weights[j] * (-1000 * x[9] * us[j] * energy + scatter * vf)
            jac[i, 11] += weights[j] * scatter * 2 * v0
            values[i] += weights[j] * value
    return values, jac
