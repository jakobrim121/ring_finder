"""Energy-coupled electron light along a circular mean trajectory.

Parameters: start(m), axis chart(3), transverse total-turn chart(3), initial
visible range(m), traversed fraction, log instrumental angular width(rad).
The mean curve has exactly the CSDA arclength. ESTAR stopping and Fermi–Rossi
scattering determine opening, yield and angular diffusion at each point.

Two declared approximations are available: free angular diffusion, and an
angular bridge conditioned at the endpoint. The circular mean is a compact
approximation to a particular trajectory; neither model is exact electron
transport. In particular, lateral scattering and bremsstrahlung secondaries
are not represented by this single source. Adequacy must be measured.
"""

from functools import lru_cache
import math
import numpy as np
from numba import njit
from scipy.special import roots_legendre
from material import material, bounded_path
from spectrum import moment_table
from angular import normalization
from wctering.receiver.point_source import cap


@lru_cache(None)
def rule(order):
    v, w = roots_legendre(order)
    v = (v + 1) / 2
    return 2 * v - v * v, w * (1 - v)


@lru_cache(None)
def angular_rule():
    v, w = roots_legendre(64)
    theta = (v + 1) * np.pi / 2
    return theta, w * np.pi**2 * np.sin(theta)


@njit(cache=True)
def polynomial(value, grid, coef):
    j = min(max(np.searchsorted(grid, value, side="right") - 1, 0), len(grid) - 2)
    d = value - grid[j]
    return ((coef[0, j] * d + coef[1, j]) * d + coef[2, j]) * d + coef[3, j], (
        3 * coef[0, j] * d + 2 * coef[1, j]
    ) * d + coef[2, j]


@njit(cache=True)
def trajectory(x, us):
    """Exact constant-curvature position/tangent, with analytic Jacobians."""
    norm = np.linalg.norm(x[3:6])
    a = x[3:6] / norm
    P = np.eye(3) - np.outer(a, a)
    ja = np.zeros((3, 12))
    ja[:, 3:6] = P / norm
    v = x[6:9]
    bend = P @ v
    power = bend @ bend
    jb = np.zeros((3, 12))
    jb[:, 3:6] = (-(a @ v) * np.eye(3) - np.outer(a, v)) @ ja[:, 3:6]
    jb[:, 6:9] = P
    dpower = 2 * bend @ jb
    length = 1000 * x[9] * x[10]
    dl = np.zeros(12)
    dl[9] = 1000 * x[10]
    dl[10] = 1000 * x[9]
    pos = np.empty((len(us), 3))
    axis = pos.copy()
    jp = np.zeros((len(us), 3, 12))
    jt = jp.copy()
    for i, u in enumerate(us):
        if power < 1e-6:
            C = 1 - power * u * u / 2 + power * power * u**4 / 24 - power**3 * u**6 / 720
            S = u - power * u**3 / 6 + power * power * u**5 / 120 - power**3 * u**7 / 5040
            H = u * u / 2 - power * u**4 / 24 + power * power * u**6 / 720 - power**3 * u**8 / 40320
            dC = -u * u / 2 + power * u**4 / 12 - power * power * u**6 / 240
            dS = -(u**3) / 6 + power * u**5 / 60 - power * power * u**7 / 1680
            dH = -(u**4) / 24 + power * u**6 / 360 - power * power * u**8 / 13440
        else:
            angle = math.sqrt(power) * u
            C = math.cos(angle)
            S = math.sin(angle) / math.sqrt(power)
            H = (1 - C) / power
            dC = -0.5 * u * S
            dS = (u * C - S) / (2 * power)
            dH = (0.5 * u * S - H) / power
        displacement = S * a + H * bend
        pos[i] = 1000 * x[:3] + length * displacement
        axis[i] = C * a + S * bend
        jp[i] = length * (S * ja + H * jb + np.outer(dS * a + dH * bend, dpower)) + np.outer(displacement, dl)
        jt[i] = C * ja + S * jb + np.outer(dC * a + dS * bend, dpower)
        for j in range(3):
            jp[i, j, j] += 1000
    return pos, axis, jp, jt


@njit(cache=True)
def states(
    x,
    us,
    grid,
    energy_coef,
    scatter_coef,
    angles,
    angle_weights,
    bridge,
    mass,
    index,
    spectral_grid,
    moment0,
    moment1,
    moment2,
    margin,
):
    """Opening, normalized Frank–Tamm yield and material derivatives."""
    out = np.zeros((len(us), 9))
    r = 1000 * x[9]
    f = x[10]
    v0 = math.exp(2 * x[11])
    # A fitted core represents one realized bent trajectory. Do not smear
    # its photons by a second ensemble of complete core trajectories.
    # Below the fixed ring-energy margin, resume unresolved angular diffusion
    # from the core endpoint. A source born below the margin retains the
    # original full diffusion. This is a source hypothesis, not a truth input.
    start = min(r, margin)
    drstart = 1000.0 if r < margin else 0.0
    s0, ds0 = polynomial(start, grid, scatter_coef)
    end, dend = polynomial(r * (1 - f), grid, scatter_coef)
    qe = max(end - s0, 0.0)
    qer = 1000 * dend * (1 - f) - drstart * ds0
    qef = -r * dend
    if qe == 0.0:
        qer = 0.0
        qef = 0.0
    for i, u in enumerate(us):
        remaining = r * (1 - f * u)
        e, de = polynomial(remaining, grid, energy_coef)
        s, ds = polynomial(remaining, grid, scatter_coef)
        q = max(s - s0, 0.0)
        qr = 1000 * ds * (1 - f * u) - drstart * ds0
        qf = -r * u * ds
        if q == 0.0:
            qr = 0.0
            qf = 0.0
        if bridge and qe > 1e-15:
            ratio = q / qe
            q = q * (1 - ratio)
            qr = (1 - 2 * ratio) * qr + ratio * ratio * qer
            qf = (1 - 2 * ratio) * qf + ratio * ratio * qef
        gamma = 1 + e / mass
        beta2 = 1 - 1 / (gamma * gamma)
        dbeta = 2 * de / (mass * gamma**3)
        ft, dft = polynomial(beta2, spectral_grid, moment0)
        m1, dm1 = polynomial(beta2, spectral_grid, moment1)
        m2, dm2 = polynomial(beta2, spectral_grid, moment2)
        if ft <= 1e-30:
            out[i, 6] = v0 + max(q, 0.0)
            continue
        theta = m1 / ft
        dtheta = (dm1 - theta * dft) / ft
        chromatic = max(m2 / ft - theta * theta, 0.0)
        dchromatic = (dm2 - (m2 / ft) * dft) / ft - 2 * theta * dtheta
        variance = v0 + max(q, 0.0) + chromatic
        qr += 1000 * (1 - f * u) * dchromatic * dbeta
        qf += -r * u * dchromatic * dbeta
        z, za, zv = normalization(theta, variance, angles, angle_weights)
        out[i, 0] = theta
        out[i, 1] = ft / z
        out[i, 2] = dft * dbeta / z
        out[i, 3] = ft * dtheta / dft
        out[i, 4] = za
        out[i, 5] = zv
        out[i, 6] = variance
        out[i, 7] = qr
        out[i, 8] = qf
    return out


@njit(cache=True)
def integrate_reference(positions, normals, x, us, weights, path, emission):
    points, axes, jp, jt = path
    values = np.zeros(len(positions))
    jac = np.zeros((len(positions), 12))
    v0 = math.exp(2 * x[11])
    for j in range(len(us)):
        theta, ftz, dftz, ratio, za, zv, variance, vr, vf = emission[j]
        for i in range(len(positions)):
            delta = positions[i] - points[j]
            distance = max(np.linalg.norm(delta), 1e-9)
            ray = delta / distance
            dot = normals[i] @ ray
            accept, daccept = cap(-dot)
            den = distance * distance + 10000.0
            exposure = max(accept / den, 1e-30)
            rawcos = ray @ axes[j]
            cosine = min(max(rawcos, -1 + 1e-10), 1 - 1e-10)
            residual = math.acos(cosine) - theta
            angular = math.exp(-0.5 * residual * residual / variance)
            value = exposure * angular * ftz
            slope = (
                residual / (variance * math.sqrt(1 - cosine * cosine))
                if -1 + 1e-10 < rawcos < 1 - 1e-10
                else 0.0
            )
            spatial = np.zeros(3)
            direction = value * slope * ray
            for c in range(3):
                de = 0.0
                if accept / den > 1e-30:
                    de = (
                        daccept * (normals[i, c] - dot * ray[c]) / (distance * den)
                        + exposure * 2 * delta[c] / den
                    )
                spatial[c] = angular * ftz * de - value * slope * (axes[j, c] - cosine * ray[c]) / distance
            grad = spatial @ jp[j] + direction @ jt[j]
            energy = exposure * angular * dftz * (1 + ratio * (residual / variance - za))
            scatter = value * (residual * residual / (2 * variance * variance) - zv)
            grad[9] += 1000 * (1 - x[10] * us[j]) * energy + scatter * vr
            grad[10] += -1000 * x[9] * us[j] * energy + scatter * vf
            grad[11] += scatter * 2 * v0
            values[i] += weights[j] * value
            jac[i] += weights[j] * grad
    return values, jac


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


def material_panels(x, cut, m):
    """Integration panels for an explicitly supplied transport table."""
    r = 1000 * x[9]
    length = r * x[10]
    result = [0.0, 0.25, 0.5, 0.75, 1.0, cut]
    if length > 0:
        power = -float(m["scatter_at"].derivative()(r))
        step = np.exp(2 * x[11]) / (power * length)
        while 0 < step < 1:
            result.append(step)
            step *= 4
        for relative in (0.5, 0.25, 0.1, 0.01, 0.001, 0.0001):
            u = (1 - relative) / x[10]
            if 0 < u < 1:
                result.append(u)
    return np.unique(result)


def raw(positions, normals, x, order=16, bridge=False, particle="electron", reference=False):
    return raw_material(positions, normals, x, material(particle), order, bridge, reference)


def raw_material(positions, normals, x, m, order=16, bridge=False, reference=False):
    """Same source with a supplied table; no process-wide material mutation."""
    x = bounded_path(x, m["maximum_range_mm"])
    r = 1000 * x[9]
    f = x[10]
    cut = (1 - m["margin_mm"] / r) / f if f > 0 else (1.0 if r >= m["margin_mm"] else 0.0)
    cut = float(np.clip(cut, 0, 1))
    dc = np.zeros(12)
    if 0 < cut < 1:
        dc[9] = m["margin_mm"] / (1000 * x[9] ** 2 * f)
        dc[10] = -cut / f
    base, weight = rule(order)
    bounds = material_panels(x, cut, m)
    parts = []
    derivatives = []

    def evaluate(us, weights):
        path = trajectory(x, us)
        light = states(
            x,
            us,
            m["distance"],
            m["energy_at"].c,
            m["scatter_at"].c,
            *angular_rule(),
            bridge,
            m["mass"],
            m["index"],
            *moment_table(),
            m["margin_mm"],
        )
        result = (integrate_reference if reference else integrate)(
            positions, normals, x, us, weights, path, light
        )
        if not reference and result[0].sum() < 1e-250:
            result = integrate_reference(positions, normals, x, us, weights, path, light)
        return result

    for low, high in ((0.0, cut), (cut, 1.0)):
        if low == high:
            parts.append(np.zeros(len(positions)))
            derivatives.append(np.zeros((len(positions), 12)))
            continue
        edges = bounds[(bounds >= low) & (bounds <= high)]
        us = np.concatenate([a + (b - a) * base for a, b in zip(edges[:-1], edges[1:])])
        weights = np.concatenate([(b - a) * weight for a, b in zip(edges[:-1], edges[1:])])
        value, jac = evaluate(us, weights)
        parts.append(value)
        derivatives.append(jac)
    if np.any(dc):
        boundary = evaluate(np.array([cut]), np.ones(1))[0]
        derivatives[0] += boundary[:, None] * dc
        derivatives[1] -= boundary[:, None] * dc
    return np.asarray(parts), np.asarray(derivatives)
