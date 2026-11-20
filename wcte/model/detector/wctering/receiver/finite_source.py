"""Receiver-specific panels for the unchanged uniform finite cap source."""

from functools import lru_cache
import math
import numpy as np
from wctering.receiver.normalize import normalize
from numba import njit
from scipy.special import roots_legendre
from wctering.receiver.point_source import cap, T2, point_source


@lru_cache(maxsize=16)
def quadrature(order):
    x, w = roots_legendre(order)
    return np.ascontiguousarray(x * 0.5), np.ascontiguousarray(w * 0.5)


@njit(cache=True)
def add_panel(out, n, t, t0, length):
    f = (t0 - t) / length
    if -0.5 < f < 0.5 and math.isfinite(f):
        out[n] = f
        n += 1
    return n


@njit(cache=True)
def panels(dx, dy, dz, ax, ay, az, nx, ny, nz, length, opening, width):
    out = np.empty(64)
    out[0] = -0.5
    out[1] = 0.5
    n = 2
    if length == 0.0:
        return out[:2]
    t0 = dx * ax + dy * ay + dz * az
    rx = dx - t0 * ax
    ry = dy - t0 * ay
    rz = dz - t0 * az
    rho2 = rx * rx + ry * ry + rz * rz
    rho = math.sqrt(rho2)
    n = add_panel(out, n, 0.0, t0, length)
    # Angular landmarks track narrow cone peaks without a global dense rule.
    if rho > 0.0:
        for z in (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0):
            theta = opening + z * width
            if 0.0 < theta < math.pi:
                n = add_panel(out, n, rho / math.tan(theta), t0, length)
        clip = math.acos(1.0 - 1e-10)
        n = add_panel(out, n, rho / math.tan(clip), t0, length)
        n = add_panel(out, n, -rho / math.tan(clip), t0, length)
        for z in (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0):
            n = add_panel(out, n, z * rho, t0, length)
    # Distance denominator varies on sqrt(rho^2 + 100 mm^2).
    scale = math.sqrt(rho2 + 10000.0)
    for z in (-4.0, -1.0, 1.0, 4.0):
        n = add_panel(out, n, z * scale, t0, length)
    # mu=-(n_parallel*t+B)/sqrt(t^2+rho^2); solve mu^2=T^2.
    parallel = nx * ax + ny * ay + nz * az
    B = nx * rx + ny * ry + nz * rz
    qa = parallel * parallel - T2
    qb = 2.0 * parallel * B
    qc = B * B - T2 * rho2
    if abs(qa) < 1e-14:
        if abs(qb) > 1e-30:
            n = add_panel(out, n, -qc / qb, t0, length)
    else:
        discr = qb * qb - 4.0 * qa * qc
        if discr >= 0.0:
            q = -0.5 * (qb + math.copysign(math.sqrt(discr), qb))
            if q != 0.0:
                n = add_panel(out, n, q / qa, t0, length)
                n = add_panel(out, n, qc / q, t0, length)
            else:
                n = add_panel(out, n, -qb / (2.0 * qa), t0, length)
    if abs(B) > 1e-30:
        n = add_panel(out, n, parallel * rho2 / B, t0, length)
    return np.sort(out[:n])


@njit(cache=True)
def t_panels(t0, rho, parallel, B, length, opening, width):
    lo = t0 - 0.5 * length
    hi = t0 + 0.5 * length
    candidates = np.empty(160)
    n = 0
    candidates[n] = lo
    n += 1
    candidates[n] = hi
    n += 1
    candidates[n] = 0.0
    n += 1
    if rho > 0.0:
        for z in (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0):
            theta = opening + z * width
            if 0.0 < theta < math.pi:
                candidates[n] = rho / math.tan(theta)
                n += 1
        clip = math.acos(1.0 - 1e-10)
        candidates[n] = rho / math.tan(clip)
        n += 1
        candidates[n] = -rho / math.tan(clip)
        n += 1
        for z in (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0):
            candidates[n] = z * rho
            n += 1
        scale = 8.0 * rho
        stop = math.sqrt(rho * rho + 10000.0)
        for unused in range(50):
            if scale >= stop:
                break
            scale *= 4.0
            candidates[n] = scale
            n += 1
            candidates[n] = -scale
            n += 1
    scale = math.sqrt(rho * rho + 10000.0)
    for z in (-4.0, -1.0, 1.0, 4.0):
        candidates[n] = z * scale
        n += 1
    qa = parallel * parallel - T2
    qb = 2.0 * parallel * B
    qc = B * B - T2 * rho * rho
    if abs(qa) < 1e-14:
        if abs(qb) > 1e-30:
            candidates[n] = -qc / qb
            n += 1
    else:
        discr = qb * qb - 4.0 * qa * qc
        if discr >= 0.0:
            q = -0.5 * (qb + math.copysign(math.sqrt(discr), qb))
            if q != 0.0:
                candidates[n] = q / qa
                n += 1
                candidates[n] = qc / q
                n += 1
            else:
                candidates[n] = -qb / (2.0 * qa)
                n += 1
    if abs(B) > 1e-30:
        candidates[n] = parallel * rho * rho / B
        n += 1
    edges = np.empty(n)
    m = 0
    for j in range(n):
        if lo <= candidates[j] <= hi and math.isfinite(candidates[j]):
            edges[m] = candidates[j]
            m += 1
    return np.sort(edges[:m])


@njit(cache=True)
def ray_value(xx, yy, zz, nx, ny, nz, ax, ay, az, opening, variance):
    r = max(math.sqrt(xx * xx + yy * yy + zz * zz), 1e-9)
    ux = xx / r
    uy = yy / r
    uz = zz / r
    acceptance, unused = cap(-(nx * ux + ny * uy + nz * uz))
    cosine = min(max(ux * ax + uy * ay + uz * az, -1.0 + 1e-10), 1.0 - 1e-10)
    residual = math.acos(cosine) - opening
    return max(acceptance / (r * r + 10000.0), 1e-30) * math.exp(-0.5 * residual * residual / variance)


@njit(cache=True)
def integrate(positions, normals, x, nodes, weights):
    cx = x[0] * 1000.0
    cy = x[1] * 1000.0
    cz = x[2] * 1000.0
    norm = math.sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5])
    ax = x[3] / norm
    ay = x[4] / norm
    az = x[5] / norm
    length = x[6] * 1000.0
    opening = x[7]
    variance = math.exp(2.0 * x[8])
    width = math.sqrt(variance)
    values = np.zeros(len(positions))
    derivatives = np.zeros((len(positions), 9))
    counts = np.zeros(len(positions), np.int64)
    for i in range(len(positions)):
        dx = positions[i, 0] - cx
        dy = positions[i, 1] - cy
        dz = positions[i, 2] - cz
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]
        t0 = dx * ax + dy * ay + dz * az
        rx = dx - t0 * ax
        ry = dy - t0 * ay
        rz = dz - t0 * az
        rho = math.sqrt(rx * rx + ry * ry + rz * rz)
        near_chart = 0.0 < rho < 0.5 * length
        if near_chart:
            edges = t_panels(
                t0,
                rho,
                nx * ax + ny * ay + nz * az,
                nx * rx + ny * ry + nz * rz,
                length,
                opening,
                width,
            )
        else:
            edges = panels(dx, dy, dz, ax, ay, az, nx, ny, nz, length, opening, width)
        for b in range(len(edges) - 1):
            left = edges[b]
            right = edges[b + 1]
            angular_chart = near_chart and max(abs(left), abs(right)) <= 8.0 * rho * (1.0 + 1e-14)
            if angular_chart:
                left = math.atan2(rho, edges[b + 1])
                right = math.atan2(rho, edges[b])
            span = right - left
            if span <= 0.0:
                continue
            counts[i] += len(nodes)
            for j in range(len(nodes)):
                # The cap derivative has a fractional power at grazing
                # boundaries. Quadratic approach at both panel endpoints
                # (a monotone cubic map) smooths it for Gauss integration.
                v = nodes[j] + 0.5
                abscissa = left + span * v * v * (3.0 - 2.0 * v)
                weight = span * weights[j] * 6.0 * v * (1.0 - v)
                if angular_chart:
                    t = rho / math.tan(abscissa)
                    f = (t0 - t) / length
                    weight *= rho / (length * math.sin(abscissa) ** 2)
                    xx = rx + t * ax
                    yy = ry + t * ay
                    zz = rz + t * az
                elif near_chart:
                    t = abscissa
                    f = (t0 - t) / length
                    weight /= length
                    xx = rx + t * ax
                    yy = ry + t * ay
                    zz = rz + t * az
                else:
                    f = abscissa
                    xx = dx - f * length * ax
                    yy = dy - f * length * ay
                    zz = dz - f * length * az
                r = max(math.sqrt(xx * xx + yy * yy + zz * zz), 1e-9)
                ux = xx / r
                uy = yy / r
                uz = zz / r
                dot = nx * ux + ny * uy + nz * uz
                acceptance, dacceptance = cap(-dot)
                denominator = r * r + 10000.0
                raw = acceptance / denominator
                exposure = max(raw, 1e-30)
                rawcos = ux * ax + uy * ay + uz * az
                cosine = min(max(rawcos, -1.0 + 1e-10), 1.0 - 1e-10)
                sine = math.sqrt(1.0 - cosine * cosine)
                residual = math.acos(cosine) - opening
                angular = math.exp(-0.5 * residual * residual / variance)
                value = exposure * angular
                slope = residual / (variance * sine) if -1.0 + 1e-10 < rawcos < 1.0 - 1e-10 else 0.0
                ex = 0.0
                ey = 0.0
                ez = 0.0
                if raw > 1e-30:
                    first = dacceptance / (r * denominator)
                    second = 2.0 * exposure / denominator
                    ex = first * (nx - dot * ux) + second * xx
                    ey = first * (ny - dot * uy) + second * yy
                    ez = first * (nz - dot * uz) + second * zz
                sx = angular * ex - value * slope * (ax - cosine * ux) / r
                sy = angular * ey - value * slope * (ay - cosine * uy) / r
                sz = angular * ez - value * slope * (az - cosine * uz) / r
                vx = f * length * sx + value * slope * ux
                vy = f * length * sy + value * slope * uy
                vz = f * length * sz + value * slope * uz
                va = vx * ax + vy * ay + vz * az
                derivatives[i, 0] += weight * sx * 1000.0
                derivatives[i, 1] += weight * sy * 1000.0
                derivatives[i, 2] += weight * sz * 1000.0
                derivatives[i, 3] += weight * (vx - va * ax) / norm
                derivatives[i, 4] += weight * (vy - va * ay) / norm
                derivatives[i, 5] += weight * (vz - va * az) / norm
                derivatives[i, 6] += weight * f * (sx * ax + sy * ay + sz * az) * 1000.0
                derivatives[i, 7] += weight * value * residual / variance
                derivatives[i, 8] += weight * value * residual * residual / variance
                values[i] += weight * value
        if near_chart and length > 1e-5 * max(abs(t0), rho, 100.0):
            # Fundamental-theorem identities avoid cancellation of huge
            # opposite longitudinal derivatives on almost collinear rays.
            start = ray_value(
                dx + 0.5 * length * ax,
                dy + 0.5 * length * ay,
                dz + 0.5 * length * az,
                nx,
                ny,
                nz,
                ax,
                ay,
                az,
                opening,
                variance,
            )
            end = ray_value(
                dx - 0.5 * length * ax,
                dy - 0.5 * length * ay,
                dz - 0.5 * length * az,
                nx,
                ny,
                nz,
                ax,
                ay,
                az,
                opening,
                variance,
            )
            parallel = derivatives[i, 0] * ax + derivatives[i, 1] * ay + derivatives[i, 2] * az
            correction = 1000.0 * (end - start) / length - parallel
            derivatives[i, 0] += correction * ax
            derivatives[i, 1] += correction * ay
            derivatives[i, 2] += correction * az
            derivatives[i, 6] = 1000.0 * (0.5 * (start + end) - values[i]) / length
    return values, derivatives, counts


def raw(positions, normals, x, order=16, *, scalar=False):
    x = np.asarray(x, float)
    if x.shape != (9,) or not np.isfinite(x).all() or x[6] < 0 or np.linalg.norm(x[3:6]) < 1e-10:
        raise ValueError("Invalid source")
    from wctering.receiver.native import library, finite_integrate

    if library(scalar) is not None:
        return finite_integrate(positions, normals, x, *quadrature(order), reference=scalar)
    return integrate(positions, normals, x, *quadrature(order))


def _source(positions, normals, x, order=16, *, scalar=False):
    x = np.asarray(x, float)
    if x[6] == 0.0:
        s, j = point_source(positions, normals, np.concatenate((x[:6], x[7:])))
        return s, np.column_stack((j[:, :6], np.zeros(len(s)), j[:, 6:]))
    values, jac, _ = raw(positions, normals, x, order, scalar=scalar)
    return normalize(values, jac)


def scalar_source(positions, normals, parameters, order=16):
    """Reference arithmetic for charge endpoints and their occupancy starts."""
    from wctering.receiver.cache import evaluate

    return evaluate(
        lambda p, n, x: _source(p, n, x, order, scalar=True),
        positions,
        normals,
        parameters,
        kind="finite_scalar",
        order=order,
    )
