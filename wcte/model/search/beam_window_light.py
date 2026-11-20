"""Integrate a charged path's direct and specular beam-window light."""

from dataclasses import dataclass
from functools import lru_cache

from numba import njit
import numpy as np
from scipy.special import roots_legendre

from material import material
from physical_pulse import flight_integral
from source import angular_rule, material_panels, moment_table, polynomial, rule, states, trajectory
from wctering.receiver.point_source import T, T2, Z0, cap
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


@dataclass(frozen=True)
class Window:
    # The water-facing surface in the detector survey frame, in mm.
    # WCSim v1.12.29 WCSimConstructBeamPipe.cc specifies the 77 mm radius.
    # A working optical coefficient is explicit so measured reflectivity
    # can replace the smooth simulation surface for real-data fits.
    z: float = -1348.763
    radius: float = 77.0
    reflectivity: float = 0.9


@njit(cache=True)
def rays(source, receiver, z, radius):
    """Return whether a direct ray is blocked and a specular ray is valid."""
    dz = receiver[2] - source[2]
    opposite = (source[2] - z) * (receiver[2] - z) < 0.0
    if opposite and dz != 0.0:
        u = (z - source[2]) / dz
        x = source[0] + u * (receiver[0] - source[0])
        y = source[1] + u * (receiver[1] - source[1])
        blocked = x * x + y * y <= radius * radius
    else:
        blocked = False
    a, b = source[2] - z, receiver[2] - z
    if a <= 0.0 or b <= 0.0:
        return blocked, False
    u = a / (a + b)
    x = source[0] + u * (receiver[0] - source[0])
    y = source[1] + u * (receiver[1] - source[1])
    return blocked, x * x + y * y <= radius * radius


@njit(cache=True)
def integrate(position, normal, points, axes, moments, weight, flight, core, z, radius, reflection):
    """Render one emission integral and its first two time moments per path."""
    rate = np.zeros((4, len(position)))
    time = np.zeros_like(rate)
    time2 = np.zeros_like(rate)
    for j in range(len(points)):
        source, axis = points[j], axes[j]
        theta, strength, variance = moments[j]
        for i in range(len(position)):
            receiver = position[i]
            blocked, mirror = rays(source, receiver, z, radius)
            dx, dy, dz = receiver - source
            for reflected in (0, 1):
                if (reflected == 0 and blocked) or (reflected == 1 and not mirror):
                    continue
                # Reflect the receiver across the plane. The emitter sees
                # the downward ray, while the tube sees the reflected ray.
                longitudinal = dz if reflected == 0 else 2.0 * z - receiver[2] - source[2]
                distance = np.sqrt(dx * dx + dy * dy + longitudinal * longitudinal)
                if distance <= 0.0:
                    continue
                rx, ry, rz = dx / distance, dy / distance, longitudinal / distance
                nz = rz if reflected == 0 else -rz
                incidence = -(rx * normal[i, 0] + ry * normal[i, 1] + nz * normal[i, 2])
                aperture = cap(incidence)[0]
                if aperture <= 0.0:
                    continue
                cosine = min(1.0, max(-1.0, rx * axis[0] + ry * axis[1] + rz * axis[2]))
                residual = np.arccos(cosine) - theta
                value = (
                    weight[j]
                    * strength
                    * aperture
                    * np.exp(-0.5 * residual**2 / variance)
                    / (distance**2 + 10000.0)
                )
                if reflected:
                    value *= reflection
                index = 2 * reflected + int(not core[j])
                arrival = flight[j] + N_GROUP * distance / C
                rate[index, i] += value
                time[index, i] += value * arrival
                time2[index, i] += value * arrival * arrival
    return rate, time, time2


def project(model, path, particle, order, *, window=Window()):
    """Render above- and below-margin light for both optical paths."""
    water = material(particle)
    ran = 1000.0 * path[9]
    length = ran * path[10]
    cut = np.clip((ran - water["margin_mm"]) / length, 0.0, 1.0)
    edges = material_panels(path, cut, water)
    start, span = edges[:-1], np.diff(edges)
    node, strength = rule(order)
    u = (start[:, None] + span[:, None] * node).ravel()
    weights = length * (span[:, None] * strength).ravel()
    points, axes, *_ = trajectory(path, u)
    value = states(
        path,
        u,
        water["distance"],
        water["energy_at"].c,
        water["scatter_at"].c,
        *angular_rule(),
        False,
        water["mass"],
        water["index"],
        *moment_table(),
        water["margin_mm"],
    )
    primitive = flight_integral(particle)
    elapsed = (primitive(ran) - primitive(np.maximum(ran - length * u, 0.0))) / C
    good = (value[:, 1] > 0.0) & (weights > 0.0)
    rate, first, second = integrate(
        model.pos,
        model.normals,
        points[good],
        axes[good],
        value[good][:, [0, 1, 6]],
        weights[good],
        elapsed[good],
        u[good] < cut,
        window.z,
        window.radius,
        window.reflectivity,
    )
    valid = rate > 1e-100
    time = np.divide(first, rate, out=np.zeros_like(first), where=valid)
    var = np.divide(second, rate, out=np.zeros_like(second), where=valid) - time**2
    return rate, time[:, model.ix].T, np.maximum(var[:, model.ix].T, 0.0)


@njit(cache=True)
def add_root(edges, n, root, length):
    if 0.0 < root < length:
        edges[n] = root
        return n + 1
    return n


@njit(cache=True)
def intervals(source, axis, receiver, length, z, radius, reflected, edges):
    """Split a straight track exactly at the finite disc's projected rim.

    The plane intersection has a linear numerator and denominator in
    arclength. Requiring x²+y² ≤ R² gives a quadratic with at most two
    roots. Split also at the plane because a reflection requires both ends
    of the optical path to be in front of the water-facing window.
    """
    a0 = source[2] - z
    bz = receiver[2] - z
    az = axis[2]
    edges[0], edges[1] = 0.0, length
    n = 2
    if az != 0.0:
        n = add_root(edges, n, -a0 / az, length)
    sign = 1.0 if reflected else -1.0
    x = bz * source[0] + sign * a0 * receiver[0]
    y = bz * source[1] + sign * a0 * receiver[1]
    dx = bz * axis[0] + sign * az * receiver[0]
    dy = bz * axis[1] + sign * az * receiver[1]
    w, dw = bz + sign * a0, sign * az
    c = x * x + y * y - (radius * w) ** 2
    b = 2.0 * (x * dx + y * dy - radius**2 * w * dw)
    a = dx * dx + dy * dy - (radius * dw) ** 2
    discriminant = b * b - 4.0 * a * c
    if abs(a) < 1e-14:
        if abs(b) > 1e-14:
            n = add_root(edges, n, -c / b, length)
    elif discriminant > 0.0:
        q = -0.5 * (b + np.copysign(np.sqrt(discriminant), b))
        if q != 0.0:
            n = add_root(edges, n, q / a, length)
            n = add_root(edges, n, c / q, length)
    # At most five endpoints. Sort this small buffer in place so the hot
    # receiver loop has no temporary copy from a general-purpose sort.
    for i in range(1, n):
        edge = edges[i]
        j = i
        while j > 0 and edge < edges[j - 1]:
            edges[j] = edges[j - 1]
            j -= 1
        edges[j] = edge
    return n


@njit(cache=True, inline="always")
def cap_value(mu):
    """Evaluate just the projected receiver area for a value-only fit."""
    if mu >= T:
        return mu
    if mu <= -T:
        return 0.0
    h = np.sqrt(max(T2 - mu * mu, 0.0))
    phi = np.arctan2(h, -mu * Z0)
    return max((np.arctan2(h, Z0) - Z0 * h + mu * T2 * phi) / (np.pi * T2), 0.0)


@njit(cache=True)
def clipped_straight(
    positions, normals, origin, axis, length, opening, width, nodes, weights, z, radius, reuse_storage=False
):
    """Render direct and specular finite light through the window intervals."""
    strength = np.empty((2, len(positions)))
    time = np.zeros_like(strength)
    variance = np.zeros_like(strength)
    inverse_beta = min(1.4, max(1.0, 1.344 * np.cos(opening)))
    edge_storage = np.empty(5)
    for reflect in (0, 1):
        for i in range(len(positions)):
            receiver = positions[i]
            value, first, second = 0.0, 0.0, 0.0
            if reflect and receiver[2] <= z:
                strength[reflect, i] = 0.0
                continue
            # The rim calculation uses five doubles. Reusing that storage
            # leaves the interval roots and quadrature exactly as before.
            edges = edge_storage if reuse_storage else np.empty(5)
            count = intervals(origin, axis, receiver, length, z, radius, reflect, edges)
            for j in range(count - 1):
                left, right = edges[j], edges[j + 1]
                span = right - left
                if span < 1e-9:
                    continue
                # Keep every quadrature panel even on unobstructed
                # direct rays to preserve the numerical integral.
                at = 0.5 * (right + left)
                hx = origin[0] + at * axis[0]
                hy = origin[1] + at * axis[1]
                hz = origin[2] + at * axis[2]
                a, b = hz - z, receiver[2] - z
                hits_plane = (a > 0.0 and b > 0.0) if reflect else (a * b < 0.0)
                if hits_plane:
                    fraction = a / (a + b if reflect else a - b)
                    x = hx + fraction * (receiver[0] - hx)
                    y = hy + fraction * (receiver[1] - hy)
                    if (x * x + y * y <= radius * radius) != bool(reflect):
                        continue
                elif reflect:
                    continue
                for v, w in zip(nodes, weights):
                    emit = left + span * v
                    if reuse_storage:
                        # The former length-three array allocated twice per
                        # quadrature node. Each scalar has the same operation order.
                        x = origin[0] + emit * axis[0]
                        y = origin[1] + emit * axis[1]
                        sz = origin[2] + emit * axis[2]
                    else:
                        x, y, sz = origin + emit * axis
                    dx, dy = receiver[0] - x, receiver[1] - y
                    dz = 2.0 * z - receiver[2] - sz if reflect else receiver[2] - sz
                    distance2 = dx * dx + dy * dy + dz * dz
                    distance = np.sqrt(max(distance2, 1e-14))
                    nz = -dz if reflect else dz
                    mu = -(normals[i, 0] * dx + normals[i, 1] * dy + normals[i, 2] * nz) / distance
                    aperture = cap_value(mu)
                    if aperture <= 0:
                        continue
                    cosine = (axis[0] * dx + axis[1] * dy + axis[2] * dz) / distance
                    theta = np.arccos(min(1.0, max(-1.0, cosine)))
                    light = (
                        span
                        * w
                        * aperture
                        * np.exp(-0.5 * ((theta - opening) / width) ** 2)
                        / (distance2 + 10000.0)
                    )
                    at = (inverse_beta * emit + N_GROUP * distance) / C
                    value += light
                    first += light * at
                    second += light * at * at
            strength[reflect, i] = value
            if value > 1e-100:
                at = first / value
                time[reflect, i] = at
                variance[reflect, i] = max(second / value - at * at, 0.0)
    return strength, time, variance


def finite(
    model,
    origin,
    axis,
    length,
    opening,
    width,
    order=12,
    window=Window(),
    *,
    all_times=False,
    reuse_storage=False,
):
    """Optical shape with exact disc clipping and common light normalization."""
    from angular import normalization

    beta2 = min(1.0, (N_PHASE * np.cos(opening)) ** -2)
    grid, table, *_ = moment_table()
    emitted = polynomial(beta2, grid, table)[0] if beta2 > grid[0] else 0.0
    angles, angle_weight = angular_rule()
    scale = emitted / normalization(opening, width**2, angles, angle_weight)[0]
    u, w = legendre(order)
    rate, time, variance = clipped_straight(
        model.pos,
        model.normals,
        origin,
        axis,
        length,
        opening,
        width,
        u,
        w,
        window.z,
        window.radius,
        reuse_storage=reuse_storage,
    )
    index = slice(None) if all_times else model.ix
    return scale * rate, time[:, index].T, variance[:, index].T


@lru_cache(maxsize=6)
def legendre(order):
    nodes, weights = roots_legendre(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights
