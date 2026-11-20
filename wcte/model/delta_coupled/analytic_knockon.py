"""Smooth interpolation and exact geometry derivatives of the knock-on source.

The physics table is unchanged. Tensor cubic splines replace piecewise-linear
interpolation so an optimizer sees continuous first and second derivatives.
Qualification compares the two interpolants, independently checks derivatives,
and still requires the usual doubled-quadrature likelihood agreement.
"""

from functools import lru_cache
import math
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from knockon_light import table, material, bounded_path, trajectory, cap


@lru_cache(None)
def coefficients():
    saved = table()
    # The first spline yields polar polynomial coefficients for each parent
    # speed. Interpolating those coefficients gives the tensor product, with
    # no fit to event data and no extra inference parameters.
    polar = CubicSpline(saved["polar_rad"], saved["density_per_sr"], axis=1)
    opening = CubicSpline(saved["opening_rad"], polar.c, axis=2)
    return np.ascontiguousarray(opening.c.transpose(1, 3, 0, 2))


@njit(cache=True)
def angular_value(opening, polar, opening_grid, polar_grid, coef):
    a = min(max(np.searchsorted(opening_grid, opening) - 1, 0), len(opening_grid) - 2)
    b = min(max(np.searchsorted(polar_grid, polar) - 1, 0), len(polar_grid) - 2)
    bounded = min(max(opening, opening_grid[0]), opening_grid[-1])
    x, y = bounded - opening_grid[a], polar - polar_grid[b]
    c = coef[a, b]
    v = np.empty(4)
    dy = np.empty(4)
    for j in range(4):
        v[j] = ((c[j, 0] * y + c[j, 1]) * y + c[j, 2]) * y + c[j, 3]
        dy[j] = (3 * c[j, 0] * y + 2 * c[j, 1]) * y + c[j, 2]
    value = ((v[0] * x + v[1]) * x + v[2]) * x + v[3]
    dx = (3 * v[0] * x + 2 * v[1]) * x + v[2]
    dp = ((dy[0] * x + dy[1]) * x + dy[2]) * x + dy[3]
    if opening <= opening_grid[0] or opening >= opening_grid[-1]:
        dx = 0.0
    return value, dx, dp


@njit(cache=True)
def integrate(
    positions,
    normals,
    points,
    axes,
    jp,
    jt,
    openings,
    opening_jac,
    weights,
    length,
    length_jac,
    opening_grid,
    polar_grid,
    coef,
):
    values = np.zeros(len(positions))
    jac = np.zeros((len(positions), 12))
    for j in range(len(weights)):
        for i in range(len(positions)):
            delta = positions[i] - points[j]
            distance = max(np.linalg.norm(delta), 1e-9)
            ray = delta / distance
            cosine = min(max(ray @ axes[j], -1.0), 1.0)
            density, da, dp = angular_value(openings[j], math.acos(cosine), opening_grid, polar_grid, coef)
            dot = normals[i] @ ray
            accept, daccept = cap(-dot)
            den = distance * distance + 10000.0
            exposure = max(accept / den, 1e-30)
            value = exposure * density
            slope = -dp / math.sqrt(1 - cosine * cosine) if abs(cosine) < 1 - 1e-12 else 0.0
            spatial = -exposure * slope * (axes[j] - cosine * ray) / distance
            if accept / den > 1e-30:
                spatial += density * (
                    daccept * (normals[i] - dot * ray) / (distance * den) + exposure * 2 * delta / den
                )
            gradient = spatial @ jp[j] + (exposure * slope * ray) @ jt[j] + exposure * da * opening_jac[j]
            values[i] += weights[j] * length * value
            jac[i] += weights[j] * (length * gradient + value * length_jac)
    return values, jac


def render(positions, normals, path, particle, order):
    if particle == "electron":
        return np.zeros(len(positions)), np.zeros((len(positions), 12))
    m = material(particle)
    path = bounded_path(path, m["maximum_range_mm"])
    nodes, weights = roots_legendre(order)
    us = (nodes + 1) / 2
    points, axes, jp, jt = trajectory(path, us)
    remaining = 1000 * path[9] * (1 - path[10] * us)
    energy = m["energy_at"](remaining)
    de = m["energy_at"].derivative()(remaining)
    gamma = 1 + energy / m["mass"]
    beta = np.sqrt(1 - 1 / gamma**2)
    cosine = 1 / (m["index"] * beta)
    openings = np.arccos(np.clip(cosine, -1.0, 1.0))
    slope = np.zeros_like(us)
    above = cosine < 1.0
    slope[above] = de[above] / (
        m["index"] * m["mass"] * beta[above] ** 3 * gamma[above] ** 3 * np.sqrt(1 - cosine[above] ** 2)
    )
    opening_jac = np.zeros((order, 12))
    opening_jac[:, 9] = slope * 1000 * (1 - path[10] * us)
    opening_jac[:, 10] = -slope * 1000 * path[9] * us
    length = 1000 * path[9] * path[10]
    length_jac = np.zeros(12)
    length_jac[9:11] = 1000 * path[10], 1000 * path[9]
    saved = table()
    return integrate(
        positions,
        normals,
        points,
        axes,
        jp,
        jt,
        openings,
        opening_jac,
        weights / 2,
        length,
        length_jac,
        saved["opening_rad"],
        saved["polar_rad"],
        coefficients(),
    )
