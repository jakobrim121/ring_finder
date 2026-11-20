"""Scalar evaluation of C's parent-coupled light integral and Jacobian.

The spline, receiver response, and longitudinal integration rule are the
original ones. Cache the opening-angle interval once per emission point.
Scalar geometry and Horner evaluation avoid arrays and tiny BLAS calls in
the PMT loop. Do not change how the optimizer qualifies a candidate here.
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit

from lookup import cap


@njit(cache=True, inline="always")
def cubic_and_derivative(coefficients, offset):
    a, b, c, d = coefficients
    value = ((a * offset + b) * offset + c) * offset + d
    derivative = (3 * a * offset + 2 * b) * offset + c
    return value, derivative


@njit(cache=True, inline="always")
def angular(opening_offset, polar, polar_grid, coefficients, clamped):
    # The table uses a uniform polar grid. Correct the estimated cell against
    # the actual endpoints, including exact knots and rounding at the limits.
    cell = int(polar * (len(polar_grid) - 1) / polar_grid[-1])
    cell = min(max(cell, 0), len(polar_grid) - 2)
    while cell > 0 and polar <= polar_grid[cell]:
        cell -= 1
    while cell < len(polar_grid) - 2 and polar > polar_grid[cell + 1]:
        cell += 1
    offset = polar - polar_grid[cell]
    p0, d0 = cubic_and_derivative(coefficients[cell, 0], offset)
    p1, d1 = cubic_and_derivative(coefficients[cell, 1], offset)
    p2, d2 = cubic_and_derivative(coefficients[cell, 2], offset)
    p3, d3 = cubic_and_derivative(coefficients[cell, 3], offset)
    x = opening_offset
    value = ((p0 * x + p1) * x + p2) * x + p3
    opening_derivative = 0.0 if clamped else (3 * p0 * x + 2 * p1) * x + p2
    polar_derivative = ((d0 * x + d1) * x + d2) * x + d3
    return value, opening_derivative, polar_derivative


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
    coefficients,
):
    values = np.zeros(len(positions))
    jacobian = np.zeros((len(positions), 12))
    for j, weight in enumerate(weights):
        opening = openings[j]
        cell = min(max(np.searchsorted(opening_grid, opening) - 1, 0), len(opening_grid) - 2)
        bounded = min(max(opening, opening_grid[0]), opening_grid[-1])
        clamped = opening <= opening_grid[0] or opening >= opening_grid[-1]
        spline = coefficients[cell]
        px, py, pz = points[j]
        ax, ay, az = axes[j]
        for i in range(len(positions)):
            dx = positions[i, 0] - px
            dy = positions[i, 1] - py
            dz = positions[i, 2] - pz
            distance = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)
            rx, ry, rz = dx / distance, dy / distance, dz / distance
            cosine = min(max(rx * ax + ry * ay + rz * az, -1.0), 1.0)
            density, da, dp = angular(
                bounded - opening_grid[cell], math.acos(cosine), polar_grid, spline, clamped
            )
            nx, ny, nz = normals[i]
            dot = nx * rx + ny * ry + nz * rz
            accept, daccept = cap(-dot)
            denominator = distance * distance + 10000.0
            exposure = max(accept / denominator, 1e-30)
            value = exposure * density
            slope = -dp / math.sqrt(1 - cosine * cosine) if abs(cosine) < 1 - 1e-12 else 0.0
            gx = -exposure * slope * (ax - cosine * rx) / distance
            gy = -exposure * slope * (ay - cosine * ry) / distance
            gz = -exposure * slope * (az - cosine * rz) / distance
            if accept / denominator > 1e-30:
                gx += density * (
                    daccept * (nx - dot * rx) / (distance * denominator) + exposure * 2 * dx / denominator
                )
                gy += density * (
                    daccept * (ny - dot * ry) / (distance * denominator) + exposure * 2 * dy / denominator
                )
                gz += density * (
                    daccept * (nz - dot * rz) / (distance * denominator) + exposure * 2 * dz / denominator
                )
            sx, sy, sz = exposure * slope * rx, exposure * slope * ry, exposure * slope * rz
            values[i] += weight * length * value
            for k in range(12):
                spatial = gx * jp[j, 0, k] + gy * jp[j, 1, k] + gz * jp[j, 2, k]
                direction = sx * jt[j, 0, k] + sy * jt[j, 1, k] + sz * jt[j, 2, k]
                gradient = spatial + direction + exposure * da * opening_jac[j, k]
                jacobian[i, k] += weight * (length * gradient + value * length_jac[k])
    return values, jacobian


def install(light):
    """Use the faster arithmetic in an isolated fitter process."""
    light.integrate = integrate
    light.roots_legendre = lru_cache(maxsize=16)(light.roots_legendre)
