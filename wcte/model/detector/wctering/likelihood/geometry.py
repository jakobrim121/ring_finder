"""Detector prism constraints, diffuse background, and finite-track initialization.

The flat receiver expression here is used only to choose a finite starting
length. The fitted likelihood uses the spherical receiver response."""

from pathlib import Path
import numpy as np
from wctering.logmath import logsumexp

ROOT = Path(__file__).resolve().parent.parent
from wctering.seeding.optical_seed_fit import (
    active_geometry,
    axis_angles,
    axis_derivatives,
)
from wctering.seeding.geometry import (
    WCTE_PRISM_FACE_NORMALS_XZ,
    WCTE_PRISM_APOTHEM_MM,
    WCTE_PRISM_Y_MIN_MM,
    WCTE_PRISM_Y_MAX_MM,
)

NORMALS = np.zeros((18, 3))
NORMALS[:16, 0] = WCTE_PRISM_FACE_NORMALS_XZ[:, 0]
NORMALS[:16, 2] = WCTE_PRISM_FACE_NORMALS_XZ[:, 1]
NORMALS[16, 1] = 1.0
NORMALS[17, 1] = -1.0
UPPER = np.r_[np.repeat(WCTE_PRISM_APOTHEM_MM, 16), WCTE_PRISM_Y_MAX_MM, -WCTE_PRISM_Y_MIN_MM] / 1000
BOX = [
    (-2.0, 2.0),
    (WCTE_PRISM_Y_MIN_MM / 1000, WCTE_PRISM_Y_MAX_MM / 1000),
    (-2.0, 2.0),
]


def point_source(pos, normals, x):
    vertex = x[:3] * 1000
    theta, phi, opening, logwidth = x[3:]
    width = np.exp(logwidth)
    axis, da = axis_derivatives(theta, phi)
    delta = pos - vertex
    r = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    u = delta / r[:, None]
    dot = np.sum(normals * u, axis=1)
    inc = np.clip(-dot, 0, 1)
    raw = inc / (r * r + 100**2)
    exposure = np.maximum(raw, 1e-30)
    de = (
        (normals - dot[:, None] * u) / np.maximum(r * inc, 1e-30)[:, None]
        + 2 * delta / (r * r + 100**2)[:, None]
    ) * 1000
    de[raw <= 1e-30] = 0.0
    cosine = np.clip(u @ axis, -1 + 1e-10, 1 - 1e-10)
    sine = np.sqrt(1 - cosine * cosine)
    residual = np.arccos(cosine) - opening
    log = np.log(exposure) - 0.5 * (residual / width) ** 2
    s = np.exp(log - logsumexp(log))
    dv = de - residual[:, None] / width**2 * (axis - cosine[:, None] * u) / (r * sine)[:, None] * 1000
    daxis = residual[:, None] / (width**2 * sine)[:, None] * (u @ da.T)
    jac = np.column_stack((dv, daxis, residual / width**2, (residual / width) ** 2))
    jac -= s @ jac
    return s, s[:, None] * jac


def common_background(vertex):
    ids, pos, normals = active_geometry()
    delta = pos - vertex
    r = np.linalg.norm(delta, axis=1)
    u = delta / r[:, None]
    e = np.maximum(np.clip(-np.sum(normals * u, axis=1), 0, 1) / (r * r + 100**2), 1e-30)
    return 0.5 * e / e.sum() + 0.5 / len(ids)


def finite_start(vertex, axis, opening, width):
    """Navigation only: account for point width with a tentative finite extent."""
    _, pos, normals = active_geometry()
    v = np.asarray(vertex) / 1000.0
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x = np.r_[v, axis_angles(axis), np.deg2rad(opening), np.log(np.deg2rad(width))]
    shape, _ = point_source(pos, normals, x)
    delta = pos - np.asarray(vertex)
    par = delta @ axis
    rho = np.sqrt(np.maximum(np.sum(delta * delta, axis=1) - par * par, 0.0))
    intrinsic = min(width, 8.0)
    variance = max(np.deg2rad(width) ** 2 - np.deg2rad(intrinsic) ** 2, 0.0)
    length = (
        np.clip(
            np.sqrt(12 * variance) * (shape @ rho) / max(np.sin(np.deg2rad(opening)) ** 2, 0.01),
            100.0,
            3000.0,
        )
        / 1000.0
    )
    margin = UPPER - NORMALS @ v
    assert margin.min() >= -1e-7

    def exit_distance(direction):
        den = NORMALS @ direction
        active = den > 1e-12
        return max(0.0, float(np.min(margin[active] / den[active])))

    back = min(0.5 * length, 0.95 * exit_distance(-axis))
    forward = min(0.5 * length, 0.95 * exit_distance(axis))
    if back + forward < 0.001:
        raise ValueError("Insufficient water chord for finite seed")
    return np.r_[
        v - back * axis,
        v + forward * axis,
        np.deg2rad(opening),
        np.log(np.deg2rad(intrinsic)),
    ]
