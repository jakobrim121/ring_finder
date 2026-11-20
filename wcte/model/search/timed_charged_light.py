"""Compare visible cores with calibrated unresolved short charged sources."""

import numpy as np

from material import material
from numba import njit
from source import angular_rule, material_panels, moment_table, rule, states, trajectory
from wctering.receiver.point_source import T, cap
from wctering.seeding.coherent_timing import C, N_GROUP


@njit(cache=True)
def projected_aperture(incidence):
    """Evaluate the same cap exposure as the charge integral."""
    result = np.empty_like(incidence)
    for i in range(incidence.shape[0]):
        for j in range(incidence.shape[1]):
            mu = incidence[i, j]
            result[i, j] = cap(mu)[0] if abs(mu) < T else max(mu, 0.0)
    return result


def photon_time(event, normals, paths, particle, order=4):
    """Integrate the arrival moment over emission, including soft tails."""
    water = material(particle)
    positions = event.pmt_positions_mm
    rate = np.zeros(len(positions))
    first_moment = rate.copy()
    node, weight = rule(order)
    spectrum = moment_table()
    angular = angular_rule()
    for path in paths:
        remaining = 1000.0 * path[9]
        length = remaining * path[10]
        cut = np.clip((remaining - water["margin_mm"]) / length, 0.0, 1.0)
        edges = material_panels(path, cut, water)
        start, span = edges[:-1], np.diff(edges)
        u = (start[:, None] + span[:, None] * node).ravel()
        weights = (span[:, None] * weight).ravel()
        points, axes, _, _ = trajectory(path, u)
        light = states(
            path,
            u,
            water["distance"],
            water["energy_at"].c,
            water["scatter_at"].c,
            *angular,
            False,
            water["mass"],
            water["index"],
            *spectrum,
            water["margin_mm"],
        )
        endpoints = np.r_[0.0, u]
        energy = water["energy_at"](np.clip(remaining - length * endpoints, 0.0, water["maximum_range_mm"]))
        inverse_beta = 1.0 / np.sqrt(1.0 - (1.0 + energy / water["mass"]) ** -2)
        flight = length * np.cumsum(np.diff(endpoints) * 0.5 * (inverse_beta[1:] + inverse_beta[:-1])) / C
        delta = positions[:, None] - points
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-8)
        ray = delta / distance[:, :, None]
        incidence = -np.einsum("ij,ikj->ik", normals, ray)
        aperture = projected_aperture(incidence)
        theta = np.arccos(np.clip(np.einsum("ikj,kj->ik", ray, axes), -1.0, 1.0))
        angular_weight = np.exp(-0.5 * (theta - light[:, 0]) ** 2 / light[:, 6])
        intensity = length * weights * light[:, 1] * angular_weight
        intensity *= aperture / (distance**2 + 10000.0)
        rate += intensity.sum(1)
        first_moment += np.sum(intensity * (flight + N_GROUP * distance / C), axis=1)
    center = np.mean([p[:3] for p in paths], axis=0)
    fallback = N_GROUP * np.linalg.norm(positions - 1000.0 * center, axis=1) / C
    return np.divide(first_moment, rate, out=fallback, where=rate > 1e-30)
