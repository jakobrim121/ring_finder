"""Use local track timing to refine charge ownership without changing counts."""

import numpy as np

from chain_sources import paths
from source import trajectory
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def prediction_time(event, parameters, particles, amplitude):
    """Approximate the first direct wavefront from each emitting section."""
    from material import material

    geometry = [p for p, _ in paths(parameters[:-2])]
    if len(geometry) != len(particles):
        raise ValueError("Expected a serial charged-track hypothesis")
    positions = event.pmt_positions_mm
    node = np.linspace(0.0, 1.0, 17)
    arrival = []
    support = []
    offset = 0.0
    for p, mass in zip(geometry, particles):
        m = material(mass)
        points, directions, _, _ = trajectory(p, node)
        range_mm = 1000 * p[9]
        section = range_mm * p[10]
        energy = m["energy_at"](np.clip(range_mm - section * node, 0.0, m["maximum_range_mm"]))
        beta = np.sqrt(np.maximum(1 - (1 + energy / m["mass"]) ** -2, 1e-8))
        velocity = np.r_[0.0, np.cumsum(0.5 * (1 / beta[1:] + 1 / beta[:-1]) * np.diff(node) * section / C)]
        delta = positions[:, None, :] - points
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-8)
        angle = np.arccos(np.clip(np.sum(delta * directions, axis=2) / distance, -1.0, 1.0))
        opening = np.arccos(np.clip(1 / (N_PHASE * beta), 0.0, 1.0))
        width = max(np.exp(p[11]), np.deg2rad(1.0))
        emissive = range_mm - section * node >= m["margin_mm"]
        weight = (
            np.exp(-0.5 * ((angle - opening) / width) ** 2)
            * np.maximum(1 - (N_PHASE * beta) ** -2, 0.0)
            * emissive
        )
        time = offset + velocity + N_GROUP * distance / C
        mean = np.sum(time * weight, axis=1) / np.maximum(weight.sum(1), 1e-30)
        spread = np.sqrt(
            np.sum(weight * (time - mean[:, None]) ** 2, axis=1) / np.maximum(weight.sum(1), 1e-30)
        )
        arrival.append(mean - np.minimum(spread, 2.0) * np.clip(np.log1p(amplitude), 0.0, 1.5))
        support.append(weight.sum(1))
        offset += velocity[-1]
    return np.column_stack(arrival), np.column_stack(support)
