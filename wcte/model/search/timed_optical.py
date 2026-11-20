"""Compare detector-timed optical modes with a calibrated charge partition."""

import numpy as np

from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def arrival(event, geometry, *, positions=None):
    """Approximate prompt arrival from each complete optical source.

    Finite sections project each receiver to its emitting interval. Point
    cones use their fitted vertex.
    """
    means = []
    positions = event.pmt_positions_mm if positions is None else positions
    lengths = geometry.get("lengths_mm", [None] * len(geometry["vertices"]))
    for vertex, axis, angle, length in zip(
        geometry["vertices"], geometry["axes"], geometry["openings_deg"], lengths
    ):
        direction = np.asarray(axis) / np.linalg.norm(axis)
        theta = np.deg2rad(np.clip(angle, 10.0, 42.3))
        delta = positions - vertex
        if length is not None and length > 0:
            z = delta @ direction
            rho = np.linalg.norm(delta - z[:, None] * direction, axis=1)
            emission = np.clip(z - rho / np.tan(theta), -0.5 * length, 0.5 * length)
        else:
            emission = np.zeros(len(positions))
        flight = N_PHASE * np.cos(theta) * emission
        flight += N_GROUP * np.linalg.norm(delta - emission[:, None] * direction, axis=1)
        means.append(flight / C)
    return np.column_stack(means)
