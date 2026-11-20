"""Refine short-cone vertices in the time residual of the fitted light."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from wctering.seeding.coherent_timing import C, N_GROUP
from wctering.seeding.geometry import wcte_contains_vertex


def clock_modes(residual, weight):
    """Find one or two distinct relative times in the excess charge."""
    lower = float(np.floor(residual.min()))
    cell = np.maximum(np.floor((residual - lower) * 2.0).astype(int), 0)
    density = gaussian_filter1d(np.bincount(cell, weights=weight), 2.0)
    peak = int(np.argmax(density))
    yield lower + 0.5 * (peak + 0.5)
    far = abs(np.arange(len(density)) - peak) > 5
    if far.any():
        alternate = int(np.argmax(density * far))
        if density[alternate] >= 0.65 * density[peak]:
            yield lower + 0.5 * (alternate + 0.5)


def refine(points, positions, times, weight, maximum=5):
    """Keep distinct peaks of an event-relative spherical wavefront."""
    travel = 1000.0 * N_GROUP / C
    spatial_prior = 1.0 / 0.23**2
    candidates = []
    for seed in points:
        delta = positions - seed
        seed_delay = times - travel * np.linalg.norm(delta, axis=1)
        for clock in clock_modes(seed_delay, weight):
            value = np.r_[seed, clock]
            for _ in range(6):
                delta = positions - value[:3]
                distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
                residual = times - travel * distance - value[3]
                # Positive residual charge seeds a compact source, while
                # the flat component leaves unrelated pulses as outliers.
                kernel = np.exp(-0.5 * (residual / 1.45) ** 2)
                probability = weight * kernel / (kernel + 0.18)
                jacobian = np.column_stack([travel * delta / distance[:, None], -np.ones(len(distance))])
                jacobian *= np.sqrt(probability[:, None])
                curvature = jacobian.T @ jacobian
                curvature.flat[::5] += [spatial_prior, spatial_prior, spatial_prior, 0.1]
                slope = jacobian.T @ (np.sqrt(probability) * residual)
                slope[:3] += spatial_prior * (value[:3] - seed)
                step = np.linalg.solve(curvature, -slope)
                step *= min(1.0, 0.12 / max(np.linalg.norm(step[:3]), 1e-20))
                value += step
                if np.linalg.norm(step) < 0.001:
                    break
            if not wcte_contains_vertex(1000.0 * value[:3]):
                continue
            residual = times - travel * np.linalg.norm(positions - value[:3], axis=1) - value[3]
            score = float(weight @ np.exp(-0.5 * (residual / 1.45) ** 2))
            if score >= 12.0:
                candidates.append((score, value[:3].copy()))
    selected = []
    for _, point in sorted(candidates, key=lambda x: x[0], reverse=True):
        if all(np.linalg.norm(point - old) > 0.12 for old in selected):
            selected.append(point)
        if len(selected) == maximum:
            break
    return selected
