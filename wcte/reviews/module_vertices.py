"""Locate detached emission using the angular response within each mPMT."""

from functools import lru_cache

import numpy as np
from numba import njit


@lru_cache(maxsize=1)
def grid_response():
    from timed_charged_light import projected_aperture
    from wctering.likelihood.geometry import active_geometry
    from wctering.seeding.coherent_timing import C, N_GROUP
    from wctering.seeding.geometry import wcte_contains_vertex

    ids, pos, normal = active_geometry()
    coordinates = np.linspace(-1210.0, 1210.0, 11)
    xyz = np.stack(np.meshgrid(coordinates, coordinates, coordinates), -1).reshape(-1, 3)
    xyz = np.array([p for p in xyz if wcte_contains_vertex(p)])
    distance_vector = pos[None] - xyz[:, None]
    distance = np.maximum(np.linalg.norm(distance_vector, axis=-1), 1.0)
    cosine = -np.einsum("jrc,rc->jr", distance_vector, normal) / distance
    aperture = (projected_aperture(cosine) + 0.008) / (distance**2 + 10000.0)
    modules, groups = np.unique(ids // 100, return_inverse=True)
    result = np.zeros_like(aperture)
    for group in range(len(modules)):
        include = groups == group
        total = aperture[:, include].sum(1)
        result[:, include] = (aperture[:, include] / total[:, None]) * include.sum()
    return ids, xyz, N_GROUP * distance / C, np.log(np.maximum(result, 1e-8))


@njit(cache=True)
def pulse_modes(time, weights, low, interval, count):
    """Accumulate the observed clock in each birth voxel before smoothing."""
    score = np.zeros(len(time))
    offset = np.zeros_like(score)
    width = 1.7 / interval
    kernel = np.exp(-0.5 * (np.arange(count) / width) ** 2)
    for i in range(len(time)):
        binning = np.zeros(count)
        row = weights[0] if len(weights) == 1 else weights[i]
        for j in range(time.shape[1]):
            x = (time[i, j] - low) / interval
            index = int(np.floor(x))
            fraction = x - index
            if 0 <= index < count:
                binning[index] += row[j] * (1.0 - fraction)
            if 0 <= index + 1 < count:
                binning[index + 1] += row[j] * fraction
        for k in range(count):
            value = 0.0
            for j in range(count):
                value += kernel[abs(k - j)] * binning[j]
            if value > score[i]:
                score[i] = value
                offset[i] = low + interval * k
    return score, offset


def birth_points(event, model, base, *, directional):
    ids, vertices, times, incidence = grid_response()
    if not np.array_equal(ids, model.active):
        return []
    ro = model.ix
    pos = event.pmt_positions_mm
    measured = event.times_ns - np.median(event.times_ns)
    charge = np.minimum(event.charges_pe, 2.0)
    _, slope = model.obs.score(base, ro)
    excess = np.clip(-slope[ro] * base[ro], -0.70, 2.5)
    taken = excess > 0.20
    if np.count_nonzero(taken) < 20:
        return []
    times = measured - times[:, ro]
    weight = charge * np.maximum(excess, 0.0)
    all_weights = weight[None] * np.exp(0.36 * incidence[:, ro]) if directional else weight[None]
    lo, hi = np.quantile(times[:, taken], [0.10, 0.85])
    step = 0.4
    best, offset = pulse_modes(times, all_weights, lo - 3.4, step, max(12, int((hi - lo + 6.8) / step) + 1))
    chosen = []
    for index in np.argsort(best)[::-1]:
        p = vertices[index]
        if any(np.linalg.norm(p - previous) < 330.0 for previous, *_ in chosen):
            continue
        chosen.append((p, float(offset[index]), pos - p, measured, charge, excess))
        if len(chosen) == 4:
            break
    return chosen
