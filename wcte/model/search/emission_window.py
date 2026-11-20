"""Locate a finite source from charge and time along its inferred light axis."""

import numpy as np

from charge_starts import allowed_chord, weighted_quantile
from joint_prompt_fit import wavefront
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def coherent_interval(event, center, axis, opening, weight, *, clock=None):
    """Estimate the emitting interval independently of the fitted cone center.

    A PMT position gives an emission coordinate along a straight Cherenkov
    trajectory. A robust first-pulse time weight removes incompatible hits.
    For a known interaction clock, retain its timing reference while the
    emitting interval moves along that trajectory.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    direction = event.pmt_positions_mm - 1000.0 * center
    z = direction @ axis
    radius = np.linalg.norm(direction - z[:, None] * axis, axis=1)
    sine, cosine = np.sin(opening), np.cos(opening)
    where = z - radius * cosine / sine
    water = allowed_chord(center, axis)
    if water is None:
        return None
    clock_sample = (event.times_ns - np.median(event.times_ns)) - (
        N_PHASE * cosine * where + N_GROUP * radius / sine
    ) / C
    if not np.any(weight > 0.0):
        return None
    if clock is None:
        clock = weighted_quantile(clock_sample, weight, [0.5])[0]
    timing = np.exp(-0.5 * ((clock_sample - clock) / 2.0) ** 2)
    q = weight * timing * (where >= 1000.0 * (water[0] - 0.12)) * (where <= 1000.0 * (water[1] + 0.12))
    if q.sum() < 12.0:
        return None
    lo, hi = weighted_quantile(where / 1000.0, q, [0.12, 0.88])
    lo, hi = max(lo - 0.025, water[0]), min(hi + 0.025, water[1])
    if hi - lo < 0.025:
        return None
    p = np.r_[center + 0.5 * (lo + hi) * axis, axis, hi - lo, opening, np.log(np.deg2rad(2.4))]
    arrival = wavefront(event.pmt_positions_mm, p)[0]
    # Shifting the cone center also shifts its clock by the particle flight.
    offset = clock + 1000.0 * 0.5 * (lo + hi) * N_PHASE * cosine / C
    return p, arrival + offset, q
