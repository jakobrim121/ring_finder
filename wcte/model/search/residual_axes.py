"""Find a localized secondary cone on the residual of the full optical model."""

import numpy as np

from wctering.seeding.coherent_timing import C, N_GROUP


def timed_residual_vertices(event, intensity, *, limit=4):
    """Locate point-like arrivals in the charge left by an optical mixture."""
    from scipy.ndimage import gaussian_filter1d
    from wctering.likelihood.geometry import NORMALS, UPPER

    xyz = event.pmt_positions_mm
    relative = event.times_ns - np.median(event.times_ns)
    weight = np.minimum(event.charges_pe, 2.0) / np.maximum(intensity, 0.6)
    edges = np.arange(-32.0, 10.5, 0.5)
    grid = (
        np.array(
            np.meshgrid(
                np.arange(-1.4, 1.41, 0.35), np.arange(-0.70, 1.76, 0.35), np.arange(-1.4, 1.41, 0.35)
            )
        )
        .reshape(3, -1)
        .T
    )
    grid = grid[(UPPER - grid @ NORMALS.T).min(1) > 0.015]
    peaks = []
    for point in grid:
        distance = np.linalg.norm(xyz - 1000.0 * point, axis=1)
        corrected = relative - N_GROUP * distance / C
        bins = np.searchsorted(edges, corrected) - 1
        valid = (bins >= 0) & (bins < len(edges) - 1)
        density = np.bincount(bins[valid], weights=weight[valid], minlength=len(edges) - 1)
        # Subtract a broad time component so a dense diffuse background
        # does not favor a nearby detector surface by itself.
        sharp = gaussian_filter1d(density, 2.0) - gaussian_filter1d(density, 6.0)
        peaks.append((float(sharp.max()), point))
    chosen = []
    for strength, point in sorted(peaks, key=lambda row: row[0], reverse=True):
        if strength < 2.0:
            break
        if all(np.linalg.norm(point - old) > 0.32 for old in chosen):
            chosen.append(point)
        if len(chosen) == limit:
            break
    return chosen
