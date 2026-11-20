"""Draw the WCTE mPMT projection used by the LicketyFit event displays."""

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _projection():
    path = Path(__file__).with_name("display_data") / "mpmt_projection.csv"
    layout = np.loadtxt(path, delimiter=",", skiprows=1, dtype=int)
    slots, x, y, rotations = layout.T
    centers = np.column_stack((x, y))
    channel = np.tile(np.arange(19), len(slots))
    # One central PMT, six on the middle ring, and twelve on the outer ring.
    middle = (channel >= 1) & (channel <= 6)
    outer = channel > 6
    angle = (
        middle * (7 - channel) * (2 * np.pi / 6)
        + outer * (19 - channel) * (2 * np.pi / 12)
        - np.deg2rad(np.repeat(rotations, 19))
    )
    radius = 0.2 * (channel > 0) + 0.2 * outer
    offsets = np.column_stack((radius * np.sin(angle), radius * np.cos(angle)))
    pmts = np.repeat(slots, 19) * 100 + channel
    return slots, centers, pmts, np.repeat(centers, 19, axis=0) + offsets


def plot_event(pmt_ids, values, *, title=None, ring_count=None):
    """Draw one PE value or one dominant component label per canonical WCTE PMT."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Circle

    ids = np.asarray(pmt_ids)
    values = np.asarray(values, dtype=float)
    if ids.ndim != 1 or ids.shape != values.shape:
        raise ValueError("PMT IDs and display values must be aligned one-dimensional arrays")
    slots, centers, pmts, coordinates = _projection()
    rows = np.searchsorted(pmts, ids)
    if (rows == len(pmts)).any() or not np.array_equal(pmts[rows], ids):
        raise ValueError("Use WCTE PMT IDs as 100 * slot + zero-based position")

    cmap = plt.get_cmap("plasma")
    label = "Charge [PE]"
    ticks = None
    if ring_count is None:
        norm = Normalize(0, max(1.0, float(values.max(initial=0.0))))
    else:
        ticks = np.arange(ring_count + 1)
        names = ["Background"] + [f"Ring {n}" for n in ticks[1:]]
        ring_colors = cmap(np.linspace(0.18, 0.97, ring_count))
        cmap = ListedColormap(["#777d8a", *ring_colors])
        norm = BoundaryNorm(np.arange(ring_count + 2) - 0.5, cmap.N)
        label = "Dominant fitted component"

    with plt.style.context("dark_background"):
        figure, axis = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
        outlines = [Circle(center, 0.48) for center in centers]
        axis.add_collection(PatchCollection(outlines, facecolor="none", edgecolor="0.4", linewidth=0.55))
        for slot, (x, y) in zip(slots, centers, strict=True):
            axis.text(x - 0.5, y - 0.5, str(slot), fontsize=5.5, ha="center", va="center", color="0.8")
        hit_positions = coordinates[rows]
        hits = axis.scatter(*hit_positions.T, c=values, s=10, cmap=cmap, norm=norm, linewidths=0)
        colorbar = figure.colorbar(hits, ax=axis, pad=0.025, label=label)
        if ticks is not None:
            colorbar.set_ticks(ticks, labels=names)
        x_min, y_min = centers.min(axis=0) - 0.85
        x_max, y_max = centers.max(axis=0) + 0.7
        axis.set(xlim=(x_min, x_max), ylim=(y_min, y_max), aspect="equal", title=title)
    return figure, axis
