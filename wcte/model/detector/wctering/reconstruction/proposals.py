"""Geometry proposals for comparing different ring counts.

merge combines two of three rings using their allocated weights. The fourth
ring search splits each of three axes in four planes, giving twelve starts.
Neither proposal makes a count decision; every candidate still needs a fit."""

import numpy as np


def merge(fit, i, j):
    pi = np.asarray(fit["weights"])
    w = pi[[i + 1, j + 1]]
    w = w / w.sum()
    vertices = np.asarray(fit["vertices"])
    axes = np.asarray(fit["axes"])
    axis = w @ axes[[i, j]]
    if np.linalg.norm(axis) < 1e-8:
        axis = axes[[i, j][int(np.argmax(w))]].copy()
    axis /= np.linalg.norm(axis)
    keep = [n for n in range(3) if n not in (i, j)]
    return dict(
        vertices=[w @ vertices[[i, j]], *vertices[keep]],
        axes=[axis, *axes[keep]],
        openings_deg=[
            w @ np.asarray(fit["openings_deg"])[[i, j]],
            *np.asarray(fit["openings_deg"])[keep],
        ],
        widths_deg=[
            w @ np.asarray(fit["widths_deg"])[[i, j]],
            *np.asarray(fit["widths_deg"])[keep],
        ],
        weights=np.r_[pi[0], pi[i + 1] + pi[j + 1], pi[np.asarray(keep) + 1]],
    )
