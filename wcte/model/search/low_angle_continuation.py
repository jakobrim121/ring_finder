"""Optical coordinates and high-order checks for connected finite paths."""

import numpy as np

LINEAGE_COSINE = np.cos(np.deg2rad(15.0))


def coordinates(paths, photons):
    return dict(
        vertices=1000.0 * paths[:, :3],
        axes=paths[:, 3:6].copy(),
        openings_deg=np.rad2deg(paths[:, 7]),
        widths_deg=np.rad2deg(np.exp(paths[:, 8])),
        lengths_mm=1000.0 * paths[:, 6],
        weights=photons / photons.sum(),
    )


def verified(fit, x):
    """Re-evaluate both hypotheses with the same higher-order light integral."""
    values = []
    for fit.order in (16, 32):
        fit.last = None
        values.append(fit.evaluate(x))
    return values[-1] if abs(values[0][0] - values[1][0]) < 0.01 else None
