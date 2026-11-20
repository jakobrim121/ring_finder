"""Small detector-only search for the primary-only adequacy stage."""

import numpy as np


from .primary import Aperture, PrimaryChart


def seeds(model):
    """Scan visible range, retaining every massive-particle hypothesis.

    The aperture normal is an initialization; both direction coordinates
    subsequently move. No beam energy or generator species is supplied.
    """
    port = Aperture()
    lower, upper = model.range_bounds
    ranges = np.unique(
        np.r_[np.geomspace(max(lower * 1.01, 0.025), upper * 0.999, 14), min(2.0, upper * 0.999)]
    )
    ranked = []
    for ran in ranges:
        start = np.r_[
            0.0,
            0.0,
            port.z,
            0.0,
            0.0,
            1.0,
            np.zeros(3),
            ran,
            1.0,
            np.log(np.deg2rad(2.0)),
            np.log(model.response),
            np.log(max(0.08 * model.total, 1.0)),
        ]
        chart = PrimaryChart(model, start)
        x = chart.full(chart.initial)[0]
        ranked.append((model(x, True)["ll"], x))
    return sorted(ranked, key=lambda row: -row[0])
