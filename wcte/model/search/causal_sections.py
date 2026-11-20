"""Check that two proposed charged legs share one interaction clock."""

import numpy as np

from charge_starts import weighted_quantile
from coherent_partition import prediction_time


def check(event, model, state, fractions):
    """Profile independent clocks only to measure disagreement with a join."""
    n = len(model.particles)
    if n < 2 or fractions.shape[1] != n + 1:
        return None
    q = np.minimum(event.charges_pe, 2.0)
    t = event.times_ns - np.median(event.times_ns)
    arrivals, illumination = prediction_time(event, state["parameters"], model.particles, event.charges_pe)
    r = t[:, None] - arrivals
    labels = fractions.argmax(1)
    estimates = []
    for j in range(n):
        # The earliest time is most informative when this section owns the
        # measured pulse. Exclude weak overlap with another first arrival.
        mask = (labels == j + 1) & (illumination[:, j] > 1e-6) & (fractions[:, j + 1] > 0.55)
        if np.count_nonzero(mask) < 15:
            return None
        weight = q[mask] * fractions[mask, j + 1]
        local = r[mask, j]
        lo, center, hi = weighted_quantile(local, weight, [0.20, 0.50, 0.80])
        estimates.append(
            dict(clock_ns=float(center), width_ns=float(0.5 * (hi - lo)), pmts=int(np.count_nonzero(mask)))
        )
    clocks = np.array([g["clock_ns"] for g in estimates])
    widths = np.array([g["width_ns"] for g in estimates])
    # The tolerance allows the calibration spread and the use of the
    # optical rather than electronics waveform's onset. Flight time
    # between serial sections has already been integrated by the model.
    tolerance = 1.25 + 0.25 * (widths[1:] + widths[:-1])
    gaps = abs(np.diff(clocks))
    return dict(
        compatible=bool(np.all(gaps <= tolerance)),
        section_times=estimates,
        adjacent_clock_gaps_ns=gaps.tolist(),
        adjacent_clock_tolerance_ns=tolerance.tolist(),
    )
