"""Check whether apparent rings are duplicate views of one charged track."""

import numpy as np

from charge_starts import weighted_quantile
from coherent_partition import prediction_time
from wcte.counting import track_counting


def closure(event, model, state, fractions, incumbent):
    """Require one optical path and one clock to explain every bright region.

    The incumbent identifies the regions to check. A real additional ring
    can survive the test through its own independent light even if most
    other PMTs agree with the single-track hypothesis.
    """
    charge = event.charges_pe
    capped = np.minimum(charge, 2.0)
    direction_time, illumination = prediction_time(event, state["parameters"], model.particles, charge)
    residual = (event.times_ns - np.median(event.times_ns))[:, None] - direction_time
    core = state["posterior"][1:].T
    direct = core * (illumination > 1e-6)
    weights = direct * capped[:, None]
    if not weights.any():
        return None
    clock = weighted_quantile(residual.ravel(), weights.ravel(), [0.5])[0]
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(charge, 0.25)) / 2.355
    arrival = np.exp(-0.5 * ((residual - clock) / (2.0 * sigma[:, None])) ** 2)
    coherent = np.sum(direct * arrival, axis=1)
    labelled = incumbent.argmax(1)
    groups = []
    for k in range(1, incumbent.shape[1]):
        owned = labelled == k
        total = capped[owned].sum()
        if not total:
            return None
        groups.append(
            dict(
                optical=float(capped[owned] @ fractions[owned, 1] / total),
                prompt=float(capped[owned] @ coherent[owned] / total),
                independent_pmts=int(np.count_nonzero(owned & (fractions[:, 1] < 0.2))),
            )
        )
    return groups


def test(event, alternatives, reference, margin):
    """Allow a demotion only if the fitted one-track model closes the split."""
    incumbent = max((r for r in alternatives if r[2]["count"] > 1), key=lambda r: r[0])
    optical = incumbent[1]
    for score, fractions, record, model, state in reference:
        # A one-track candidate still has to compete with the brightest
        # qualified multi-track explanation in the calibrated charge fit.
        if score <= incumbent[0] + margin:
            continue
        groups = closure(event, model, state, fractions, optical)
        if groups is None:
            continue
        supported = all(
            g["optical"] >= 0.70 and g["prompt"] >= 0.60 and g["independent_pmts"] < track_counting.get().scaled_support(18) for g in groups
        )
        record.update(single_track_closure=groups, closure_supported=supported)
