"""Screened primary sharing when the two-ring count is already screened.

Fit every original point start and the original finite bank start. Only if all
these independent starts qualify and agree closely can the remaining finite
starts be omitted. This is an empirical screen, not a bound on truth F1; full
per-event validation is required. It never reads truth or event identities.
"""

import numpy as np
from wctering.optimization.fit import best_fit


def primary_bank(search, starts):
    from wctering.reconstruction.workflow import charge_bank

    points = []
    for name, geometry in starts:
        fit = search.fit(2, "point", geometry)
        fit["origin"] = name
        points.append(fit)
    name, geometry = starts[0]
    finite = search.fit(2, "finite", geometry)
    finite["origin"] = name
    # Preserve the original point-before-finite order for equal likelihoods.
    proposals = [points[0], finite, *points[1:]]
    note = dict(accepted=False)
    search.quick_two_charge = note
    if not all(fit.get("qualified") for fit in proposals):
        note["reason"] = "An original point start or finite bank did not qualify"
        return charge_bank(search, starts)
    selected = best_fit(proposals, "ll")
    labels = selected["posterior"].argmax(0)
    weight = np.minimum(search.event.charges_pe, 2.0)
    discrepancies = []
    # Inferior point basins are still fitted and must qualify, but agreement
    # is required only between the best point basin and original finite bank.
    # This is an empirical shortcut; it is accepted only after per-event tests.
    for fit in (best_fit(points, "ll"), finite):
        other = fit["posterior"].argmax(0)
        exchanged = np.asarray([0, 2, 1])[other]
        discrepancy = min(
            float(weight[labels != candidate].sum() / weight.sum()) for candidate in (other, exchanged)
        )
        discrepancies.append(discrepancy)
    note["hard_partition_disagreements"] = discrepancies
    if max(discrepancies) > 0.01:
        note["reason"] = "The best point fit and finite bank disagree"
        return charge_bank(search, starts)
    note["accepted"] = True
    return proposals
