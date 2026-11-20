"""Screened sharing for an already fixed one-ring count.

Different point and finite explanations can produce different background
assignments. Require their hard partitions to agree closely before taking this
shortcut; otherwise run the entire original H1 search. This detector-only
agreement check is a screening rule, NOT a proof of the truth-F1 requirement.
The independent per-event evaluator remains the acceptance criterion.
"""

import traceback
import numpy as np
from wctering.reconstruction.one_ring_search import Search
from wctering.reconstruction.one_ring_fit import best_fit, unresolved
from wctering.reconstruction.one_ring_seeds import single_ring_starts
from wctering.results import Reconstruction


def try_fixed_one(event, diagnostics, bank, labels):
    search = Search(event, bank)
    try:
        fits = [search.fit(1, kind, True) for kind in ("point", "finite")]
        for geometry in single_ring_starts(event, diagnostics, labels, search.background):
            fits.append(search.fit(1, "point", True, geometry))
        point = best_fit([fit for fit in fits if fit["kind"] == "point"], "ll")
        if point is not None and point["ll"] > fits[0].get("ll", -np.inf) + 1e-7:
            fits.append(search.fit(1, "finite", True, point["geometry"]))
        selected = best_fit(fits, "ll")
        note = dict(attempts=len(fits), accepted=False)
        diagnostics["_quick_fixed_one"] = note
        if selected is None or point is None or unresolved(fits, "ll"):
            note["reason"] = "Incomplete fixed-count comparison"
            return None
        finite = best_fit([fit for fit in fits if fit["kind"] == "finite"], "ll")
        if finite is None:
            note["reason"] = "Finite family did not qualify"
            return None
        # Charge capping is an observable weighting convention. Truth is never
        # read here. The small disagreement threshold conservatively sends
        # model-dependent background assignments back to the original search.
        weights = np.minimum(event.charges_pe, 2.0)
        changed = point["posterior"].argmax(0) != finite["posterior"].argmax(0)
        disagreement = float(weights[changed].sum() / weights.sum())
        note["point_finite_hard_disagreement"] = disagreement
        if disagreement > 0.025:
            note["reason"] = "Point and finite background assignments disagree"
            return None
        # The best point and finite fits need not be independent: a finite
        # track can collapse onto a new point optimum. Also compare with both
        # original bank starts, before the extra timing starts were introduced.
        # A materially different basin is handled by the full V101 search.
        original_disagreements = []
        for original in fits[:2]:
            if not original["qualified"]:
                note["reason"] = "An original bank start did not qualify"
                return None
            changed = selected["posterior"].argmax(0) != original["posterior"].argmax(0)
            original_disagreements.append(float(weights[changed].sum() / weights.sum()))
        note["original_bank_hard_disagreements"] = original_disagreements
        if max(original_disagreements) > 0.025:
            note["reason"] = "Timing alternatives changed the original bank partition"
            return None
        note["accepted"] = True
        index = next(i for i, fit in enumerate(search.fits) if fit is selected)
        detail = dict(
            initial_k=1,
            selected_count=1,
            fallback=False,
            quick_fixed_one=note,
            selected_fit_index=index,
            count_fit_index=index,
            fits=[{key: value for key, value in fit.items() if key != "posterior"} for fit in search.fits],
        )
        return Reconstruction(1, event, selected["posterior"].T, detail)
    except Exception:
        diagnostics["_quick_fixed_one"] = dict(accepted=False, traceback=traceback.format_exc())
        return None
