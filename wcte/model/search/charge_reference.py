"""Retain V103's fixed-count optical geometry before the physics comparison."""

from wctering.optimization.fit import best_fit
from wctering.reconstruction.candidates import FitSearch
from wctering.reconstruction.one_ring_partition import try_fixed_one
from wctering.reconstruction.proposals import merge
from wctering.reconstruction.two_ring_partition import primary_bank
from wctering.reconstruction.workflow import charge_bank, foreground
from wctering.results import Reconstruction, seed_partition


def reference(event, diagnostic, bank):
    """Avoid exhaustive occupancy rescues while keeping their main charge fit."""
    count, labels, fallback = seed_partition(event, diagnostic)
    detail = dict(initial_k=count, selected_count=count, fits=[], fallback=True)
    if count == 0 or count not in bank or 1 not in bank:
        return Reconstruction(count, event, fallback.T, detail)
    if count == 1:
        single = try_fixed_one(event, diagnostic, bank, labels)
        if single is not None:
            return single
    search = FitSearch(event, bank)
    starts = [("bank", bank[count])]
    if count == 1:
        starts.extend(foreground(event, diagnostic))
    if count == 2 and 3 in bank:
        starts.extend(((f"merge_{a}_{b}", merge(bank[3], a, b)) for a, b in ((0, 1), (0, 2), (1, 2))))
    rows = primary_bank(search, starts) if count == 2 else charge_bank(search, starts)
    winner = best_fit(rows, "ll")
    if winner is None:
        return Reconstruction(count, event, fallback.T, detail)
    fits = [dict(r) for r in rows]
    index = next((j for j, r in enumerate(rows) if r is winner))
    detail.update(
        fallback=False,
        selected_fit_index=index,
        fits=[{key: value for key, value in r.items() if key != "posterior"} for r in fits],
    )
    return Reconstruction(count, event, winner["posterior"].T, detail)
