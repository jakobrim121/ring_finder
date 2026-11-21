"""Qualify the optical charge partition before the physical track comparison."""

import numpy as np

from wctering.reconstruction.candidates import FitSearch
from wctering.reconstruction.one_ring_partition import try_fixed_one
from wctering.reconstruction.proposals import merge
from wctering.reconstruction.two_ring_partition import primary_bank
from wctering.reconstruction.workflow import charge_bank, foreground
from wctering.results import Reconstruction, seed_partition
from wctering.seeding.partition_support import _final_partition_background_assignment
from wcte.counting import optical_sources_resolved, track_counting


def reference(event, diagnostic, bank):
    """Retry a lower optical count when a fitted component fails the ring cuts."""
    initial = diagnostic["predicted_ring_class"]
    rejected = []
    available = (diagnostic.get("unified_counter") or {}).get("available_classes")
    for count in range(initial, -1, -1):
        if count != initial:
            if count > 0 and available is not None and not available[count]:
                continue
            # Only the original count owns the polished seed partition.
            diagnostic = dict(diagnostic, predicted_ring_class=count)
            diagnostic.pop("analytic_partition", None)
            diagnostic["final_partition_background_assignment"] = _final_partition_background_assignment(
                diagnostic, count
            )
        for result in fitted_partitions(event, diagnostic, bank):
            if count == 0 or countable(result, diagnostic, bank):
                if rejected:
                    result.diagnostics["ring_definition_rejected_counts"] = rejected
                return result
        rejected.append(count)


def countable(result, diagnostic, bank):
    from readout import observation

    geometry = result.geometry
    if geometry is None:
        partition = diagnostic.get("analytic_partition") or {}
        geometry = (
            dict(
                axes=[ring["direction"] for ring in partition["components"]],
                vertices=[ring["vertex_mm"] for ring in partition["components"]],
            )
            if partition.get("applied")
            else bank.get(result.count)
        )
    if geometry and not optical_sources_resolved(geometry["axes"], geometry["vertices"]):
        return False
    event = result.hits
    charge = event.charges_pe
    ids, tubes = np.unique(event.pmt_ids, return_inverse=True)
    minimum = track_counting.get().min_track_pmts
    if len(ids) < minimum:
        return False
    # The optical fit models charge proportions. Use the observed PE as
    # the intensity estimate in the calibrated photon-count posterior.
    obs = observation(charge)
    for ring in result.fractions.T[1:]:
        missing = np.ones(len(ids))
        np.multiply.at(missing, tubes, obs.no_source(charge, np.clip(ring, 0.0, 1.0)))
        if (1.0 - missing).sum() < minimum:
            return False
    return True


def fitted_partitions(event, diagnostic, bank):
    """Try the best fitted partition first, with the seed partition as fallback."""
    count, labels, fallback = seed_partition(event, diagnostic)
    detail = dict(initial_k=count, selected_count=count, fits=[], fallback=True)
    if count == 0 or count not in bank or 1 not in bank:
        yield Reconstruction(count, event, fallback.T, detail)
        return
    if count == 1:
        single = try_fixed_one(event, diagnostic, bank, labels)
        if single is not None:
            yield single
            yield Reconstruction(count, event, fallback.T, detail)
            return
    search = FitSearch(event, bank)
    starts = [("bank", bank[count])]
    if count == 1:
        starts.extend(foreground(event, diagnostic))
    if count == 2 and 3 in bank:
        starts.extend(((f"merge_{a}_{b}", merge(bank[3], a, b)) for a, b in ((0, 1), (0, 2), (1, 2))))
    rows = primary_bank(search, starts) if count == 2 else charge_bank(search, starts)
    qualified = [(i, row) for i, row in enumerate(rows) if row["qualified"]]
    fits = [{key: value for key, value in row.items() if key != "posterior"} for row in rows]
    for index, row in sorted(qualified, key=lambda entry: -entry[1]["ll"]):
        info = dict(detail, fallback=False, selected_fit_index=index, fits=fits)
        yield Reconstruction(count, event, row["posterior"].T, info)
    yield Reconstruction(count, event, fallback.T, detail)
