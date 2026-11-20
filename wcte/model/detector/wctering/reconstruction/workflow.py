"""Build charge-fit starts and evaluate both optical source families."""

from wctering.foreground.search import fit_starts
from wctering.results import seed_partition
from wctering.seeding.event_cache import fresh_event_local_cache


def foreground(event, diagnostic):
    """Extract one-ring starts from the retained optical seed geometry."""
    reference = diagnostic["_foreground_reference"]
    inputs = diagnostic["_foreground_inputs"]
    _, original_labels, _ = seed_partition(event, reference)
    with fresh_event_local_cache():
        return fit_starts(reference, inputs, original_labels)


def charge_bank(search, starts):
    """Both charge families are fitted for every declared initial geometry."""
    rows = []
    for name, geometry in starts:
        for kind in ("point", "finite"):
            fit = search.fit(len(geometry["vertices"]), kind, geometry)
            if fit.get("excluded"):
                continue
            fit["origin"] = name
            rows.append(fit)
    return rows
