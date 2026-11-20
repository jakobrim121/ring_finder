"""Compare a beam-entry split with a matched low-turn charged trajectory."""

import numpy as np

from balanced_search import feasible
from chain_comparison import counted
from charge_starts import recoil_start
from entrance_track import EntranceChart
from lineage_competitor import OneLineage, starting_paths
from physical_pulse import components, profile
from track_coordinates import fit


def score(event, comparison, two_leg, state, port, budget, *, per_module=False):
    """Use the same entrance, charge response and relative clock for both counts."""
    charges = event.charges_pe
    start_axis = np.array([0.0, 0.0, 1.0])
    options = dict.fromkeys(((two_leg.particles[0],) * 2, two_leg.particles, ("proton",) * 2))
    nominees = []
    for species in options:
        trial = OneLineage(event.pmt_ids, charges, comparison.model.bg, species, 4, 8)
        near_axis = starting_paths(state, two_leg.particles, species)
        if near_axis is None:
            continue
        aligned = near_axis.copy()
        aligned[3:6] = start_axis
        aligned[6:9] -= (aligned[6:9] @ start_axis) * start_axis
        aligned = starting_paths({"parameters": aligned}, species, species)
        seeds = []
        for x in (near_axis, aligned):
            x = recoil_start(trial, x)
            if x is None or not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5:
                continue
            seeds.append((trial(x, True)["ll"], x))
        if seeds:
            seed_ll, seed = max(seeds, key=lambda pair: pair[0])
            nominees.append((seed_ll, trial, seed))
    results = []
    for _, trial, seed in sorted(nominees, key=lambda q: -q[0])[:2]:
        solver, x = fit(
            trial,
            seed,
            budget,
            budget,
            full_likelihood=True,
            chart=EntranceChart(trial, seed, port),
            check_initial=True,
        )
        if (
            x[:2] @ x[:2] > port.radius**2 + 1e-9
            or not feasible(trial, x)
            or trial.hardware_margin(x) <= 1e-5
        ):
            continue
        args = event.pmt_ids, charges, comparison.model.bg, trial.particles, 4
        fine = OneLineage(*args, 32)
        endpoint = fine(x, True)
        error = abs(OneLineage(*args, 16)(x, True)["ll"] - endpoint["ll"])
        if error >= 0.01 or counted(fine, endpoint).shape[1] == 1:
            continue
        rates, centers = components(event, fine, endpoint, separate_knockon=True)
        pulse = profile(event, fine, endpoint, precomputed=(rates, centers), timing_only=True)
        if pulse is None:
            continue
        # Two physical sections are still present in the low-turn fit.
        # Use the original dimension and pay equally for the event clock.
        value = 2.0 * (endpoint["ll"] + pulse["timing_ll"]) - (fine.dimension + 1) * comparison.log_n
        report = dict(
            score=float(value),
            charge_ll=float(endpoint["ll"]),
            timing_ll=pulse["timing_ll"],
            particles=fine.particles,
            evaluations=int(solver.nfev),
            quadrature_ll_error=float(error),
        )
        joint = None
        if per_module:
            from module_optics import shapes

            measured = comparison.measured(
                counted(fine, endpoint),
                report,
                shapes(fine, endpoint),
                rates,
                centers + pulse["clock_ns"],
                fine.dimension + 1,
                (5,),
            )
            joint = measured.joint
        results.append((report, joint))
    return max(results, key=lambda row: row[0]["score"], default=(None, None))
