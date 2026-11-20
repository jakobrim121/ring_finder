"""Reconstruct a surveyed two-section trajectory in a three-ring beam fit."""

from itertools import combinations

import numpy as np

from balanced_search import feasible
from chain_comparison import counted
from coupled_model import Model as Coupled
from entrance_track import EntranceChart
from joined_segments import separated_directions
from track_coordinates import fit as optimize


def pairs(geometry, amplitude, two_cones):
    """Try both the independent two-cone seed and pairs of three-cone ridges."""
    if two_cones is not None and len(two_cones["axes"]) == 2:
        yield two_cones, amplitude.sum() * np.asarray(two_cones["weights"])
    if len(geometry["axes"]) != 3:
        return
    for keep in combinations(range(3), 2):
        trial = {
            key: [geometry[key][i] for i in keep]
            for key in ("vertices", "axes", "openings_deg", "widths_deg", "lengths_mm")
            if key in geometry
        }
        (j,) = set(range(3)) - set(keep)
        amp = np.r_[amplitude[0] + amplitude[j + 1], amplitude[np.asarray(keep) + 1]]
        yield trial, amp


def track_fit(event, model, masses, x, port, budget, *, module_weight=0.0, value_only=False):
    """Refine the geometry at the measured aperture and check the integral."""
    args = event.pmt_ids, event.charges_pe, model.bg, masses, 4
    trial = Coupled(*args, 8)
    if module_weight:
        from module_optics import ConditionalSearch

        # Guide the geometry toward the measured PMT pattern. Final
        # verification always uses the ordinary calibrated likelihood.
        trial.obs = ConditionalSearch(trial.obs, trial.active, trial.ix, module_weight)
    chart = EntranceChart(trial, x, port)
    result, z = optimize(trial, x, budget, budget, full_likelihood=True, chart=chart, check_initial=True)
    if z[:2] @ z[:2] > port.radius**2 + 1e-9 or not feasible(trial, z) or trial.hardware_margin(z) <= 1e-5:
        return None
    fine = Coupled(*args, 32)
    state = fine(z, True)
    f = counted(fine, state)
    lower = Coupled(*args, 16)
    if value_only:
        from charge_probe import log_likelihood

        checked = log_likelihood(lower, z, knockon=True)
    else:
        checked = lower(z, True)["ll"]
    error = abs(state["ll"] - checked)
    angle, resolved = separated_directions(fine, state)
    report = dict(
        family="aperture_pair_review",
        count=f.shape[1] - 1,
        particles=masses,
        order=16,
        parameters=z.tolist(),
        evaluations=int(result.nfev),
        supported=False,
        quadrature_ll_error=float(error),
        angle_deg=float(np.rad2deg(angle)),
    )
    return report, (fine, state, f) if f.shape[1] == 3 and error < 0.01 and resolved else None
