"""Look for a third serial section without preserving the two-track geometry."""

import numpy as np

from balanced_search import feasible
from chain_comparison import MARGIN, counted
from chain_sources import paths
from charge_probe import charge_state
from charge_starts import recoil_start
from coupled_model import Model
from entrance_track import Aperture, EntranceChart
from gated_photon_time import log_density
from joined_segments import arc
from module_optics import ConditionalSearch
from qualified_optical_tracks import paths_for_cone
from serial_refinement import leg_starts
from source import trajectory
from track_coordinates import fit as optimize
from wcte.counting import track_counting


def starts(model, restored, cone, photons, *, extra=()):
    """Insert or append an arc along either of the two fitted paths."""
    trial, state, _ = restored
    x = state["parameters"]
    pair = [p for p, _ in paths(x[:-2])]
    axes = np.r_[np.linspace(0.20, 0.80, 4), 0.9, 1.0]
    ranked, models = [], {}
    candidates = list(paths_for_cone(model, cone, photons))
    candidates.extend(extra)
    for particle, missing in candidates:
        for j, parent in enumerate(pair):
            if j == 0 and particle == "electron":
                continue
            endpoint = trajectory(parent, axes)[0] / 1000.0
            near = np.linalg.norm(endpoint - missing[:3], axis=1)
            for k in dict.fromkeys([len(axes) - 1, *np.argsort(near)[:2]]):
                left = arc(parent, 0.0, axes[k])
                if left is None:
                    continue
                end, direction = trajectory(left, np.ones(1))[:2]
                end = end[0] / 1000.0
                if direction[0] @ missing[3:6] >= track_counting.get().merge_cosine:
                    continue
                child = missing.copy()
                # Translate the residual arc to the parent's endpoint.
                # The full search can change the lengths of both sections.
                child[:3] = end
                if j == 1:
                    legs = (pair[0], left, child)
                    masses = (*trial.particles, particle)
                else:
                    child_end = trajectory(child, np.ones(1))[0][0] / 1000.0
                    onward = pair[1].copy()
                    onward[:3] = child_end
                    legs = (left, child, onward)
                    masses = (trial.particles[0], particle, trial.particles[1])
                args = (trial.active[trial.ix], trial.q, model.bg, masses, 4)
                if masses not in models:
                    models[masses] = Model(*args, 4)
                probe = models[masses]
                start = recoil_start(probe, np.r_[legs[0][:3], *[p[3:] for p in legs], x[-2:]])
                if start is None or not feasible(probe, start) or probe.hardware_margin(start) <= 1e-5:
                    continue
                ll, mean, core = charge_state(probe, start, knockon=True)
                hit = mean[model.ix]
                support = [
                    np.sum(1.0 - model.obs.no_source(hit, (r[model.ix] / hit).clip(0.0, 1.0))) for r in core
                ]
                if min(support) >= track_counting.get().scaled_support(19.0):
                    ranked.append((ll, j, masses, start, args))
    seen = []
    for _, j, masses, start, args in sorted(ranked, key=lambda row: -row[0]):
        if any(
            k == j
            and a[12:15] @ start[12:15] > 0.96
            and a[21:24] @ start[21:24] > 0.96
            and abs(a[9] * a[10] - start[9] * start[10]) < 0.10
            for k, a in seen
        ):
            continue
        seen.append((j, start))
        yield masses, start, args
        if len(seen) == 3:
            return


def timed_starts(compare, restored, cones=None):
    """Keep the endpoint that exposed the third section in the residual scan."""
    from additional_track_scan import endpoint_cones
    from timed_track_fit import interval_flight

    model, event = compare.model, compare.event
    fit, state, _ = restored
    pair = [p for p, _ in paths(state["parameters"][:-2])]
    ranked, models = [], {}
    if cones is None:
        cones = endpoint_cones(compare, restored)
    for j, time_only, merit, prefix, point, axis, opening, length, _ in cones:
        for particle in dict.fromkeys((fit.particles[j], "pion", "proton")):
            if j == 1:
                masses = (*fit.particles, particle)
            else:
                masses = (fit.particles[0], particle, fit.particles[1])
            if masses not in models:
                models[masses] = Model(event.pmt_ids, event.charges_pe, model.bg, masses, 4, 4)
            probe = models[masses]
            photons = min(np.sqrt(merit) * 5.0, model.total)
            for child in leg_starts(probe, point, axis, opening, photons, length, particle):
                sections = (
                    (pair[0], prefix, np.r_[point, child])
                    if j == 1
                    else (prefix, np.r_[point, child], pair[1])
                )
                x = recoil_start(
                    probe, np.r_[sections[0][:3], *[p[3:] for p in sections], state["parameters"][-2:]]
                )
                if x is None or not feasible(probe, x) or probe.hardware_margin(x) <= 1e-5:
                    continue
                ll, mean, core = charge_state(probe, x, knockon=True)
                hit = mean[model.ix]
                bright = core[:, model.ix]
                occupancy = [
                    np.sum(1.0 - model.obs.no_source(hit, (row / hit).clip(0.0, 1.0))) for row in bright
                ]
                if min(occupancy) < track_counting.get().scaled_support(19.0):
                    continue
                arrivals, elapsed = [], 0.0
                for (p, _), mass in zip(paths(x[:-2]), masses):
                    centers, _, delay, _ = interval_flight(event, p, mass)
                    arrivals.append(centers + elapsed)
                    elapsed += delay
                arrivals = np.column_stack(arrivals)
                # Estimate the nuisance clock without knowing a trigger
                # origin. Keep the full charge profile in the shortlist.
                weight = bright.T * compare.q[:, None] / hit[:, None]
                offset = np.sum(weight * (compare.time[:, None] - arrivals)) / max(weight.sum(), 1e-10)
                rates = np.column_stack([np.maximum(hit - bright.sum(0), 0.0), bright.T])
                time_score = log_density(
                    compare.time,
                    *compare.gate,
                    arrivals + offset,
                    rates,
                    compare.sigma,
                    model.obs.offsets,
                    model.obs.coefficients,
                ).sum()
                ranked.append((ll + 0.5 * time_score, j, time_only, masses, x))
    seen = []
    # A bright upstream leg can dominate the full score even when the
    # third source starts at the end of the downstream leg. Retain one
    # independent basin per junction and proposal score.
    for section, time_only in ((1, False), (0, False), (1, True), (0, True)):
        candidates = (r for r in ranked if r[1:3] == (section, time_only))
        for _, _, _, masses, x in sorted(candidates, key=lambda r: -r[0]):
            if any(
                x[12:15] @ a[12:15] > 0.96
                and x[21:24] @ a[21:24] > 0.96
                and abs(x[9] * x[10] - a[9] * a[10]) < 0.10
                for a in seen
            ):
                continue
            seen.append(x)
            yield masses, x, (event.pmt_ids, event.charges_pe, model.bg, masses, 4)
            break


def check(compare, incumbent, restored, cone, photons, budget, *, scan=False, timed=False, cones=None):
    """Give an omitted charged section the same transport and readout law."""
    model = compare.model
    port = Aperture()
    reports = []
    reference = incumbent.score
    extra = ()
    if scan:
        from additional_track_scan import nominate

        extra = nominate(compare, incumbent, restored)
    proposals = (
        timed_starts(compare, restored, cones)
        if timed
        else starts(model, restored, cone, photons, extra=extra)
    )
    for masses, start, args in proposals:
        search = Model(*args, 8)
        search.obs = ConditionalSearch(search.obs, search.active, search.ix, 1.0)
        solution, x = optimize(
            search,
            start,
            budget,
            budget,
            full_likelihood=True,
            chart=EntranceChart(search, start, port),
            check_initial=True,
        )
        if (
            x[:2] @ x[:2] > port.radius**2 + 1e-9
            or not feasible(search, x)
            or search.hardware_margin(x) <= 1e-5
        ):
            continue
        fine = Model(*args, 32)
        state = fine(x, True)
        fractions = counted(fine, state)
        coarse = Model(*args, 16)
        error = abs(state["ll"] - coarse(x, True)["ll"])
        if error >= 0.01:
            continue
        if timed:
            from timed_track_fit import endpoint

            timed_fit = endpoint(compare.event, fine, state, budget, aperture=port)
            if timed_fit is not None:
                fitted, candidate, check = timed_fit
                trial = counted(fitted, candidate)
                if trial.shape[1] == 4:
                    fine, state, fractions = fitted, candidate, trial
                    x = state["parameters"]
                    error = check["quadrature_ll_error"]
        if fractions.shape[1] != 4:
            continue
        record = dict(
            family="third_charged_section",
            count=3,
            particles=masses,
            parameters=x.tolist(),
            order=16,
            evaluations=int(solution.nfev),
            quadrature_ll_error=float(error),
        )
        measured = compare.physical(record, restored=(fine, state, fractions))
        if measured is None:
            continue
        evidence = compare.resolved(incumbent, measured)
        gap = measured.score - reference - 2.0 * (measured.joint - incumbent.joint).max(initial=0.0)
        record.update(gain=float(gap), extra_ring=evidence)
        reports.append(record)
        if gap > MARGIN and evidence["lower_pmts"] >= track_counting.get().min_track_pmts:
            return True, reports
    return False, reports
