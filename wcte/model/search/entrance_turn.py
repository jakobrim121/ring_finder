"""Search the measured two-ring seed for a shared beam-entry interaction."""

import numpy as np

from balanced_search import feasible
from chain_comparison import MARGIN, counted
from charge_starts import allowed_chord, recoil_start
from compare_count_models import Comparison
from coupled_model import Model as Coupled
from entrance_track import Aperture, EntranceChart
from gated_optical_fit import fit as fit_optical
from joined_segments import separated_directions
from material import material
from qualified_optical_tracks import paths_for_cone
from residual_joint_optics import path_from_geometry
from recoil_competitor import render
from track_coordinates import fit as optimize
from wctering.seeding.coherent_timing import N_PHASE


def initial_parent(launch, join, mass, angle, water=None):
    """Infer the entering direction, yield and range from a measured join."""
    axis = join - launch
    length = np.linalg.norm(axis)
    if length < 0.015:
        return
    axis /= length
    if axis[2] <= 0.0:
        return
    if water is None:
        water = allowed_chord(launch, axis)
    if water is None or water[0] > 0.0 or water[1] < length:
        return
    table = material(mass)
    opening = min(angle, np.arccos(1.0 / N_PHASE) - 0.004)
    beta = 1.0 / (N_PHASE * np.cos(opening))
    midpoint_energy = table["mass"] * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
    midpoint_energy = np.clip(midpoint_energy, table["energy"][0] + 1.0, table["energy"][-1])
    midpoint_range = float(table["range_at"](midpoint_energy)) / 1000.0
    minimum = length + 0.001
    maximum = table["maximum_range_mm"] / 1000.0
    for value in np.unique(
        np.clip([0.5 * length + midpoint_range, length + table["margin_mm"] / 1000.0], minimum, maximum)
    ):
        if not minimum <= value <= maximum:
            continue
        yield np.r_[launch, axis, np.zeros(3), value, length / value, np.log(np.deg2rad(2.0))]


def nominate(event, model, geometry, amplitude, *, scan=False, fast=False, value_only=False):
    """Fit each of the two measured cones as the outgoing particle."""
    if len(geometry["axes"]) != 2:
        return []
    try:
        cones = [path_from_geometry(geometry, j) for j in range(2)]
    except (ArithmeticError, ValueError):
        return []
    port = Aperture()
    launch = np.array([0.0, 0.0, port.z])
    args = event.pmt_ids, event.charges_pe, model.bg
    logs = np.log([model.response, max(amplitude[0], 1e-8)])
    ranked, models = [], {}
    for i, cone in enumerate(cones):
        for mass, path in paths_for_cone(model, cone, amplitude[i + 1]):
            if mass == "electron":
                continue
            start = path[:3].copy()
            if start[2] <= port.z + 0.012:
                continue
            for offset in (0.0, 0.08):
                join = start + offset * path[3:6]
                length = path[9] * path[10] - offset
                if length <= 0.012:
                    continue
                child = path.copy()
                child[:3] = join
                child[9] -= offset
                child[10] = length / child[9]
                displacement = join - launch
                distance = np.linalg.norm(displacement)
                if distance < 0.015:
                    continue
                water = allowed_chord(launch, displacement / distance)
                if water is None:
                    continue
                for parent in ("pion", "proton"):
                    particles = parent, mass
                    if particles not in models:
                        models[particles] = Coupled(*args, particles, 4, 4)
                    trial = models[particles]
                    for incoming in initial_parent(launch, join, parent, cones[1 - i][7], water):
                        if incoming[3:6] @ child[3:6] >= np.cos(np.deg2rad(15.0)):
                            continue
                        x = recoil_start(trial, np.r_[incoming, child[3:], logs])
                        if x is None or not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5:
                            continue
                        if value_only:
                            from charge_probe import charge_state

                            ll, mean, core = charge_state(trial, x, knockon=True)
                            light = core[:, model.ix]
                            means_hit = mean[model.ix]
                        else:
                            state = trial(x, True)
                            rate = np.exp(x[-2:])
                            ll, means_hit = state["ll"], state["means_hit"]
                            if fast:
                                core = trial.primary_cache[2][:, model.ix]
                            else:
                                core = [
                                    render(model, leg, mass, 4)[1][model.ix]
                                    for leg, mass in zip((incoming, child), particles)
                                ]
                            light = rate[0] * np.asarray(core)
                        support = [
                            float(
                                np.sum(
                                    1.0 - model.obs.no_source(means_hit, np.clip(row / means_hit, 0.0, 1.0))
                                )
                            )
                            for row in light
                        ]
                        # Use the actual counted source for the fast seed
                        # screen. Leave room for the geometry to improve
                        # from this coarse integral before the final check.
                        if min(support) >= (18.0 if fast else 25.0):
                            ranked.append((ll, particles, x))
    if scan:
        from aperture_junction_seeds import nominate as timed_starts

        ranked.extend(timed_starts(event, model, geometry, amplitude, port, initial_parent))
    kept = []
    for score, particles, x in sorted(ranked, key=lambda row: -row[0]):
        if any(
            particles == mass and x[12:15] @ q[12:15] > 0.97 and abs(x[9] * x[10] - q[9] * q[10]) < 0.09
            for _, mass, q in kept
        ):
            continue
        kept.append((score, particles, x))
        if len(kept) == (2 if fast else 3):
            break
    return kept


def fit(event, model, geometry, basis, amplitude, bank, winner, single, budget):
    """Require both entry-constrained charged paths to survive a one-ring rival."""
    if (
        event.timing_selection["prompt_window_ns"] is None
        or winner[2]["count"] != 1
        or single is None
        or (geometry.get("emission_model") == "full_water_line")
    ):
        return (None, [])
    other = bank.get("2")
    if other is None or len(other["axes"]) != 2:
        return (None, [])
    old = winner[2]
    physical_reference = old.get("particles") and old.get("order")
    if not physical_reference and len(basis) != 2:
        return (None, [])
    port = Aperture()
    initial = np.asarray(other["weights"]) * amplitude.sum()
    starts = nominate(event, model, other, initial, scan=False, fast=True, value_only=False)
    if not starts:
        return (None, [])
    comparison = Comparison(event, model)
    reference = None
    reports = []
    best = None
    control = None
    for _, particles, x in starts:
        args = (event.pmt_ids, event.charges_pe, model.bg, particles, 4)
        trial = Coupled(*args, 8)
        result, z = optimize(
            trial,
            x,
            budget,
            budget,
            full_likelihood=True,
            chart=EntranceChart(trial, x, port),
            check_initial=True,
        )
        if (
            z[:2] @ z[:2] > port.radius**2 + 1e-09
            or not feasible(trial, z)
            or trial.hardware_margin(z) <= 1e-05
        ):
            continue
        fine = Coupled(*args, 32)
        state = fine(z, True)
        coarse = Coupled(*args, 16)
        low_ll = coarse(z, True)["ll"]
        error = abs(state["ll"] - low_ll)
        order = 16
        fractions = counted(fine, state)
        turn, resolved = separated_directions(fine, state)
        record = dict(
            family="entrance_turn",
            count=fractions.shape[1] - 1,
            supported=False,
            particles=particles,
            parameters=z.tolist(),
            order=order,
            evaluations=int(result.nfev),
            quadrature_ll_error=error,
            angle_deg=float(np.rad2deg(turn)),
            gain=0.0,
        )
        reports.append(record)
        if error >= 0.01 or fractions.shape[1] != 3 or not resolved:
            continue
        # Only evaluate the timed reference after the turn passes physical and optical support.
        if reference is None:
            if physical_reference:
                reference = comparison.physical(old)
            else:
                reference = comparison.optical(geometry, basis, amplitude, winner[1], old)
            if reference is None:
                return (None, reports)
        rival = comparison.physical(record, restored=(fine, state, fractions))
        if rival is None:
            continue
        delta = 2.0 * (rival.joint - reference.joint)
        gain = rival.score - reference.score - delta.max(initial=0.0)
        evidence = comparison.resolved(reference, rival)
        record.update(gain=float(gain), evidence=evidence)
        if gain <= MARGIN or evidence["lower_pmts"] < 30.0:
            continue
        if control is None and len(basis) == 2:
            control = fit_optical(event, geometry, model.bg, budget, strength=1.0)
        if control is not None:
            penalty = (fine.dimension - 10) * np.log(len(model.active))
            gap = 2.0 * (rival.joint.sum() - control[2]["joint_score"]) - penalty
            record["optical_control_gap"] = float(gap)
            if gap <= MARGIN:
                continue
        from pulse_residual import qualify

        accepted, checks = qualify(
            event,
            single,
            fine,
            state,
            fractions,
            old,
            (order, budget),
            compare_lineage="conversion",
            within_timing="survey_lineage",
            allow_profiled_turn=False,
        )
        record["count_control"] = checks
        # Compare shallow turns with a port-constrained single lineage before splitting the ring.
        if accepted and turn < np.deg2rad(35.0):
            from aperture_lineage import score as score_lineage

            null, by_module = score_lineage(event, comparison, fine, state, port, budget, per_module=False)
            record["surveyed_lineage"] = null
            if null is not None and rival.score - null["score"] <= MARGIN:
                continue
        if accepted:
            from detached_ring_review import review as review_detached

            extra, diagnostics = review_detached(
                event, model, bank, fine, state, fractions, 3, refine_tracks=True
            )
            record["detached_tracks"] = diagnostics
            if extra is not None:
                _, expanded, detail = extra
                for key in ("particles", "parameters", "order"):
                    record["entry_" + key] = record.pop(key)
                record.update(count=3, detached_track=detail)
                fractions = expanded
            elif diagnostics:
                continue
            if best is None or gain > best[0]:
                best = (float(gain), fractions, record)
    if best is not None:
        best[2]["supported"] = True
    return (best, reports)
