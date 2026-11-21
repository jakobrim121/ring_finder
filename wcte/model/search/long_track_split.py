"""Seed a two-leg hypothesis when one long arc has absorbed both footprints."""

import numpy as np

from balanced_search import feasible
from bare_model import Model as Bare
from chain_comparison import MARGIN
from chain_sources import paths
from charge_starts import recoil_start, weighted_quantile
from coherent_partition import prediction_time
from joined_segments import arc, separated_directions
from junction_residual_seeds import directions, ridge_scores
from material import material
from recoil_competitor import render
from serial_refinement import leg_starts
from source import trajectory
from terminal_track_seeds import pulse_support, ridge_peaks
from wctering.seeding.coherent_timing import C
from wcte.counting import track_counting


def peaks(event, model, state):
    """Look for a finite outgoing footprint after trimming the fitted arc.

    A fit bridging two tracks can also move away from their interaction
    transversely. Probe that uncertainty along the measured arc's bend.
    """
    x = state["parameters"]
    (mass,) = model.particles
    length = x[9] * x[10]
    if length < 0.65 or mass == "electron" or len(event.pmt_ids) < 180:
        return
    q = np.minimum(event.charges_pe, 2.0)
    time = event.times_ns - np.median(event.times_ns)
    arrival, illuminated = prediction_time(event, x, (mass,), np.zeros_like(q))
    weight = q * state["posterior"][1] * (illuminated[:, 0] > 1e-6)
    if weight.sum() < 85.0:
        return
    clock = weighted_quantile(time - arrival[:, 0], weight, [0.5])[0]
    response, background = np.exp(x[-2:])
    water = material(mass)
    grid = directions()
    phases = np.array([0.32, 0.46, 0.60, 0.74, 0.88])
    points, tangents, _, _ = trajectory(x[:12], phases)
    windows = (0.30, 0.60, 1.05)
    for u, point, tangent in zip(phases, points, tangents):
        prefix = arc(x[:12], 0.0, u)
        if prefix is None:
            continue
        nodes = np.linspace(0.0, u, 9)
        energy = water["energy_at"](1000.0 * (x[9] - length * nodes))
        inv_beta = 1.0 / np.sqrt(1.0 - (1.0 + energy / water["mass"]) ** -2)
        birth = clock + 500.0 * length / C * ((inv_beta[:-1] + inv_beta[1:]) @ np.diff(nodes))
        cosine = grid @ tangent
        # Screen turns close to the lineage boundary too. Their fits later
        # have to beat a continuously bending charged trajectory.
        preview = np.cos(np.radians(track_counting.get().turn_offset(-4.0)))
        axes = grid[(cosine < preview) & (cosine > 0.30)]
        bend = x[6:9] - (x[6:9] @ tangent) * tangent
        if np.linalg.norm(bend) > 0.03:
            bend /= np.linalg.norm(bend)
        shifts = (0.0, -0.13, 0.13) if np.linalg.norm(bend) > 0.03 else (0.0,)
        for offset in shifts:
            parent = prefix.copy()
            parent[:3] += offset * bend
            if model.hardware_margin(np.r_[parent, x[-2:]]) <= 1e-5:
                continue
            _, light, _ = render(model, parent, mass, 4)
            mean = response * light + background * model.bg
            _, gradient = model.obs.score(mean, model.ix)
            excess = np.clip(-gradient[model.ix] * mean[model.ix], -0.7, 2.0)
            if np.count_nonzero(excess > 0.25) < 45:
                continue
            endpoint = point + 1000.0 * offset * bend
            delta = event.pmt_positions_mm - endpoint
            distance = np.linalg.norm(delta, axis=1)
            scores = ridge_scores(
                delta, distance, axes, time, q, excess, birth, tuple(1000.0 * length for length in windows)
            )
            for window, score in zip(windows, scores):
                for strength, axis, opening, span in ridge_peaks(
                    score[:, :, 1],
                    axes,
                    delta,
                    time,
                    birth,
                    q,
                    excess,
                    1000.0 * window + 90.0,
                    (0.12, window + 0.09),
                ):
                    if strength >= 24.0:
                        rank = strength / np.sqrt(1.0 + window / 0.45)
                        yield rank, parent, endpoint / 1000.0, axis, opening, strength, span


def starts(event, model, state):
    """Use low-order calibrated charge to rank the remaining material modes."""
    (mass,) = model.particles
    trials, chosen, candidates = {}, [], []
    for _, prefix, vertex, axis, opening, light, span in sorted(
        peaks(event, model, state), key=lambda row: row[0], reverse=True
    ):
        if any(
            np.linalg.norm(vertex - v) < 0.16 and axis @ a > 0.98 and abs(span - s) < 0.22
            for v, a, s in chosen
        ):
            continue
        chosen.append((vertex, axis, span))
        for daughter in dict.fromkeys((mass, "pion", "proton")):
            particles = mass, daughter
            if particles not in trials:
                trials[particles] = Bare(event.pmt_ids, event.charges_pe, model.bg, particles, 4, 8)
            trial = trials[particles]
            for leg in leg_starts(trial, vertex, axis, opening, light, span, daughter):
                if leg[6] * leg[7] < 0.14:
                    continue
                x = recoil_start(trial, np.r_[prefix, leg, state["parameters"][-2:]])
                if x is not None and feasible(trial, x) and trial.hardware_margin(x) > 1e-5:
                    candidates.append((float(trial(x, True)["ll"]), particles, x))
        if len(chosen) == 8:
            break
    distinct = []
    for score, particles, x in sorted(candidates, key=lambda row: row[0], reverse=True):
        if any(
            particles == masses and abs(x[9] * x[10] - p[9] * p[10]) < 0.18 and x[12:15] @ p[12:15] > 0.98
            for _, masses, p in distinct
        ):
            continue
        distinct.append((score, particles, x))
        if len(distinct) == 2:
            break
    return distinct


def evidence(
    event,
    single,
    model,
    state,
    fractions,
    diagnostics,
    lineage_budget=0,
    *,
    comparison=(),
    timed=False,
    separate_knockon=False,
    resolved_turn=False,
):
    """Require two measured arcs and test a smooth trajectory through them."""
    from causal_sections import check
    from continuous_reference import fit as continuous_fit

    local = [p for p, _ in paths(state["parameters"][:-2])]
    spans = [1000.0 * p[9] * p[10] for p in local]
    diagnostics["track_lengths_mm"] = spans
    if min(spans) < 180.0 or "electron" in model.particles:
        return False
    direction = trajectory(local[0], np.ones(1))[1][0]
    turn = np.arccos(np.clip(direction @ local[1][3:6], -1.0, 1.0))
    diagnostics["turn_deg"] = float(np.rad2deg(turn))
    if turn <= track_counting.get().merge_angle_rad:
        return False
    if resolved_turn:
        # Use the same separation test as the other connected-track routes.
        # A fitted kink inside the combined optical widths does not resolve
        # a second direction at the boundary of the ring definition.
        _, resolved = separated_directions(model, state)
        diagnostics["resolved_turn"] = resolved
        if not resolved:
            return False
    clock = check(event, model, state, fractions)
    diagnostics["clock"] = clock
    if clock is None or not clock["compatible"]:
        return False
    support = pulse_support(event, model, state, fractions)
    diagnostics["conditioned_support"] = support
    if support is None or min(row["leave_one_module"] for row in support) < track_counting.get().scaled_support(45.0):
        return False
    weak = int(np.argmin([row["pmt_support"] for row in support]))
    null = continuous_fit(event, *single, model, state, weak)
    if null is None:
        return False
    rival, rival_state, _, _ = null
    gap = 2.0 * (state["ll"] - rival_state["ll"]) - (model.dimension - rival.dimension) * np.log(
        len(model.active)
    )
    diagnostics["continuous_score_gap"] = float(gap)
    if gap <= MARGIN:
        return False
    if lineage_budget:
        from lineage_competitor import fit

        trial_score = state["ll"]
        previous = []
        if timed:
            from physical_pulse import profile

            pulse = profile(event, model, state, separate_knockon=separate_knockon)
            if pulse is None:
                return False
            trial_score += pulse["timing_ll"]
            diagnostics["pulse_fit"] = pulse
            # Reuse other measured arcs as starts and as timing controls.
            # A high charge likelihood alone can reward a split whose two
            # sections have the wrong speed and earliest-light pattern.
            ordered = sorted(
                (r for r in comparison if r.get("particles") and r["count"] in (1, 2)),
                key=lambda r: r["gain"],
                reverse=True,
            )
            for record in ordered[:4]:
                particles = record["particles"]
                if len(particles) == 2 and len(previous) < 3:
                    previous.append((particles, np.asarray(record["parameters"])))
                if record["count"] != 1:
                    continue
                from coupled_model import Model as Coupled

                rival = Coupled(event.pmt_ids, event.charges_pe, model.bg, particles, 4, 2 * record["order"])
                rival_state = rival(np.asarray(record["parameters"]), True)
                control = profile(event, rival, rival_state, separate_knockon=separate_knockon)
                if control is None:
                    continue
                separation = 2.0 * (trial_score - rival_state["ll"] - control["timing_ll"])
                if separation <= MARGIN:
                    diagnostics["one_ring_pulse_control"] = dict(control, joint_gap=separation)
                    return False
        lineage = fit(
            event,
            single[0],
            model,
            state,
            lineage_budget,
            other_starts=previous,
            timed=timed,
            separate_knockon=separate_knockon,
        )
        diagnostics["one_lineage"] = lineage
        if lineage is None or 2.0 * (trial_score - lineage["ll"] - lineage.get("timing_ll", 0.0)) <= MARGIN:
            return False
    return True
