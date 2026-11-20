"""Search for a charged daughter after removing a possibly overfit track end."""

import numpy as np

from bare_model import Model as Bare
from balanced_search import feasible
from charge_starts import allowed_chord, recoil_start, weighted_quantile
from coherent_partition import prediction_time
from joined_segments import arc, separated_directions
from junction_residual_seeds import OPENINGS, directions, ridge_scores
from material import material
from recoil_competitor import render
from serial_refinement import leg_starts
from source import trajectory
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def peaks(event, model, state, *, variable_span=False):
    """Subtract each candidate prefix before looking for the outgoing light.

    A long fitted track can run past a real interaction and assign that
    daughter to the parent. Recompute the excess for each plausible end.
    """
    (mass,) = model.particles
    x = state["parameters"]
    length = x[9] * x[10]
    # The high-confidence continuation search targets substantial terminal
    # tracks. A sub-160 mm candidate from a very short parent can still
    # be tested by the wider search, which has its own visibility checks.
    minimum_length = 0.48 if variable_span == "resolved_continuation" else (0.10 if variable_span else 0.22)
    if mass == "electron" or length < minimum_length:
        return
    q = np.minimum(event.charges_pe, 2.0)
    time = event.times_ns - np.median(event.times_ns)
    arrival, illumination = prediction_time(event, x, (mass,), np.zeros_like(q))
    weight = q * state["posterior"][1] * (illumination[:, 0] > 1e-6)
    if weight.sum() < 28.0:
        return
    clock = weighted_quantile(time - arrival[:, 0], weight, [0.5])[0]
    response, diffuse = np.exp(x[-2:])
    water = material(mass)
    grid = directions()
    ends = (0.32, 0.48, 0.62, 0.74, 0.85, 0.94, 1.0) if variable_span else (0.52, 0.66, 0.78, 0.88, 0.96, 1.0)
    for end in ends:
        parent = arc(x[:12], 0.0, end)
        if parent is None:
            continue
        # The short path must begin after the surviving parent's optical
        # flight time. Use CSDA speed along that prefix to place its clock.
        u = np.linspace(0.0, end, 9)
        energy = water["energy_at"](1000.0 * (x[9] - length * u))
        inv_beta = 1.0 / np.sqrt(1.0 - (1.0 + energy / water["mass"]) ** -2)
        birth = clock + 500.0 * length / C * ((inv_beta[:-1] + inv_beta[1:]) @ np.diff(u))
        point, tangent, _, _ = trajectory(parent, np.ones(1))
        delta = event.pmt_positions_mm - point[0]
        distance = np.linalg.norm(delta, axis=1)
        _, light, _ = render(model, parent, mass, 4)
        mean = response * light + diffuse * model.bg
        ll, gradient = model.obs.score(mean, model.ix)
        excess = np.clip(-gradient[model.ix] * mean[model.ix], -0.7, 2.0)
        if np.count_nonzero(excess > 0.25) < 20:
            continue
        axes = grid[grid @ tangent[0] < np.cos(np.deg2rad(18.0 if variable_span else 20.0))]
        windows = (120.0, 300.0, 730.0) if variable_span else (240.0,)
        scores = ridge_scores(delta, distance, axes, time, q, excess, birth, windows)
        for length_mm, score in zip(windows, scores):
            # Larger intervals catch more unrelated light. Compare their
            # peaks after allowing for the extra searched interval.
            if variable_span:
                limit, spans = length_mm + 110.0, (0.025, length_mm / 1000.0 + 0.11)
            else:
                limit, spans = 500.0, (0.04, 0.48)
            for support, axis, opening, span in ridge_peaks(
                score[:, :, 1], axes, delta, time, birth, q, excess, limit, spans
            ):
                rank = support / np.sqrt(1.0 + length_mm / 240.0) if variable_span else ll + support
                yield rank, parent, point[0] / 1000.0, axis, opening, support, span


def ridge_peaks(strength, axes, delta, time, birth, charge, excess, limit, spans):
    """Keep distinct directions and infer a finite span from prompt hits."""
    found = []
    for flat in np.argsort(strength.ravel())[::-1]:
        j, k = np.unravel_index(flat, strength.shape)
        support, axis, opening = strength[j, k], axes[j], OPENINGS[k]
        if support < 12.0:
            break
        if any(axis @ prior > 0.972 for prior in found):
            continue
        along = delta @ axis
        radial = np.linalg.norm(delta - along[:, None] * axis, axis=1)
        sine, cosine = np.sin(opening), np.cos(opening)
        emission = along - radial * cosine / sine
        expected = (N_PHASE * cosine * emission + N_GROUP * radial / sine) / C
        use = (emission >= -95.0) & (emission <= limit)
        weight = (
            charge * np.maximum(excess, 0.0) * use * np.exp(-0.5 * ((time - birth - expected) / 2.0) ** 2)
        )
        found.append(axis)
        if weight.sum() >= 15.0:
            extent = weighted_quantile(emission, weight, [0.80])[0] / 1000.0 + 0.035
            yield support, axis, opening, np.clip(extent, *spans)
        if len(found) == 2:
            break


def starts(event, model, state, *, variable_span=False, bright_only=False):
    """Shortlist contained, energy-compatible tracks at a residual peak."""
    # This route seeks two well-measured overlapping arcs. The earlier
    # residual searches already address shorter, fainter sections.
    if bright_only and (len(event.charges_pe) < 360 or np.minimum(event.charges_pe, 2.0).sum() < 480.0):
        return []
    (mass,) = model.particles
    candidates, checked, models = [], [], {}
    by_endpoint = {}
    for _, parent, point, axis, opening, charge, span in sorted(
        peaks(event, model, state, variable_span=variable_span), key=lambda row: row[0], reverse=True
    ):
        if any(
            np.linalg.norm(point - v) < 0.13
            and axis @ a > 0.96
            and (not variable_span or abs(span - length) < 0.20)
            for v, a, length in checked
        ):
            continue
        end = tuple(point)
        if variable_span and by_endpoint.get(end, 0) >= 2:
            continue
        by_endpoint[end] = by_endpoint.get(end, 0) + 1
        checked.append((point, axis, span))
        water = allowed_chord(point, axis)
        if water is None or water[0] > 1e-5 or water[1] < 0.018:
            continue
        materials = (
            (mass,) if variable_span == "resolved_continuation" else (mass, "pion", "proton", "electron")
        )
        for daughter in dict.fromkeys(materials):
            layout = mass, daughter
            if layout not in models:
                models[layout] = Bare(event.pmt_ids, event.charges_pe, model.bg, layout, 4, 4)
            trial = models[layout]
            for child in leg_starts(trial, point, axis, opening, charge, span, daughter):
                x = recoil_start(trial, np.r_[parent, child, state["parameters"][-2:]])
                if x is None or not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5:
                    continue
                candidates.append((float(trial(x, True)["ll"]), layout, x))
        if len(checked) >= (12 if variable_span else 3):
            break
    ordered = sorted(candidates, key=lambda row: row[0], reverse=True)
    if not variable_span:
        return ordered
    distinct = []
    for score, masses, x in ordered:
        if any(
            masses == prior
            and np.linalg.norm(x[:3] - y[:3]) < 0.08
            and abs(x[9] * x[10] - y[9] * y[10]) < 0.12
            and x[12:15] @ y[12:15] > 0.97
            for _, prior, y in distinct
        ):
            continue
        distinct.append((score, masses, x))
        if len(distinct) == 3:
            break
    return distinct


def evidence(event, single, trial, state, fractions, *, compare_soft=False, diagnostics=None):
    """Keep only a resolved turn with its own prompt light and common clock."""
    from causal_sections import check
    from ring_supported import preserves_prompt
    from track_resolution import competing_footprint
    from chain_comparison import counted

    angle, resolved = separated_directions(trial, state)
    # This is a search for conspicuous lost sections. Below-threshold
    # alternatives need a different test when photon statistics are sparse.
    measured = [np.sum(1.0 - trial.obs.no_source(state["means_hit"], f)) for f in fractions.T[1:]]
    if diagnostics is not None:
        diagnostics.update(turn_deg=float(np.rad2deg(angle)), section_support=measured)
    if not resolved:
        return None
    if min(measured) < 36.0:
        return None
    if compare_soft == "resolved_continuation":
        from chain_sources import paths

        sections = [path for path, _ in paths(state["parameters"][:-2])]
        # Require a macroscopic length and a resolved change of direction
        # before assigning overlapping light to a second track. This test
        # treats the mass as a fitted transport parameter.
        if (
            trial.particles[0] != trial.particles[1]
            or np.rad2deg(angle) < 28.0
            or sections[1][9] * sections[1][10] < 0.16
        ):
            return None
    first, previous = single
    prior = first, previous, counted(first, previous)
    if prior[-1].shape[1] != 2:
        return None
    retained, lost = preserves_prompt(event, prior, (trial, state, fractions))
    if diagnostics is not None:
        diagnostics["lost_prompt_hits"] = lost
    if not retained:
        return None
    clock = check(event, trial, state, fractions)
    if diagnostics is not None:
        diagnostics["clock"] = clock
    if not clock or not clock["compatible"]:
        return None
    found = competing_footprint(event, single, (trial, state), fractions)
    if compare_soft in ("conditioned", "resolved_continuation"):
        timed = pulse_support(event, trial, state, fractions)
        if diagnostics is not None:
            diagnostics["conditioned_support"] = timed
        if timed is None or min(row["lower_statistical"] for row in timed) < 30.0:
            return None
        # A bright overlapping continuation can be earlier than a real
        # daughter at the same tube. Use the probability of any daughter
        # PE when it has substantial support across separate modules.
        if found is None and min(row["leave_one_module"] for row in timed) >= 50.0:
            found = dict(short_section=int(np.argmin(measured)) + 1, conditioned_overlap=timed)
    if found is None:
        return None
    if compare_soft and min(measured) < 65.0:
        # A genuine recoil can lie below the ring definition. Compare its
        # calibrated PE budget at the fitted location before adding a ring.
        from recoil_competitor import search

        soft = search(event, *single, trial, state, fractions, found["short_section"] - 1)
        if diagnostics is not None:
            diagnostics["below_support_competitor"] = soft
        margin = 6.6349 + 5.0 * np.log(len(trial.active))
        if compare_soft == "conditioned" and soft is not None:
            # Near the visibility cut, also let the recoil move along the
            # parent. The alternative has a different topology, so compare
            # raw likelihood at the two fitted endpoints.
            from branched_recoil import fit

            branch = fit(event, *single, trial, state, fractions, found["short_section"] - 1)
            if diagnostics is not None:
                diagnostics["branch_competitor"] = branch
            if branch is not None and 2.0 * (state["ll"] - branch["ll"]) <= 6.6349:
                return None
        if soft is not None and 2.0 * (state["ll"] - soft["ll"]) <= margin:
            return None
    return dict(
        found, turn_deg=float(np.rad2deg(angle)), section_support=measured, lost_prompt_hits=lost, clock=clock
    )


def pulse_support(event, model, state, fractions):
    """Measure both sections using one clock and the first-PE likelihood."""
    from module_supported_ring import lower_bound
    from resolved_ring_evidence import first_pulse_support

    window = event.timing_selection["prompt_window_ns"]
    if window is None:
        return None
    center = np.median(event.times_ns)
    time = event.times_ns - center
    q = event.charges_pe
    arrival, light = prediction_time(event, state["parameters"], model.particles, np.zeros_like(q))
    weight = np.minimum(q, 2.0)[:, None] * fractions[:, 1:] * (light > 1e-6)
    if not weight.any():
        return None
    clock = weighted_quantile((time[:, None] - arrival).ravel(), weight.ravel(), [0.5])[0]
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(q, 0.25)) / 2.355**2)
    rates = state["means_hit"][:, None] * fractions
    # Very small core predictions should remain valid even when the timing
    # moment is numerically zero on a tube far outside the direct cone.
    arrival = np.where(light > 1e-6, arrival + clock, time[:, None] + 2.0)
    gate = np.asarray(window) - center
    return [
        lower_bound(
            event.pmt_ids,
            first_pulse_support(
                time, gate, rates, arrival, sigma, j, model.obs.offsets, model.obs.coefficients, True
            ),
        )
        for j in range(2)
    ]
