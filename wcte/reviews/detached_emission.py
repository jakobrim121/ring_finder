"""Look for a finite secondary at a distinct, causally allowed birth point."""

import numpy as np

from aperture_receiver_ridges import rank_at
from light_competition import Template
from module_vertices import birth_points
from short_kink import matched


def measured_origins(competition, expect, optical, *, triangulate=False):
    from charge_starts import weighted_quantile
    from wctering.seeding.coherent_timing import C, N_GROUP
    from wctering.seeding.geometry import wcte_contains_vertex

    model, event = competition.model, competition.event
    _, slope = model.obs.score(expect, model.ix)
    positive = np.maximum(-slope[model.ix] * expect[model.ix], 0.0)
    clock = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0) * positive
    if q.sum() < 9.0:
        return []
    detected = []
    if triangulate:
        from detached_conversion import residual_vertices

        sites = residual_vertices(event, np.clip(positive, 0.07, 2.0), count=5)
        for point in sites:
            travel = 1000.0 * N_GROUP * np.linalg.norm(event.pmt_positions_mm * 0.001 - point, axis=1) / C
            birth = weighted_quantile(clock - travel, q, [0.5])[0]
            detected.append((point, float(birth)))
    else:
        modes = birth_points(event, model, expect, directional=True)
        detected = [(np.asarray(v) * 0.001, float(t)) for v, t, *_ in modes]
    for geometry in optical:
        if len(geometry["axes"]) == 1:
            continue
        for vertex in geometry["vertices"]:
            v = np.asarray(vertex, float)
            if not wcte_contains_vertex(v) or any(
                np.linalg.norm(v * 0.001 - old) < 0.19 for old, _ in detected
            ):
                continue
            travel = N_GROUP * np.linalg.norm(event.pmt_positions_mm - v, axis=1) / C
            birth = weighted_quantile(clock - travel, q, [0.5])[0]
            detected.append((v * 0.001, float(birth)))
    return detected[:6]


def scan(competition, beam, state, mass, optics, *, triangulate=False, electron_refit=False):
    from charge_starts import weighted_quantile
    from geometry import hardware_margin
    from joined_segments import arc
    from material import material
    from physical_pulse import flight_integral, wavefront
    from serial_refinement import leg_starts
    from source import trajectory
    from value_only_receiver import receiver_values
    from wctering.seeding.coherent_timing import C

    model, event = competition.model, competition.event
    coeff, bg = np.exp(state["parameters"][-2:])
    primary = state["parameters"][:12]
    directions = [g for _, g, _ in optics]
    original_time = event.times_ns - np.median(event.times_ns)
    core, _, _ = wavefront(model.pos[model.ix], model.normals[model.ix], primary, mass)
    weights = np.minimum(event.charges_pe, 2.0) * state["posterior"][1]
    time_zero = weighted_quantile(original_time - core[0], weights, [0.5])[0]
    flight = flight_integral(mass)
    water = material(mass)
    footprint = []
    for cut in (1.0, 0.63):
        first = arc(primary, 0.0, cut)
        if first is None:
            continue
        response = coeff * 1000.0 * first[9] * first[10]
        parts = receiver_values(model.pos, model.normals, first, 4, mass)
        expect = response * parts.sum(0) + bg * model.bg
        conditional = None
        if triangulate:
            from selected_rate_adjoint import cache

            parent_arrival = wavefront(model.pos, model.normals, first, mass, order=6)[0]
            conditional = cache(
                competition.selection,
                np.column_stack([bg * model.bg, response * parts.T]),
                parent_arrival.T + time_zero,
            )
        u = np.linspace(0.12, 1.0, 6)
        path_nodes = trajectory(first, u)[0] * 0.001
        remaining = 1000.0 * first[9] * (1.0 - first[10] * u)
        elapsed = (flight(1000.0 * first[9]) - flight(remaining)) / C
        kinetic = water["energy_at"](remaining)
        for point, birth in measured_origins(competition, expect, directions, triangulate=triangulate):
            distance = np.linalg.norm(point - path_nodes, axis=1)
            arrival = elapsed + 1000.0 * distance / C
            supported = (distance >= 0.12) & (arrival <= birth - time_zero + 0.8)
            if not supported.any():
                continue
            permitted = float(max(30.0, np.max(kinetic[supported])))
            fastest = float(np.min(arrival[supported]))
            relative = max(birth - time_zero, fastest)
            ridges = rank_at(
                competition, point, time_zero + relative, expect, directions, conditional=conditional
            )
            for _, axis, angle, pe, span in ridges[: 4 if electron_refit else 2]:
                for particle in dict.fromkeys((mass, "pion", "proton", "electron")):
                    target = material(particle)
                    for leg in leg_starts(beam, point, axis, angle, pe, span, particle):
                        endpoint = np.r_[point, leg]
                        energy = float(target["energy_at"](1000.0 * endpoint[9]))
                        if energy > permitted or hardware_margin(endpoint) <= 1e-5:
                            continue
                        light = (
                            coeff
                            * 1000.0
                            * endpoint[9]
                            * endpoint[10]
                            * receiver_values(model.pos, model.normals, endpoint, 4, particle).sum(0)
                        )
                        gain = model.obs.score(expect + light, model.ix)[0]
                        footprint.append((float(gain), first, endpoint, particle, relative, coeff, bg))
    return footprint


def evaluate(competition, model_record, incoming, ordinal, *, counted=True, extra_dimension=0):
    from physical_pulse import wavefront
    from source import raw

    _, prefix, leg, outgoing, relative, coeff, bg = model_record
    model = competition.model
    low = bg * model.bg
    rates = [low]
    clocks = []
    for coords, particle, clock in ((prefix, incoming, 0.0), (leg, outgoing, relative)):
        size = coeff * 1000.0 * coords[9] * coords[10]
        coarse = raw(model.pos, model.normals, coords, 16, False, particle)[0]
        fine = raw(model.pos, model.normals, coords, 32, False, particle)[0]
        arrival = wavefront(model.pos, model.normals, coords, particle, order=8)[0]
        low = low + size * coarse.sum(0)
        rates.extend(size * fine)
        clocks.extend(arrival + clock)
    rates = np.asarray(rates)
    amplitudes = rates.sum(1)
    if amplitudes[1] < 1.0e-8 or (counted and amplitudes[3] < 1.0e-8):
        return None
    difference = abs(model.obs.score(low, model.ix)[0] - model.obs.score(rates.sum(0), model.ix)[0])
    if difference > 0.06:
        return None
    limits = {}
    for j, k in ((1, slice(1, 3)), (2, slice(3, 5))):
        absolute = amplitudes[k].sum()
        limits[j] = (0.72 * absolute, 1.30 * absolute)
    cores = ((1,), (3,)) if counted else ((1,),)
    template = Template(
        f"detached_emission_{ordinal}",
        rates / np.maximum(amplitudes[:, None], 1e-300),
        np.column_stack(clocks),
        amplitudes,
        cores,
        np.zeros(4, int),
        22 + extra_dimension,
        limits,
        np.array([0, 1, 1, 2, 2]),
    )
    trial = competition.fit(template, iterations=24)
    if trial is not None:
        trial.update(
            start_mm=1000.0 * leg[:3],
            relative_birth_ns=float(relative),
            daughter_species=outgoing,
            quadrature_error=float(difference),
            incoming_path=prefix,
            daughter_path=leg,
        )
    return trial


def electron_rivals(competition, choices, incoming, *, split_tail=False):
    from charge_starts import weighted_quantile
    from material import material
    from triangulated_electron_refit import refine
    from value_only_receiver import receiver_values

    model = competition.model
    event = competition.event
    starts, alternatives, results = [], [], []
    q = np.minimum(event.charges_pe, 2.0)
    electron = material("electron")
    theta = np.arccos(1.0 / electron["index"])
    for rating, parent, daughter, particle, clock, response, background in choices:
        j = int(particle != "electron")
        if sum(mode == j for mode, _ in starts) >= 2 or any(
            j == mode and np.linalg.norm(daughter[:3] - d[:3]) < 0.16 and daughter[3:6] @ d[3:6] > 0.94
            for mode, d in starts
        ):
            continue
        starts.append((j, daughter))
        field = receiver_values(model.pos, model.normals, parent, 6, incoming).sum(0)
        mean = response * 1000.0 * parent[9] * parent[10] * field + background * model.bg
        root = daughter
        if j:
            face = receiver_values(model.pos, model.normals, daughter, 4, particle).sum(0)[model.ix]
            expected = response * 1000.0 * daughter[9] * daughter[10] * face
            weight = q * expected / np.maximum(expected + mean[model.ix], 1.0e-12)
            if weight.sum() < 14.0:
                continue
            radius = 0.001 * electron["range_at"](30.0 + 0.17 * weight.sum())
            ray = event.pmt_positions_mm * 0.001 - daughter[:3]
            axis = daughter[3:6]
            axial = ray @ axis
            transverse = np.linalg.norm(ray - axial[:, None] * axis, axis=1)
            position = weighted_quantile(axial - transverse / np.tan(theta), weight, [0.50])[0]
            vertex = daughter[:3] + (np.clip(position, -0.55, 0.55) - 0.25 * radius) * axis
            root = np.r_[vertex, axis, np.zeros(3), radius, 1.0, daughter[11]]
        alternatives.append((rating, parent, root, clock, response, background, mean))
        if len(starts) == 4:
            break
    for i, (rating, parent, root, clock, response, background, mean) in enumerate(alternatives):
        for mode in (False, True):
            preferred = refine(model, parent, root, clock, mean, response, incoming, below_margin=mode)
            if preferred is None:
                continue
            row = (rating, parent, preferred, "electron", clock, response, background)
            fit = evaluate(competition, row, incoming, f"electron_{i}", counted=not mode, extra_dimension=2)
            if fit is not None:
                label = "detached_submargin" if mode else "triangulated_refit"
                fit["label"] = f"{label}_electron_{i}"
                results.append(fit)
                if split_tail and not mode:
                    from lepton_stopping_turn import compare as stopping

                    results.extend(
                        stopping(competition, parent, preferred, clock, response, background, incoming, i)
                    )
    return results


def compare(competition, particle, optical, *, triangulate=False, electron_refit=False, split_tail=False):
    from coupled_model import Model
    from value_seed_beam_candidate import fit

    old = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        beam, state, _, record = fit(competition.event, competition.model.bg, particle, budget=56)
        if beam is None or record["count"] != 1:
            return [], 0
        parent = matched(competition, beam, state, ((0,),), "detached_emission_single")
        starts = scan(
            competition,
            beam,
            state,
            particle,
            optical,
            triangulate=triangulate,
            electron_refit=electron_refit,
        )
    finally:
        Model.delta_scale = old
    result = [] if parent is None else [parent]
    selected = []
    ordered = sorted(starts, key=lambda r: -r[0])
    for row in ordered:
        p, a = row[2][:3], row[2][3:6]
        if any(np.linalg.norm(p - v) < 0.16 and a @ b > 0.96 for v, b in selected):
            continue
        selected.append((p, a))
        fit = evaluate(competition, row, particle, len(selected))
        if fit is not None:
            if triangulate:
                fit["label"] = "triangulated_" + fit["label"]
            result.append(fit)
        if triangulate and row[3] == "electron":
            from material import material

            rate, parent, daughter, species, clock, response, background = row
            reduced = daughter.copy()
            reduced[9] = min(reduced[9], 0.999e-3 * material(species)["margin_mm"])
            control = evaluate(
                competition,
                (rate, parent, reduced, species, clock, response, background),
                particle,
                f"submargin_{len(selected)}",
                counted=False,
            )
            if control is not None:
                result.append(control)
        if len(selected) == 3:
            break
    if electron_refit:
        result.extend(electron_rivals(competition, ordered, particle, split_tail=split_tail))
    return result, len(starts)
