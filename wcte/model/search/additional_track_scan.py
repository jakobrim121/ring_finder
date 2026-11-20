"""Search the measured excess at possible downstream interaction points."""

import numpy as np

from analog_optical import contain_finite
from aperture_score_starts import cone_scores, measurements
from chain_sources import paths
from charge_probe import receiver_values, secondaries
from charge_starts import allowed_chord
from detached_conversion import residual_vertices
from junction_residual_seeds import directions
from joined_segments import arc
from module_optics import shapes
from physical_pulse import components, flight_integral, profile, wavefront
from qualified_optical_tracks import paths_for_cone
from source import trajectory
from wctering.seeding.coherent_timing import C


def locations(model, state, event, mean, slope):
    """Include known charged arcs and independent excess pulse vertices."""
    origin = profile(event, model, state, separate_knockon=True, timing_only=True)
    if origin is None:
        return
    clock = origin["clock_ns"]
    tracks = [p for p, _ in paths(state["parameters"][:-2])]
    for path, mass in zip(tracks, model.particles):
        span = path[9] * path[10]
        u = np.array([0.45, 0.72, 1.0])
        at, axis = trajectory(path, u)[:2]
        primitive = flight_integral(mass)
        initial = 1000.0 * path[9]
        remaining = np.maximum(initial - 1000.0 * span * u, 0.0)
        delays = (primitive(initial) - primitive(remaining)) / C
        for point, unit, elapsed in zip(at, axis, delays):
            yield point / 1000.0, unit, clock + elapsed
        clock += delays[-1]
    i = model.ix
    weights = np.clip(-slope[i] * mean[i], 0.0, 1.0)
    if np.count_nonzero(weights > 0.25) >= 20:
        for point in residual_vertices(event, weights, count=2):
            yield point, None, 0.0


def cones(compare, incumbent, restored):
    """Return a small set of third wavefronts from an all-PMT score scan."""
    fit, state, _ = restored
    model, event = compare.model, compare.event
    mean = shapes(fit, state)
    _, slope = model.obs.score(mean, model.ix)
    time, sigma, hits, derivative = measurements(compare, incumbent.rates, incumbent.times)
    direction_grid = directions()[::2]
    spans = np.array([55.0, 150.0, 330.0])
    openings = np.deg2rad([25.0, 32.0, 37.0, 41.0])
    ranked = []
    for point, tangent, clock in locations(fit, state, event, mean, slope):
        # The light must reveal an independent direction. Off-trajectory
        # candidates have no local charged tangent or shared flight clock.
        grid = (
            direction_grid
            if tangent is None
            else direction_grid[direction_grid @ tangent < np.cos(np.deg2rad(15.0))]
        )
        score = cone_scores(
            model.pos - 1000.0 * point,
            model.normals,
            grid,
            openings,
            spans,
            time,
            sigma,
            hits,
            compare.gate,
            clock,
            -slope,
            derivative,
            mean,
            timing=tangent is not None,
        )
        top = []
        for idx in np.argsort(score.ravel())[::-1]:
            i, j, k = np.unravel_index(idx, score.shape)
            merit, axis, theta = score[i, j, k], grid[i], openings[j]
            if merit < 12.0 or len(top) == 2:
                break
            if any(axis @ a > 0.96 for a in top):
                continue
            chord = allowed_chord(point, axis)
            if chord is None or chord[0] > 0.001 or chord[1] < 0.016:
                continue
            length = min(spans[k] / 1000.0, chord[1])
            cone = np.r_[point + 0.5 * length * axis, axis, length, theta, np.log(np.deg2rad(2.8))]
            try:
                contain_finite(cone)
            except (ValueError, ArithmeticError):
                continue
            top.append(axis)
            ranked.append((merit, cone, tangent is not None))
    chosen = []
    for merit, cone, connected in sorted(ranked, key=lambda row: -row[0]):
        if any(np.linalg.norm(cone[:3] - old[:3]) < 0.14 and cone[3:6] @ old[3:6] > 0.96 for old in chosen):
            continue
        chosen.append(cone)
        yield merit, cone, connected
        if len(chosen) == 4:
            return


def nominate(compare, incumbent, restored):
    """Seed a calibrated charged section at a selected optical residual."""
    for merit, cone, _ in cones(compare, incumbent, restored):
        photons = min(max(np.sqrt(merit) * 5.0, 20.0), compare.model.total)
        yield from paths_for_cone(compare.model, cone, photons)


def endpoint_cones(compare, restored, *, shared=None):
    """Find a direction using the residual after a tested physical endpoint."""
    fit, state, _ = restored
    model, event = compare.model, compare.event
    pair = [p for p, _ in paths(state["parameters"][:-2])]
    response, background = np.exp(state["parameters"][-2:])
    base = background * model.bg
    base_diffuse = base[model.ix].copy()
    if shared is None:
        rate, centers = components(event, fit, state, separate_knockon=True)
        timing = profile(event, fit, state, precomputed=(rate, centers), timing_only=True)
        if timing is None:
            return
        shared = timing["clock_ns"]
    origin = shared
    i = model.ix
    base_rate, base_center = [], []
    openings = np.deg2rad([23.0, 29.0, 34.0, 38.0, 41.5])
    spans = np.array([65.0, 145.0, 270.0, 520.0])
    grid = directions()
    for section, (p, mass) in enumerate(zip(pair, fit.particles)):
        length = p[9] * p[10]
        scores = []
        ends = np.unique(np.r_[[0.36, 0.56, 0.76, 0.93, 1.0], 1.0 + np.array([0.065, 0.135]) / length])
        for u in ends:
            prefix = arc(p, 0.0, u)
            if prefix is None or model.hardware_margin(np.r_[prefix, state["parameters"][-2:]]) <= 1e-5:
                continue
            tail, tail_rate, elapsed = wavefront(event.pmt_positions_mm, model.normals[i], prefix, mass)
            core, soft = (
                response * (1000.0 * u * length) * receiver_values(model.pos, model.normals, prefix, 4, mass)
            )
            recoil = response * fit.delta_scale * secondaries(model.pos, model.normals, prefix, mass, 4)
            soft_rate = np.minimum(response * tail_rate[1], soft[i])
            # Keep any diffuse excess and the completed earlier legs.
            # Scanning the clipped null frees light which would otherwise
            # be explained by extending the wrong charged heading.
            diffuse = base_diffuse + recoil[i] + soft[i] - soft_rate
            centers_at_join = np.column_stack([*base_center, *(tail + origin)])
            rates = np.column_stack([diffuse, *base_rate, core[i], soft_rate])
            mean = base + core + soft + recoil
            _, slope = model.obs.score(mean, i)
            time, sigma, hits, derivative = measurements(compare, rates, centers_at_join)
            point, tangent = trajectory(prefix, np.ones(1))[:2]
            point, tangent = point[0], tangent[0]
            dirs = grid[grid @ tangent < np.cos(np.deg2rad(15.0))]
            args = (
                model.pos - point,
                model.normals,
                dirs,
                openings,
                spans,
                time,
                sigma,
                hits,
                compare.gate,
                origin + elapsed,
                -slope,
                derivative,
                mean,
            )
            # Reserve a start for a wavefront with timing evidence on its
            # own. Inclusion in the shortlist changes neither the model
            # score nor the support required for a counted ring.
            for timing_only in (False, True):
                values = cone_scores(*args, time_only=timing_only)
                kept = []
                for flat in np.argsort(values.ravel())[::-1]:
                    a, b, c = np.unravel_index(flat, values.shape)
                    merit, axis = values[a, b, c], dirs[a]
                    if merit < 12.0 or len(kept) == 2:
                        break
                    if any(axis @ other > 0.96 for other in kept):
                        continue
                    chord = allowed_chord(point / 1000.0, axis)
                    if chord is None or chord[0] > 0.001 or chord[1] < 0.016:
                        continue
                    span = min(spans[c] / 1000.0, chord[1])
                    kept.append(axis)
                    scores.append(
                        (
                            timing_only,
                            merit,
                            prefix,
                            point / 1000.0,
                            axis,
                            openings[b],
                            span,
                            origin + elapsed,
                        )
                    )
        for timing_only in (False, True):
            kept = []
            for _, merit, prefix, point, axis, opening, span, birth in sorted(
                (r for r in scores if r[0] == timing_only), key=lambda r: -r[1]
            ):
                if any(axis @ a > 0.96 and np.linalg.norm(point - b) < 0.10 for a, b in kept):
                    continue
                kept.append((axis, point))
                yield section, timing_only, merit, prefix, point, axis, opening, span, birth
                if len(kept) == 5:
                    break
        core, soft = response * (1000.0 * length) * receiver_values(model.pos, model.normals, p, 4, mass)
        recoil = response * fit.delta_scale * secondaries(model.pos, model.normals, p, mass, 4)
        arrival, light, delay = wavefront(event.pmt_positions_mm, model.normals[i], p, mass)
        soft_rate = np.minimum(response * light[1], soft[i])
        base += recoil + soft + core
        base_diffuse += recoil[i] + soft[i] - soft_rate
        # Separate the earlier charged light from this prefix's diffuse
        # rate while keeping its exact total charge in the null model.
        base_rate.extend([core[i], soft_rate])
        base_center.extend(arrival + origin)
        origin += delay
