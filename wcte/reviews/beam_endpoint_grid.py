"""Seed two charged sections from independent, measured beam-prefix candidates."""

from functools import lru_cache, partial

import numpy as np

from finite_chains import factory, refine


@lru_cache(maxsize=2)
def library(particle):
    """Render surveyed central straight paths at fixed emitting depths."""
    from charge_starts import allowed_chord
    from material import material
    from staged_physics.primary import Aperture
    from value_only_receiver import receiver_values
    from wctering.likelihood.geometry import active_geometry

    ids, positions, normals = active_geometry()
    water = material(particle)
    entry, axis = np.array([0.0, 0.0, Aperture().z]), np.array([0.0, 0.0, 1.0])
    extent = allowed_chord(entry, axis)
    if extent is None or extent[0] > 1e-5:
        return ids, ()
    prior = []
    for length in (
        0.022,
        0.045,
        0.074,
        0.12,
        0.19,
        0.28,
        0.40,
        0.55,
        0.74,
        0.95,
        1.19,
        1.47,
        1.78,
        2.10,
        2.42,
    ):
        if length > 0.997 * extent[1]:
            continue
        # Fit a residual visible range after the turn. This grid is common
        # to every event. No beam energy label sets an available range.
        for left in (0.015, 0.055, 0.13, 0.24, 0.40, 0.72, 1.15, 1.8, 2.9, 4.4):
            distance = length + left + 0.001 * water["margin_mm"]
            if distance * 1000.0 >= water["maximum_range_mm"]:
                continue
            path = np.r_[entry, axis, np.zeros(3), distance, length / distance, np.log(np.radians(1.6))]
            values = 1000.0 * length * receiver_values(positions, normals, path, 3, particle).sum(0)
            if values.sum() > 1e-12:
                prior.append((path, values))
    return ids, tuple(prior)


def calibrated_paths(competition, particle, guess):
    from charge_starts import allowed_chord
    from material import material
    from staged_physics.primary import Aperture
    from value_only_receiver import receiver_values

    measured_length, measured_range = guess
    if min(measured_length, measured_range) <= 0.0:
        return ()
    origin, axis = np.array([0.0, 0.0, Aperture().z]), np.array([0.0, 0.0, 1.0])
    water_path = allowed_chord(origin, axis)
    if water_path is None or water_path[0] > 1e-5:
        return ()
    table = material(particle)
    model = competition.model
    choices = []
    for scale, range_offset in ((0.82, -0.08), (1.0, 0.0), (1.17, 0.10)):
        length = float(np.clip(measured_length * scale, 0.017, 0.995 * water_path[1]))
        end = max(measured_range + range_offset, length + 0.001 * table["margin_mm"] + 0.012)
        if 1000.0 * end >= table["maximum_range_mm"]:
            continue
        p = np.r_[origin, axis, np.zeros(3), end, length / end, np.log(np.radians(1.6))]
        y = 1000.0 * length * receiver_values(model.pos, model.normals, p, 4, particle).sum(0)
        choices.append((p, y))
    return choices


def nominate(competition, optical, particle, geometry_seed=None):
    """Choose depth basins while allowing an independent observed lobe."""
    model = competition.model
    active, trials = library(particle)
    if not np.array_equal(active, model.active):
        return []
    extra = () if geometry_seed is None else calibrated_paths(competition, particle, geometry_seed)
    fitted = [r for r in optical if len(r[2].core_groups) in (1, 2)]
    bg_amp = max(1.0, min((r[2].amplitudes[0] for r in fitted), default=0.08 * model.total))
    bg = bg_amp * model.bg
    candidate_fields = [np.zeros_like(bg)]
    for _, _, t in fitted:
        for j in np.argsort(t.amplitudes[1:])[::-1]:
            pre = t.amplitudes[j + 1] * t.shapes[j + 1]
            if not any(np.minimum(pre, v).sum() > 0.88 * pre.sum() for v in candidate_fields):
                candidate_fields.extend([0.7 * pre, pre])
    candidates = []
    q = np.minimum(model.q, 2.0)
    selected_extra = []
    for i, (path, values) in enumerate((*trials, *extra)):
        beam = model.response * values
        if np.sum(beam[model.ix] / (beam[model.ix] + 0.45)) < 21.0:
            continue
        saved = None
        for separate in candidate_fields:
            if separate.sum() > model.total or (beam + separate).sum() > 1.62 * model.total:
                continue
            base = beam + bg + separate
            expected = q @ (beam[model.ix] / np.maximum(base[model.ix], 1e-15))
            if expected < 21.0:
                continue
            value = model.obs.score(base, model.ix)[0]
            if saved is None or value > saved[0]:
                saved = (float(value), path, beam + bg, bg_amp)
        if saved is not None:
            collection = candidates if i < len(trials) else selected_extra
            collection.append(saved)
    good = []
    for row in sorted(candidates, key=lambda r: -r[0]):
        if any(abs(row[1][9] * row[1][10] - p[1][9] * p[1][10]) < 0.09 for p in good):
            continue
        good.append(row)
        if len(good) == 6:
            break
    for row in sorted(selected_extra, key=lambda r: -r[0])[:2]:
        if not any(np.linalg.norm(row[1][9:11] - old[1][9:11]) < 0.04 for old in good):
            good.append(row)
    return good


def measured_ridges(competition, path, baseline, background, particle, geometries):
    from charge_starts import weighted_quantile
    from junction_residual_seeds import directions, ridge_scores
    from physical_pulse import wavefront
    from terminal_track_seeds import ridge_peaks

    model, event = competition.model, competition.event
    start, axis = path[:3], path[3:6]
    end = start + axis * path[9] * path[10]
    timing, _, elapsed = wavefront(model.pos[model.ix], model.normals[model.ix], path, particle, order=5)
    _, gradient = model.obs.score(baseline, model.ix)
    excess = np.clip(-gradient[model.ix] * baseline[model.ix], -0.8, 2.5)
    charge = np.minimum(event.charges_pe, 2.0)
    times = event.times_ns - np.median(event.times_ns)
    fraction = 1.0 - np.minimum(1.0, background * model.bg[model.ix] / np.maximum(baseline[model.ix], 1e-15))
    origin = weighted_quantile(times - timing[0], charge * fraction, [0.5])[0] + elapsed
    delta = event.pmt_positions_mm - 1000.0 * end
    distance = np.linalg.norm(delta, axis=1)
    known = [np.asarray(a) for g in geometries for a in g["axes"]]
    axes = np.vstack([directions(), known]) if known else directions()
    axes = axes[axes @ axis < np.cos(np.radians(13.0))]
    windows = (90.0, 220.0, 545.0)
    scores = ridge_scores(delta, distance, axes, times, charge, excess, origin, windows)
    modes = []
    for size, d in zip(windows, scores, strict=True):
        for support, axis, opening, span in ridge_peaks(
            d[:, :, 1],
            axes,
            delta,
            times,
            origin,
            charge,
            excess,
            size + 95.0,
            (0.024, (size + 95.0) * 0.001),
        ):
            modes.append((support / np.sqrt(1.0 + size / 300.0), axis, opening, support, span))
    kept = []
    for row in sorted(modes, key=lambda r: -r[0]):
        if any(row[1] @ t[1] > 0.976 for t in kept):
            continue
        kept.append(row)
        if len(kept) == 3:
            break
    return end, kept


def compare(
    competition,
    particle,
    optical,
    *,
    receiver_screen=False,
    preserve_species=False,
    geometry_seed=None,
    profile_parent=False,
    first_pulse=None,
    lineage_union=False,
    selected_gate=False,
    cone_moments=False,
    starts_per_family=2,
):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from chain_sources import paths
    from serial_refinement import leg_starts
    from source import trajectory
    from value_only_receiver import seed_log_likelihood

    chart = {}
    shortlist = []
    measured = [g for _, g, _ in optical]
    model = competition.model
    ridge = measured_ridges
    label = "beam_endpoint_grid_"
    if receiver_screen or profile_parent or selected_gate:
        from aperture_receiver_ridges import peaks

        ridge = partial(peaks, profile_parent=profile_parent, selected_gate=selected_gate)
        label = "profiled_aperture_" if profile_parent else "aperture_receiver_"
        if selected_gate:
            label = "selected_aperture_"
    if preserve_species:
        label = "species_balanced_"
    if geometry_seed is not None:
        label = "calibrated_aperture_"
    if first_pulse is not None:
        from first_pulse_ridges import peaks

        ridge = partial(peaks, include_pairs=first_pulse == "pairs")
        label = f"first_pulse_{first_pulse}_"
    if cone_moments:
        from finite_cone_moments import peaks

        ridge, label = peaks, "moment_aperture_"
    for _, parent, expectation, bg in nominate(competition, optical, particle, geometry_seed):
        point, ridges = ridge(competition, parent, expectation, bg, particle, measured)
        best = {}
        for j, (_, axis, opening, amplitude, span) in enumerate(ridges):
            for child in dict.fromkeys((particle, "pion", "proton", "electron")):
                reaction = particle, child
                if reaction not in chart:
                    chart[reaction] = factory(competition.event, model.bg, reaction, 4)
                line = chart[reaction]
                for seed in leg_starts(line, point, axis, opening, amplitude, span, child):
                    x = canonical(line, np.r_[parent, seed, np.log([model.response, bg])])
                    if x is None:
                        continue
                    x = repair_energy(line, x)
                    if x is None or not feasible(line, x) or line.hardware_margin(x) <= 1e-5:
                        continue
                    likelihood = seed_log_likelihood(line, x)
                    basin = (j, child) if preserve_species else j
                    if basin not in best or likelihood > best[basin][0]:
                        best[basin] = (likelihood, reaction, x, parent[9] * parent[10])
        shortlist.extend(best.values())
    ranked = sorted(shortlist, key=lambda x: -x[0])
    if preserve_species:
        native = [r for r in ranked if r[1][1] == particle]
        alternatives = [r for r in ranked if r[1][1] != particle]
        trials = distinct(native, starts_per_family) + distinct(alternatives, starts_per_family)
    else:
        trials = distinct(ranked, 3)
    output = []
    for test in trials:
        r = refine(competition, test, lineage_union=True) if lineage_union else refine(competition, test)
        if r is None or (r["count"] != 2 and not lineage_union):
            continue
        incoming, child = [p for p, _ in paths(r["parameters"][:-2])]
        tangent = trajectory(incoming, np.ones(1))[1][0]
        turns = np.degrees(np.arccos(np.clip(tangent @ child[3:6], -1.0, 1.0)))
        r.update(
            label=r["label"] if lineage_union and not cone_moments else label + r["label"],
            turn_deg=float(turns),
            initial_length_m=test[3],
        )
        output.append(r)
    return output, len(shortlist)


def distinct(ranked, budget):
    selected = []
    for candidate in ranked:
        if any(
            abs(candidate[3] - old[3]) < 0.10 and candidate[2][12:15] @ old[2][12:15] > 0.95
            for old in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == budget:
            break
    return selected
