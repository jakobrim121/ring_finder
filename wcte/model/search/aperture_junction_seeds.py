"""Find a possible scattering point even when both optical cones miss it."""

import numpy as np

from balanced_search import feasible
from charge_starts import allowed_chord, recoil_start, weighted_quantile
from coupled_model import Model as Coupled
from junction_residual_seeds import directions, ridge_scores
from physical_pulse import wavefront
from serial_refinement import leg_starts
from terminal_track_seeds import ridge_peaks
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE
from wcte.counting import track_counting


def entrance_intervals(event, geometry, port):
    """Rank finite incoming intervals by the light that fits one clock."""
    origin = np.array([0.0, 0.0, port.z])
    axes = [np.array([0.0, 0.0, 1.0])]
    # Try the measured forward directions as well. The fitted two-track
    # chart moves the direction and point within the surveyed aperture.
    for axis in geometry["axes"]:
        unit = np.asarray(axis) / np.linalg.norm(axis)
        if unit[2] > 0.0 and all(unit @ other < 0.997 for other in axes):
            axes.append(unit)
    relative = event.times_ns - np.median(event.times_ns)
    charge = np.minimum(event.charges_pe, 2.0)
    delta = event.pmt_positions_mm - 1000.0 * origin
    records = []
    spans = np.array([35.0, 75.0, 145.0, 260.0, 450.0, 735.0, 1150.0, 1650.0])
    angles = np.deg2rad([24.0, 29.0, 33.0, 36.5, 39.5, 41.5])
    for axis in axes:
        water = allowed_chord(origin, axis)
        if water is None or water[0] > 0.0 or water[1] < 0.04:
            continue
        z = delta @ axis
        r = np.linalg.norm(delta - z[:, None] * axis, axis=1)
        lengths = np.unique(np.minimum(spans, 1000.0 * water[1]))
        for angle in angles:
            s = z - r / np.tan(angle)
            flight = (N_PHASE * np.cos(angle) * s + N_GROUP * r / np.sin(angle)) / C
            distance = s[:, None] - np.clip(s[:, None], 0.0, lengths)
            sigma = np.maximum(20.0, np.deg2rad(2.4) * r / np.sin(angle) ** 2)
            near = np.exp(-0.5 * (distance / sigma[:, None]) ** 2)
            residual = relative - flight
            for span, weight in zip(lengths, near.T):
                estimate = charge * weight
                if estimate.sum() < 23.0:
                    continue
                clock = weighted_quantile(residual, estimate, [0.5])[0]
                strength = estimate @ np.exp(-0.5 * ((residual - clock) / 1.7) ** 2)
                if strength > 20.0:
                    # Longer intervals subtend more accidental charge.
                    records.append((strength / np.sqrt(80.0 + span), axis, span, angle))
    used = []
    for _, axis, span, angle in sorted(records, key=lambda row: -row[0]):
        if any(axis @ a > 0.985 and abs(span - l) < 0.18 * (span + l) for a, l in used):
            continue
        used.append((axis, span))
        yield origin, axis, span / 1000.0, angle
        if len(used) == 6:
            break


def nominate(event, model, geometry, amplitude, port, seed_parent):
    """Look for a charge and time ridge from each possible entrance endpoint."""
    args = event.pmt_ids, event.charges_pe, model.bg
    normalization = np.log([model.response, max(amplitude[0], 1e-8)])
    q = np.minimum(event.charges_pe, 2.0)
    time = event.times_ns - np.median(event.times_ns)
    grid = directions()
    windows = (95.0, 220.0, 450.0, 850.0)
    trials, singles, pairs = [], {}, {}
    for origin, axis, span, angle in entrance_intervals(event, geometry, port):
        join = origin + span * axis
        delta = event.pmt_positions_mm - 1000.0 * join
        distance = np.linalg.norm(delta, axis=1)
        allowed = grid[grid @ axis < track_counting.get().merge_cosine]
        for mass in ("pion", "proton"):
            if mass not in singles:
                singles[mass] = Coupled(*args, (mass,), 4, 4)
            parent = singles[mass]
            start = next(seed_parent(origin, join, mass, angle), None)
            if start is None or parent.hardware_margin(np.r_[start, normalization]) <= 1e-5:
                continue
            state = parent(np.r_[start, normalization], True)
            core = state["posterior"][1]
            observed = 1.0 - parent.obs.no_source(state["means_hit"], core.clip(0.0, 1.0))
            if observed.sum() < 22.0:
                continue
            arrival, _, elapsed = wavefront(event.pmt_positions_mm, model.normals[model.ix], start, mass)
            clock = weighted_quantile(time - arrival[0], q * observed, [0.5])[0] + elapsed
            mean = np.exp(normalization[0]) * parent.primary_cache[1] + np.exp(normalization[1]) * model.bg
            _, slope = model.obs.score(mean, model.ix)
            excess = np.clip(-slope[model.ix] * mean[model.ix], -0.7, 2.0)
            score = ridge_scores(delta, distance, allowed, time, q, excess, clock, windows)
            for window, scan in zip(windows, score):
                if scan[:, :, 1].max(initial=0.0) < 14.0:
                    continue
                for support, daughter, opening, length in ridge_peaks(
                    scan[:, :, 1],
                    allowed,
                    delta,
                    time,
                    clock,
                    q,
                    excess,
                    window + 90.0,
                    (0.016, window / 1000.0 + 0.08),
                ):
                    if support < track_counting.get().scaled_support(14.0):
                        continue
                    for species in dict.fromkeys((mass, "pion", "proton")):
                        particles = mass, species
                        if particles not in pairs:
                            pairs[particles] = Coupled(*args, particles, 4, 4)
                        pair = pairs[particles]
                        for child in leg_starts(pair, join, daughter, opening, support, length, species):
                            x = recoil_start(pair, np.r_[start, child, normalization])
                            if x is None or not feasible(pair, x) or pair.hardware_margin(x) <= 1e-5:
                                continue
                            candidate = pair(x, True)
                            visible = [
                                np.sum(1.0 - pair.obs.no_source(candidate["means_hit"], f.clip(0.0, 1.0)))
                                for f in candidate["posterior"][1:]
                            ]
                            if min(visible) >= track_counting.get().scaled_support(24.0):
                                trials.append((candidate["ll"], particles, x))
    return trials
