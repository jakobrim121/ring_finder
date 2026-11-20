"""Test a beam track and a displaced electron against a degenerate third cone."""

from dataclasses import dataclass

import numpy as np

from aperture_geometry_fit import ray_times
from chain_comparison import MARGIN
from charge_probe import receiver_values, secondaries
from charge_starts import allowed_chord, weighted_quantile
from compare_count_models import Comparison
from detached_conversion import Timing
from detached_track_fit import refine
from entrance_track import Aperture
from finite_residual_fit import optical_shape
from gated_optical_fit import fit as optical_fit
from geometry import hardware_margin
from material import material
from physical_pulse import wavefront
from qualified_optical_tracks import paths_for_cone
from residual_joint_optics import path_from_geometry
from transport_count_review import plausible_dropout


@dataclass(frozen=True)
class Source:
    material: str
    path: np.ndarray
    core: np.ndarray
    light: np.ndarray


def minimum_opening():
    """Smallest reference Cherenkov angle above the frozen energy margin."""
    # For a fixed kinetic-energy margin, the slowest countable particle
    # is the heaviest modeled species. The wavelength-averaged angle is a
    # seed diagnostic. The complete light model still integrates dispersion.
    water = material("proton")
    gamma = 1.0 + water["energy_at"](water["margin_mm"]) / water["mass"]
    return float(np.arccos(1.0 / (water["index"] * np.sqrt(1.0 - gamma**-2))))


def physical_source(model, species, path):
    if hardware_margin(path) <= 1e-5:
        return None
    length = 1000.0 * path[9] * path[10]
    core, tail = length * receiver_values(model.pos, model.normals, path, 4, species)
    if core[model.ix].sum() <= 1e-12:
        return None
    light = core + tail + secondaries(model.pos, model.normals, path, species, 4)
    return Source(species, path, model.response * core, model.response * light)


def parent_starts(model, event, cone, weight, port):
    """Use the apparent emission coordinates to initialize the water chord."""
    entrance = np.array([0.0, 0.0, port.z])
    normal = np.array([0.0, 0.0, 1.0])
    observed = cone[3:6]
    axes = [normal]
    if observed @ normal < 0.999 and observed[2] > 0.0:
        axes.append(observed)
    maximum_angle = np.arccos(1.0 / material("electron")["index"]) - 0.008
    for axis in axes:
        water = allowed_chord(entrance, axis)
        if water is None or water[0] > 0.0 or water[1] <= 0.02:
            continue
        for theta in np.unique(
            np.clip(cone[7] + np.deg2rad([-4.0, 0.0, 4.0]), minimum_opening(), maximum_angle)
        ):
            emission, _, _ = ray_times(event.pmt_positions_mm / 1000.0, entrance, axis, theta)
            inside = (emission >= -0.06) & (emission <= water[1] + 0.06)
            if weight @ inside < 30.0:
                continue
            span = weighted_quantile(emission[inside], weight[inside], [0.9])[0]
            span = np.clip(span, 0.02, water[1])
            for species in ("pion", "proton", "muon"):
                table = material(species)
                speed = 1.0 / (table["index"] * np.cos(theta))
                energy = table["mass"] * (1.0 / np.sqrt(1.0 - speed**2) - 1.0)
                energy = np.clip(energy, table["energy"][0] + 1.0, table["energy"][-1])
                remaining = float(table["range_at"](energy)) / 1000.0
                maximum = table["maximum_range_mm"] / 1000.0
                for length, ran in (
                    (span, remaining + 0.5 * span),
                    (span, span + table["margin_mm"] / 1000.0),
                ):
                    ran = min(maximum, ran)
                    if ran <= length:
                        continue
                    path = np.r_[entrance, axis, np.zeros(3), ran, length / ran, np.log(np.deg2rad(1.5))]
                    source = physical_source(model, species, path)
                    if source is not None:
                        yield source


def electron_starts(model, regions):
    """Retain phototube-supported broad ridges with independent origins."""
    for cone, photons in regions:
        if cone[7] + np.exp(cone[8]) < np.deg2rad(39.0):
            continue
        for species, path in paths_for_cone(model, cone, photons):
            if species != "electron":
                continue
            source = physical_source(model, species, path)
            if source is not None:
                yield source


def rank(compare, fractions, basis, amplitudes, cones, bank_two, port):
    model, event = compare.model, compare.event
    opening = np.array([cone[7] for cone in cones])
    q = compare.q
    primary = np.argmax(
        (q @ fractions[:, 1:])
        * (opening >= minimum_opening())
        * np.maximum([cone[5] for cone in cones], 0.0) ** 2
    )
    if cones[primary][5] <= 0.0 or opening[primary] < minimum_opening():
        return []
    parent = list(parent_starts(model, event, cones[primary], q * fractions[:, primary + 1], port))
    if not parent:
        return []
    backgrounds = amplitudes[0] * basis[0]
    measured = amplitudes @ basis
    old_light = measured - amplitudes[primary + 1] * basis[primary + 1]
    parent.sort(key=lambda p: -model.obs.score(old_light + p.light, model.ix)[0])
    regions = [(p, s) for j, (p, s) in enumerate(zip(cones, amplitudes[1:])) if j != primary]
    if bank_two is not None:
        scale = amplitudes.sum()
        for j, strength in enumerate(bank_two["weights"][1:]):
            try:
                cone = path_from_geometry(bank_two, j)
            except (ArithmeticError, ValueError):
                continue
            if cone[3:6] @ cones[primary][3:6] < 0.94:
                regions.append((cone, scale * strength))
    electrons = list(electron_starts(model, regions))
    if not electrons:
        return []
    best = []
    # Distinct masses and lepton origins are useful restarts. Do not
    # spend multiple full fits on the same shallow angular offset.
    for incoming in parent[:8]:
        for lepton in electrons:
            means = backgrounds + incoming.light + lepton.light
            ll = model.obs.score(means, model.ix)[0]
            best.append((ll, incoming, lepton))
    selected = []
    for ll, incoming, lepton in sorted(best, key=lambda row: -row[0]):
        if any(
            incoming.material == a.material
            and np.linalg.norm(a.path[:3] - incoming.path[:3]) < 0.1
            and np.linalg.norm(b.path[:3] - lepton.path[:3]) < 0.16
            for _, a, b in selected
        ):
            continue
        selected.append((ll, incoming, lepton))
        if len(selected) == 3:
            break
    return selected


def refitted_control(compare, geometry, budget):
    """Give the larger count a refit with a countable optical angle."""
    model = compare.model
    fit = optical_fit(
        compare.event, geometry, model.bg, budget, strength=1.0, minimum_opening=minimum_opening()
    )
    if fit is None:
        return None
    _, share, info = fit
    try:
        paths = [path_from_geometry(info["geometry"], j) for j in range(3)]
    except (ArithmeticError, ValueError):
        return None
    control = dict(paths=paths, photons=info["photons"], count=3, family="bounded_three_optics")
    return compare.finite(control, share)


def review(event, model, geometry, basis, amplitudes, winner, bank_two, budget, *, bank_three=None):
    """Require a calibrated, independently timed pair before reducing count."""
    if (
        winner[2]["family"] != "released"
        or len(basis) != 4
        or event.timing_selection["prompt_window_ns"] is None
        or (geometry.get("emission_model") == "full_water_line")
        or (min(geometry["openings_deg"]) >= np.rad2deg(minimum_opening()))
    ):
        return (None, [])
    try:
        cones = [path_from_geometry(geometry, j) for j in range(3)]
    except (ArithmeticError, ValueError):
        return (None, [])
    port, compare = (Aperture(), Comparison(event, model))
    modes = rank(compare, winner[1], basis, amplitudes, cones, bank_two, port)
    if not modes:
        return (None, [])
    old = compare.optical(geometry, basis, amplitudes, winner[1], winner[2])
    if old is None:
        return (None, [])
    timing = Timing.from_event(event)
    normalization = np.log([model.response, amplitudes[0]])
    best, reports, rival, prediction = (None, [], None, None)
    # Use the converged coarse charge profile to screen remote geometries before the timing fit.
    charge_floor = model.obs.score(amplitudes @ basis, model.ix)[0] - 24.0 * np.sqrt(model.total)
    for start_score, primary, electron in modes:
        from chord_profile import profile

        rows = np.vstack([basis[0], primary.light, electron.light])
        norm = rows.sum(1)
        charge, _, residual = profile(
            rows / norm[:, None], model.obs, model.ix, np.r_[amplitudes[0], norm[1:]], budget=18
        )
        if residual <= 1e-05 and charge < charge_floor:
            continue
        tracks = (primary, electron)
        flights = [
            wavefront(event.pmt_positions_mm, model.normals[model.ix], p.path, p.material)[0][0]
            for p in tracks
        ]
        clock = np.array(
            [
                weighted_quantile(compare.time - t, compare.q * p.core[model.ix], [0.5])[0]
                for t, p in zip(flights, tracks)
            ]
        )
        fitted = refine(
            event,
            model,
            [p.path for p in tracks],
            [p.material for p in tracks],
            normalization,
            timing,
            clock,
            port,
            budget,
        )
        if fitted is None:
            continue
        mean, rates, spread, shares, record = fitted
        fractions = np.column_stack([1.0 - shares.sum(1), shares])
        record.update(family="aperture_lepton_pair", count=2, supported=False, charge_start=start_score)
        measured = compare.measured(fractions, record, mean, rates, spread, 23, (1, 4))
        delta = 2.0 * (measured.joint - old.joint)
        gain = float(measured.score - old.score - delta.max(initial=0.0))
        record["gain"] = gain
        reports.append(record)
        if gain <= MARGIN or set(fractions.argmax(1)) - {0} != {1, 2}:
            continue
        supported, evidence = plausible_dropout(compare, old, measured, mean, basis, amplitudes)
        record.update(evidence)
        partner_fit = None
        if not supported:
            from converted_lepton_pair import candidate, coverage

            converted = candidate(
                event, model, compare, primary.material, record["parameters"], port, old, budget
            )
            if converted is not None:
                converted_gain, trial, extended_mean, details = converted
                supported, evidence = coverage(
                    compare, old, trial, extended_mean, amplitudes @ basis, predictive=True
                )
                record["partner_fit"] = dict(details, gain=converted_gain, evidence=evidence)
                if converted_gain > gain and supported:
                    gain, measured, mean = (converted_gain, trial, extended_mean)
                    fractions = measured.fractions
                    partner_fit = details
                else:
                    supported = False
        if not supported:
            continue
        if rival is None:
            controls = [refitted_control(compare, geometry, budget)]
            if bank_three is not None:
                controls.append(refitted_control(compare, bank_three, budget))
            rival = max((fit for fit in controls if fit is not None), key=lambda fit: fit.score, default=None)
            if rival is None:
                break
        delta = 2.0 * (measured.joint - rival.joint)
        matched = float(measured.score - rival.score - delta.max(initial=0.0))
        record["matched_gain"] = matched
        if matched <= MARGIN:
            continue
        if partner_fit is not None:
            from converted_lepton_pair import candidate
            from module_predictive import ModulePredictive

            if prediction is None:
                scoring = ModulePredictive(event, model, amplitudes[0])
                prediction = [
                    scoring.optical(geometry, basis, amplitudes, winner[1], winner[2]),
                    scoring.finite(rival.record, rival.fractions),
                ]
            if any((fit is None for fit in prediction)):
                continue
            tested = scoring.measured(
                fractions,
                record,
                mean,
                measured.rates,
                measured.times,
                measured.penalty / compare.log_n,
                measured.core_masks,
            )
            gap = min(
                (
                    tested.score - fit.score - 2.0 * (tested.joint - fit.joint).max(initial=0.0)
                    for fit in prediction
                )
            )
            record["partner_fit"]["conditional_prediction_gain"] = float(gap)
            if gap <= MARGIN:
                continue
            visible = candidate(
                event, model, compare, primary.material, record["parameters"], port, old, budget, visible=True
            )
            if visible is not None:
                visible_gain, _, _, explanation = visible
                record["visible_partner"] = dict(explanation, gain=visible_gain)
                if gain <= visible_gain + MARGIN:
                    continue
            gain = min(gain, gap)
        from remaining_wavefront import remains

        extra, blocked = ([], False)
        for j, cone in enumerate(cones):
            candidate = cone.copy()
            candidate[7] = max(candidate[7], minimum_opening())
            if not optical_shape(model, candidate, 4)[0].any():
                continue
            blocked, evidence = remains(compare, measured, mean, candidate, amplitudes[j + 1], budget // 2)
            extra.append(evidence)
            if blocked:
                break
        record["third_trials"] = extra
        if blocked:
            continue
        gain = min(gain, matched)
        record.update(gain=gain, supported=True)
        if best is None or gain > best[0]:
            best = (gain, fractions, record)
    return (best, reports)
