"""Test an independent secondary against a fitted two-section charged path."""

import numpy as np

from analog_optical import contain_finite
from chain_comparison import MARGIN
from chain_sources import paths
from detached_conversion import Timing
from finite_residual_fit import optical_shape
from module_optics import conditional
from physical_pulse import components, profile as track_profile, wavefront
from recoil_competitor import render
from residual_joint_optics import path_from_geometry
from residual_timing_vertices import refine as refine_vertices
from resolved_ring_evidence import first_pulse_support
from timed_charged_light import photon_time


def optical_modes(event, model, bank, state, track, points):
    """Reuse measured cone seeds and estimate independent residual births."""
    mean = np.exp(state["parameters"][-2]) * track.primary_cache[1]
    mean += np.exp(state["parameters"][-1]) * model.bg
    _, derivative = model.obs.score(mean, model.ix)
    hit = model.ix
    weight = np.minimum(event.charges_pe, 2.0) * np.clip(-derivative[hit] * mean[hit], 0.0, 2.0)
    new_vertices = refine_vertices(
        points, event.pmt_positions_mm / 1000.0, event.times_ns - np.median(event.times_ns), weight, maximum=3
    )
    ranked = []
    for count in ("3", "2", "1"):
        geometry = bank.get(count)
        if geometry is None:
            continue
        for i in range(len(geometry["axes"])):
            try:
                p = path_from_geometry(geometry, i)
            except (ValueError, ArithmeticError):
                continue
            for center in [p[:3], *new_vertices]:
                # A neutral source can be displaced from both charged
                # sections. Position and clock are free in the seed.
                x = p.copy()
                x[:3] = center
                try:
                    contain_finite(x)
                except (ValueError, ArithmeticError):
                    continue
                response = optical_shape(model, x, 4)[0]
                if not response.any():
                    continue
                gradient = -float(response @ derivative)
                info = (response**2) @ (1.0 / np.maximum(mean, 0.15))
                strength = max(0.0, gradient) / max(info, 1e-20)
                if strength < 10.0:
                    continue
                score = gradient * strength
                ranked.append((score, x, np.clip(strength, 10.0, 0.7 * model.total)))
    unique = []
    for _, x, strength in sorted(ranked, key=lambda row: -row[0]):
        if any(np.linalg.norm(x[:3] - old[:3]) < 0.13 and x[3:6] @ old[3:6] > 0.97 for old in unique):
            continue
        unique.append(x)
        yield x, strength
        if len(unique) == 3:
            break


def measure(event, model, exact, rate, time, photons, direct, core, extra_time):
    """Profile a calibrated secondary while holding the two-track light fixed."""
    from charge_starts import weighted_quantile
    from gated_photon_time import evaluate
    from scipy.optimize import minimize

    timing = Timing.from_event(event)
    response = photons[model.ix]
    if response.sum() <= 1e-15:
        return None
    rays = np.column_stack([time, extra_time])
    initial = weighted_quantile(timing.time - extra_time, np.minimum(model.q, 2.0) * response, [0.5])[0]

    def objective(x, detail=False):
        scale = np.exp(x[0])
        mean = exact + scale * photons
        rates = np.column_stack([rate, scale * response])
        centers = rays + np.r_[np.zeros(time.shape[1]), x[1]]
        charge, residual = model.obs.score(mean, model.ix)
        ll, d_rate, d_time = evaluate(
            timing.time,
            timing.low,
            timing.high,
            centers,
            rates,
            timing.sigma,
            model.obs.offsets,
            model.obs.coefficients,
        )
        if detail:
            shares = np.vstack([core, scale * direct])[:, model.ix] / mean[model.ix]
            return mean, rates, centers, shares.T, x[1]
        gradient = np.r_[scale * (residual @ photons - d_rate[:, -1] @ response), -d_time[:, -1].sum()]
        return -(charge + ll.sum()) / model.total, gradient / model.total

    bounds = [(np.log(0.65), np.log(1.55)), (initial - 4.0, initial + 4.0)]
    fit = minimize(
        objective,
        [0.0, initial],
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=24, ftol=1e-9),
    )
    if np.isfinite(fit.fun):
        return objective(fit.x, detail=True)
    return None


def review(event, model, bank, track, state, fractions, budget, *, refine_tracks=False):
    """Return a supported third core, or an unresolved extra-light diagnostic."""
    from qualified_optical_tracks import paths_for_cone

    reference = track_profile(event, track, state, separate_knockon=True)
    if reference is None:
        return None, []
    timing = Timing.from_event(event)
    sections = [p for p, _ in paths(state["parameters"][: track.ng])]
    points = [point for p in sections for point in (p[:3], p[:3] + p[9] * p[10] * p[3:6])]
    fits = list(optical_modes(event, model, bank, state, track, points))
    if not fits:
        return None, []
    x = state["parameters"]
    response = np.exp(x[-2])
    exact = response * track.primary_cache[1] + np.exp(x[-1]) * model.bg
    core = response * track.primary_cache[2]
    rates, clocks = components(event, track, state, separate_knockon=True)
    clocks = clocks + reference["clock_ns"]
    previous = timing.likelihood(model, exact, rates, clocks)
    within = conditional(model.active, model.ix, event.charges_pe, exact)
    score = []
    for cone, strength in fits:
        for mass, p in paths_for_cone(model, cone, strength):
            direct, total, _ = render(model, p, mass, 4)
            if not direct.any():
                continue
            mean = exact + response * total
            ll = model.obs.score(mean, model.ix)[0]
            share = response * direct[model.ix] / mean[model.ix]
            support = float(np.sum(1.0 - model.obs.no_source(mean[model.ix], share.clip(0.0, 1.0))))
            if support >= 18.0:
                score.append((ll, support, mass, p))
    reports, best = [], None
    for _, _, mass, p in sorted(score, key=lambda row: -row[0])[:budget]:
        direct, light, _ = render(model, p, mass, 32)
        if light.sum() <= 1e-12:
            continue
        flight = photon_time(event, model.normals[model.ix], [p], mass)
        trial = measure(event, model, exact, rates, clocks, response * light, response * direct, core, flight)
        if trial is None:
            continue
        mean, candidate_rates, centers, shares, birth = trial
        delta = timing.likelihood(model, mean, candidate_rates, centers) - previous
        penalty = 11.0 * np.log(len(model.active))
        gain = 2.0 * (delta.sum() - delta.max(initial=0.0)) - penalty
        # Preserve the same summed rate while marking only the directly
        # counted part of the third source for conditional occupancy.
        hit_direct = mean[model.ix] * shares[:, -1]
        timing_rates = np.column_stack([candidate_rates, hit_direct])
        timing_rates[:, -2] = np.maximum(timing_rates[:, -2] - hit_direct, 0.0)
        marked = np.column_stack([centers, centers[:, -1]])
        extra = first_pulse_support(
            timing.time,
            np.array([timing.low, timing.high]),
            timing_rates,
            marked,
            timing.sigma,
            marked.shape[1] - 1,
            model.obs.offsets,
            model.obs.coefficients,
        )
        support = float(extra.sum() - 1.645 * np.sqrt(extra @ (1.0 - extra)))
        if gain <= MARGIN:
            continue
        shape = conditional(model.active, model.ix, event.charges_pe, mean) - within
        q = np.minimum(event.charges_pe, 2.0)
        kept = (q @ np.minimum(shares[:, :2], fractions[:, 1:])) / np.maximum(q @ fractions[:, 1:], 1e-20)
        proposed = np.column_stack([1.0 - shares.sum(1), shares])
        report = dict(
            material=mass,
            point_mm=(1000.0 * p[:3]).tolist(),
            support=support,
            gain=float(gain),
            within=float(shape.sum() - shape.max(initial=0.0)),
            retained_charge=kept.tolist(),
            supported=False,
            count=3,
        )
        reports.append(report)
        if (
            support >= 30.0
            and kept.min() >= 0.65
            and report["within"] > 0.0
            and set(proposed.argmax(1)) - {0} == {1, 2, 3}
        ):
            report["supported"] = True
            if best is None or gain > best[0]:
                best = (
                    gain,
                    proposed,
                    report,
                    p,
                    birth,
                    timing.likelihood(model, mean, candidate_rates, centers),
                )
    if best is None:
        return None, reports
    gain, proposed, report, extra_path, birth, joint = best
    report.update(parameters=extra_path.tolist(), birth_clock_ns=float(birth))
    if refine_tracks:
        from detached_track_fit import refine
        from entrance_track import Aperture

        _, _, flight = wavefront(
            event.pmt_positions_mm, model.normals[model.ix], sections[0], track.particles[0]
        )
        clock = reference["clock_ns"]
        refined = refine(
            event,
            model,
            [*sections, extra_path],
            (*track.particles, report["material"]),
            x[-2:],
            timing,
            np.array([clock, clock + flight, birth]),
            Aperture(),
            46,
        )
        if refined is not None:
            mean, rates, times, shares, detail = refined
            delta = timing.likelihood(model, mean, rates, times) - joint
            # Moving the second birth and its clock adds four coordinates
            # relative to the serial pair plus the separately fitted third.
            gap = 2.0 * (delta.sum() - delta.max(initial=0.0)) - 4.0 * np.log(len(model.active))
            gate = np.array([timing.low, timing.high])
            support = []
            for i in range(3):
                probability = first_pulse_support(
                    timing.time,
                    gate,
                    rates,
                    times,
                    timing.sigma,
                    2 * i,
                    model.obs.offsets,
                    model.obs.coefficients,
                )
                support.append(float(probability.sum() - 1.645 * np.sqrt(probability @ (1.0 - probability))))
            alternate = np.column_stack([1.0 - shares.sum(1), shares])
            detail.update(gain=float(gap), support=support)
            report["joint_refit"] = detail
            if gap > MARGIN and min(support) >= 30.0 and set(alternate.argmax(1)) - {0} == {1, 2, 3}:
                report["joint_refit"]["accepted"] = True
                gain, proposed = gain + gap, alternate
    return (gain, proposed, report), reports
