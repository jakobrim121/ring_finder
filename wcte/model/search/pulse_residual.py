"""Search for a short charged daughter using the joint charge and time score."""

from math import erf, erfc, exp, expm1, log, sqrt

from numba import njit
import numpy as np

from balanced_search import feasible
from bare_model import Model as Bare
from charge_starts import recoil_start
from gated_photon_time import series, wavefront
from joined_segments import arc
from physical_pulse import components, profile
from residual_search import axis_grid
from serial_refinement import leg_starts
from source import trajectory
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


@njit(cache=True)
def timing_slope_terms(time, low, high, rates, means, sigma, offsets, coefficients):
    """Coefficients of the exact time score for adding a zero-rate source."""
    values = np.empty((4, len(time)))
    for i, t in enumerate(time):
        sky = rates[i, 0]
        a = sky / (high - low)
        b = a * (high - t)
        c, d = sky, 0.0
        for j in range(means.shape[1]):
            rate, center = rates[i, j + 1], means[i, j]
            density, survival, _ = wavefront(t - center, sigma[i])
            a += rate * density
            b += rate * survival
            c += rate * wavefront(low - center, sigma[i])[1]
            d += rate * wavefront(high - center, sigma[i])[1]
        first, last = offsets[i], offsets[i + 1]
        _, after_hit = series(b, coefficients, first, last, 1)
        start, left = series(c, coefficients, first, last, 0)
        stop, right = series(d, coefficients, first, last, 0)
        ratio = min(stop - start, -1e-15)
        inside = -expm1(ratio)
        values[:, i] = 1.0 / max(a, 1e-300), after_hit, -left / inside, right * exp(ratio) / inside
    return values


@njit(cache=True)
def timing_slope(time, low, high, means, sigma, terms):
    """Evaluate the new source's first-arrival score at trial flight times."""
    result = np.empty_like(means)
    for j in range(len(means)):
        for i, t in enumerate(time):
            center = means[j, i]
            density, survival, _ = wavefront(t - center, sigma[i])
            result[j, i] = (
                terms[0, i] * density
                + terms[1, i] * survival
                + terms[2, i] * wavefront(low - center, sigma[i])[1]
                + terms[3, i] * wavefront(high - center, sigma[i])[1]
            )
    return result


def prefix_seeds(path, *, broad=False):
    """Move the interaction along and across the measured track's endpoint."""
    span = path[9] * path[10]
    offsets = [0.35, 0.60, 0.80, 1.0, 1.0 + 0.06 / span]
    if broad:
        offsets.extend([0.24, 0.48, 0.70, 0.90])
    end = np.unique(np.clip(offsets, 0.20, 1.30))
    for u in end:
        prefix = arc(path, 0.0, u)
        if prefix is not None:
            yield prefix
    # A short cone constrains direction more tightly than transverse birth.
    # Shift the entire prefix as a start and let the physical fit move it.
    side = np.cross(path[3:6], np.eye(3)[np.argmin(abs(path[3:6]))])
    side /= np.linalg.norm(side)
    for axis in (side, np.cross(path[3:6], side)):
        for shift in (-0.10, 0.10):
            prefix = path.copy()
            prefix[:3] += shift * axis
            yield prefix


@njit(cache=True)
def optical_grid(sine, cosine, span, along, radial, incidence, axial, origin_shift=0.0):
    """Integrate the finite cone over all receivers for the charge shortlist."""
    shape = np.zeros_like(along)
    root_two = sqrt(2.0)
    angular_width = np.deg2rad(3.2) / sine**2
    for j in range(along.shape[0]):
        for i in range(along.shape[1]):
            along_ray = along[j, i] + origin_shift
            emission = along_ray - radial[j, i] * cosine / sine
            width = max(radial[j, i] * angular_width, 12.0)
            a, b = -emission / width, (span - emission) / width
            # Outside this interval, even the larger normal tail carries
            # less than 4e-17 of the unnormalized Gaussian mass.
            if a >= 8.35 or b <= -8.35:
                continue
            if a >= 0.0:
                weight = 0.5 * (erfc(a / root_two) - erfc(b / root_two))
            elif b <= 0.0:
                weight = 0.5 * (erfc(-b / root_two) - erfc(-a / root_two))
            else:
                weight = 0.5 * (erf(b / root_two) - erf(a / root_two))
            flight = min(max(emission, 0.0), span)
            distance = np.hypot(radial[j, i], along_ray - flight)
            exposure = max((incidence[i] + (flight - origin_shift) * axial[j, i]) / distance, 0.015)
            shape[j, i] = weight * width * exposure / (distance**2 + 10000.0)
        shape[j] /= max(shape[j].sum(), 1e-300)
    return shape


@njit(cache=True)
def optical_arrival(order, indices, sine, cosine, span, along, radial, origin_shift=0.0):
    """Evaluate flight times only for shortlisted directions and recorded hits."""
    arrival = np.empty((len(order), len(indices)))
    for j, axis in enumerate(order):
        for k, i in enumerate(indices):
            along_ray = along[axis, i] + origin_shift
            emission = along_ray - radial[axis, i] * cosine / sine
            flight = min(max(emission, 0.0), span)
            distance = np.hypot(radial[axis, i], along_ray - flight)
            arrival[j, k] = (N_PHASE * cosine * flight + N_GROUP * distance) / C
    return arrival


@njit(cache=True)
def optical_spans(sine, cosine, spans, along, radial, incidence, axial):
    """Share receiver projections and the lower Gaussian bound over spans."""
    shape = np.zeros((len(spans), along.shape[0], along.shape[1]))
    root_two = sqrt(2.0)
    angular_width = np.deg2rad(3.2) / sine**2
    for j in range(along.shape[0]):
        for i in range(along.shape[1]):
            z = along[j, i]
            radius = radial[j, i]
            emission = z - radius * cosine / sine
            width = max(radius * angular_width, 12.0)
            a = -emission / width
            if a >= 8.35:
                continue
            lower = erfc(a / root_two) if a >= 0.0 else erf(a / root_two)
            upper_tail = erfc(-a / root_two) if a < 0.0 else 0.0
            for k, span in enumerate(spans):
                b = (span - emission) / width
                if b <= -8.35:
                    continue
                if a >= 0.0:
                    weight = 0.5 * (lower - erfc(b / root_two))
                elif b <= 0.0:
                    weight = 0.5 * (erfc(-b / root_two) - upper_tail)
                else:
                    weight = 0.5 * (erf(b / root_two) - lower)
                flight = min(max(emission, 0.0), span)
                distance = np.hypot(radius, z - flight)
                exposure = max((incidence[i] + flight * axial[j, i]) / distance, 0.015)
                shape[k, j, i] = weight * width * exposure / (distance**2 + 10000.0)
        for k in range(len(spans)):
            shape[k, j] /= max(shape[k, j].sum(), 1e-300)
    return shape


def screened(event, model, state, *, broad=False, batch_shapes=False):
    """Compare each finite residual with the trimmed parent and unhit PMTs."""
    x = state["parameters"]
    length = x[9] * x[10]
    if not 0.035 <= length <= (2.8 if broad else 0.55) or model.particles == ("electron",):
        return
    old = profile(event, model, state, separate_knockon=True)
    if old is None:
        return
    center = np.median(event.times_ns)
    time = event.times_ns - center
    low, high = np.asarray(event.timing_selection["prompt_window_ns"]) - center
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    offsets, coefficients = model.obs.offsets, model.obs.coefficients
    grid = axis_grid(320)
    # No forward-only cut. The two truth tables show short daughters can
    # travel backward and still light a separate, readily visible cone.
    for prefix in prefix_seeds(x[:12], broad=broad):
        probe = np.r_[prefix, x[-2:]]
        if model.hardware_margin(probe) <= 1e-5:
            continue
        state = model(probe, True)
        response, diffuse = np.exp(probe[-2:])
        expected = response * model.primary_cache[1] + diffuse * model.bg
        _, charge_slope = model.obs.score(expected, model.ix)
        if np.count_nonzero(-charge_slope[model.ix] * expected[model.ix] > 0.20) < 19:
            continue
        point, tangent, _, _ = trajectory(prefix, np.ones(1))
        rates, means, elapsed = components(event, model, state, separate_knockon=True, return_elapsed=True)
        clock = old["clock_ns"]
        terms = timing_slope_terms(time, low, high, rates, means + clock, sigma, offsets, coefficients)
        axes = grid[grid @ tangent[0] < np.cos(np.deg2rad(18.0))]
        delta = model.pos - point[0]
        along = axes @ delta.T
        radial = np.sqrt(np.maximum(np.einsum("ij,ij->i", delta, delta) - along**2, 10.0))
        incidence = -np.einsum("ij,ij->i", model.normals, delta)
        axial = axes @ model.normals.T
        inverse = 1.0 / np.maximum(expected, 0.12)
        spans = (60.0, 150.0, 285.0, 530.0) if broad else (60.0, 150.0, 285.0)
        for opening in np.deg2rad([23.0, 29.0, 35.0, 40.5]):
            sine, cosine = np.sin(opening), np.cos(opening)
            shapes = (
                optical_spans(sine, cosine, spans, along, radial, incidence, axial)
                if batch_shapes
                else (optical_grid(sine, cosine, span, along, radial, incidence, axial) for span in spans)
            )
            for span, shape in zip(spans, shapes):
                charge = -shape @ charge_slope
                info = (shape**2) @ inverse
                # Use the charge derivative only as a cheap shortlist.
                # The exact joint derivative reranks directions and their
                # nominal photon yield before any physical optimization.
                shortlist = 9 if broad == "diverse" else 5
                order = np.argsort(np.maximum(charge, 0.0) ** 2 / np.maximum(info, 1e-30))[-shortlist:]
                hit = shape[order][:, model.ix]
                arrival = optical_arrival(order, model.ix, sine, cosine, span, along, radial)
                timed = timing_slope(time, low, high, arrival + clock + elapsed, sigma, terms)
                gradient = charge[order] + np.einsum("ij,ij->i", hit, timed)
                info = info[order] + np.sum((hit * timed) ** 2, axis=1)
                strength = np.maximum(gradient, 0.0) ** 2 / np.maximum(info, 1e-30)
                candidates = (
                    np.argsort(-strength, kind="stable")[:2] if broad == "diverse" else [np.argmax(strength)]
                )
                for j in candidates:
                    if strength[j] < 9.0:
                        continue
                    photons = min(gradient[j] / max(info[j], 1e-30), 0.45 * model.total)
                    # A point with no own timed footprints is a poor seed even
                    # when several very bright PMTs produce a large derivative.
                    if (
                        np.count_nonzero(
                            hit[j] * np.maximum(timed[j] - charge_slope[model.ix], 0.0)
                            > 0.07 * max(gradient[j], 1.0) / 20.0
                        )
                        < 16
                    ):
                        continue
                    yield (
                        state["ll"] + 0.5 * strength[j],
                        prefix,
                        axes[order[j]],
                        opening,
                        span / 1000.0,
                        photons,
                    )


def starts(event, model, state, *, broad=False, batch_shapes=False):
    """Refine only distinct calibrated starts that explain the local excess."""
    ranked = sorted(
        screened(event, model, state, broad=broad, batch_shapes=batch_shapes),
        key=lambda row: row[0],
        reverse=True,
    )
    proposals, origins, models = [], [], {}
    for rank, prefix, axis, opening, span, photons in ranked:
        join = trajectory(prefix, np.ones(1))[0][0] / 1000.0
        if any(np.linalg.norm(join - v) < 0.08 and axis @ direction > 0.95 for v, direction in origins):
            continue
        origins.append((join, axis))
        # Evaluate mass alternatives with an absolute light yield. Search
        # a path whose energy can support the observed Cherenkov opening.
        for mass in dict.fromkeys((*model.particles, "pion", "proton", "muon")):
            materials = model.particles + (mass,)
            if materials not in models:
                models[materials] = Bare(event.pmt_ids, event.charges_pe, model.bg, materials, 4, 8)
            trial = models[materials]
            for leg in leg_starts(trial, join, axis, opening, photons, span, mass):
                x = recoil_start(trial, np.r_[prefix, leg, state["parameters"][-2:]])
                if x is None or not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5:
                    continue
                fit = trial(x, True)
                support = min(
                    float(np.sum(1.0 - trial.obs.no_source(fit["means_hit"], f)))
                    for f in fit["posterior"][1:]
                )
                if support >= 14.0:
                    proposals.append((fit["ll"], rank, materials, x, support))
        if len(origins) == (10 if broad == "diverse" else 6):
            break
    selected = []
    # Maintain two different basins. A charge-only best start can erase a
    # near-threshold daughter before the accurate timing comparison.
    for score, rank, masses, x, support in sorted(proposals, key=lambda row: row[0], reverse=True):
        if any(
            x[12:15] @ y[12:15] > 0.96 and abs(x[9] * x[10] - y[9] * y[10]) < 0.08 for _, _, y in selected
        ):
            continue
        selected.append((float(score), masses, x))
        if len(selected) == (3 if broad == "diverse" else 2):
            break
    return selected


def extension_starts(event, model, state, *, batch_shapes=False):
    """Scan longer daughters only after a short-track search is exhausted."""
    length = state["parameters"][9] * state["parameters"][10]
    if not 0.55 < length <= 2.8 or model.particles == ("electron",):
        return []
    return starts(event, model, state, broad=True, batch_shapes=batch_shapes)


def qualify(
    event,
    single,
    trial,
    state,
    fractions,
    incumbent,
    budget,
    *,
    compare_lineage=False,
    require_long=False,
    require_short=False,
    within_timing=False,
    allow_profiled_turn=False,
):
    """Require distinct above-margin light with a shared relative clock."""
    from chain_comparison import MARGIN, counted
    from joined_segments import separated_directions
    from module_optics import compare
    from ring_supported import preserves_prompt
    from timed_two_track import grouped_pulse, restore
    from visibility_profile import fit as below_support

    angle, resolved = separated_directions(trial, state)
    detail = dict(turn_deg=float(np.rad2deg(angle)))
    if (not resolved and not allow_profiled_turn) or set(fractions.argmax(1)) - {0} != {1, 2}:
        return False, detail
    result = grouped_pulse(event, trial, state, [[0], [1]])
    if result is None:
        return False, detail
    timing, support = result
    detail["conditional_support"] = support
    if min(row["lower_statistical"] for row in support) < 27.0:
        return False, detail
    if require_long:
        # This pass specifically searches for a sustained outgoing track.
        # Shorter branches were already tested by the calibrated short pass.
        x = state["parameters"]
        if x[18] * x[19] < 0.22 or support[1]["leave_one_module"] < 42.0:
            return False, detail
    if require_short:
        # A real upstream flash from an early hard collision can leave a
        # very short track. A long apparent incoming section is tested by
        # the existing continuation fit under its own matched null.
        x = state["parameters"]
        if x[9] * x[10] > 0.105 or support[0]["leave_one_module"] < 30.0:
            return False, detail
    original = counted(*single)
    if original.shape[1] == 2:
        retained, lost = preserves_prompt(event, (*single, original), (trial, state, fractions))
        detail["lost_prompt_hits"] = lost
        if not retained:
            return False, detail
    old_model, old_state = single
    if isinstance(incumbent.get("particles"), tuple) and incumbent.get("order"):
        old_model, old_state, _ = restore(event, trial.bg, incumbent)
    previous = profile(event, old_model, old_state, separate_knockon=True)
    if previous is None:
        return False, detail
    joint = state["ll"] + timing["timing_ll"]
    gain = 2.0 * (joint - old_state["ll"] - previous["timing_ll"])
    gain -= (trial.dimension - old_model.dimension) * log(len(trial.active))
    detail["penalized_joint_gain"] = float(gain)
    if gain <= MARGIN:
        return False, detail
    weak = int(np.argmin([row["pmt_support"] for row in support]))
    contrast = compare((old_model, old_state), (trial, state), event, fractions[:, weak + 1])
    detail["within_module_contrast"] = contrast
    distributed = (
        contrast["extra_module_delta_without_brightest"] > 0.0 and contrast["extra_module_positive"] >= 3
    )
    timed_contrast = None
    if within_timing and (not distributed or contrast["delta"] <= 0.0):
        from pulse_panel_fit import within_module_wavefront

        timed_contrast = within_module_wavefront(
            event,
            (old_model, old_state),
            (trial, state),
            fractions[:, weak + 1],
            (previous["clock_ns"], timing["clock_ns"]),
        )
        detail["within_module_wavefront"] = timed_contrast
        if timed_contrast is not None:
            distributed = (
                timed_contrast["positive_modules"] >= 3
                and timed_contrast["exposed_gain_without_best"] > MARGIN
                and timed_contrast["joint_gain_without_best"] > MARGIN
            )
    if not distributed:
        return False, detail
    if support[weak]["pmt_support"] < 65.0:
        under = below_support(trial, state, weak, budget[0], budget[1], timed_event=event)
        detail["below_support"] = under
        if under is not None and 2.0 * (joint - under["ll"] - under["timing_ll"]) <= MARGIN:
            return False, detail
    if compare_lineage:
        from continuous_reference import fit as continuous
        from lineage_competitor import fit as merged
        from pulse_panel_fit import compare as module_pulse

        # Require the separate cone to improve the shape across the whole
        # detector, including modules that mostly see the dominant track.
        if contrast["delta"] <= 0.0 and (
            timed_contrast is None or timed_contrast["within_module_time"] <= 0.0
        ):
            return False, detail
        if compare_lineage == "modules":
            old_fractions = old_state["posterior"][1:].T
            panel = module_pulse(
                event,
                (old_model, old_state),
                (trial, state),
                old_fractions,
                fractions[:, 1:],
                (previous["clock_ns"], timing["clock_ns"]),
            )
            detail["module_pulse"] = panel
            if panel is None or panel["positive_modules"] < 3:
                return False, detail
            penalty = (trial.dimension - old_model.dimension) * log(len(trial.active))
            if 2.0 * panel["joint_without_best_module"] - penalty <= MARGIN:
                return False, detail
        if angle < np.deg2rad(35.0):
            same = merged(event, single[0], trial, state, budget[1], timed=True, separate_knockon=True)
            detail["merged_track"] = same
            if same is not None and 2.0 * (joint - same["ll"] - same["timing_ll"]) <= MARGIN:
                return False, detail
        smooth = continuous(event, *single, trial, state, weak)
        if smooth is not None:
            model, fitted, _, _ = smooth
            pulse = profile(event, model, fitted, separate_knockon=True)
            if pulse is not None:
                gap = 2.0 * (joint - fitted["ll"] - pulse["timing_ll"])
                gap -= (trial.dimension - model.dimension) * log(len(trial.active))
                detail["continuous_track_gap"] = float(gap)
                if gap <= MARGIN:
                    return False, detail
        if compare_lineage == "conversion":
            from detached_conversion import contest

            conversion = contest(event, single, trial, state, fractions, weak, min(budget[1], 36))
            detail["detached_conversion"] = conversion
            if conversion is not None and conversion["penalized_gap"] <= MARGIN:
                return False, detail
    return True, detail
