"""Look for a short beam-entry turn whose light reflects from the window."""

from dataclasses import dataclass
from functools import lru_cache

from numba import njit
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from beam_window_light import Window, finite
from chain_comparison import MARGIN
from charge_starts import weighted_quantile
from chord_profile import profile as charge_profile
from compare_count_models import Comparison
from diffuse_timing import fit_mixture
from entrance_track import Aperture, axes as entrance_axes
from geometry import hardware_margin
from junction_residual_seeds import directions
from module_optics import conditional
from module_supported_ring import lower_bound
from resolved_ring_evidence import first_pulse_support
from timed_optical import arrival
from wctering.likelihood.geometry import NORMALS, UPPER
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE
from wcte.counting import track_counting


@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray
    axis: np.ndarray
    length: float
    opening: float


def radar(positions, path, resolution=1.9):
    """Straight-line optical reach and time for a measured finite interval."""
    delta = positions - path.origin
    along = delta @ path.axis
    rho = np.linalg.norm(delta - along[:, None] * path.axis, axis=1)
    sine, cosine = np.sin(path.opening), np.cos(path.opening)
    emit = along - rho * cosine / sine
    distance = emit - np.clip(emit, 0.0, path.length)
    width = np.maximum(20.0, rho * np.deg2rad(resolution) / sine**2)
    arrival = (N_PHASE * cosine * emit + N_GROUP * rho / sine) / C
    return np.exp(-0.5 * (distance / width) ** 2), arrival


@njit(cache=True)
def fast_ridges(positions, time, weight, origin, axes, openings, lengths, clock, z, radius):
    """Reuse each cone's projection and time factors over its span grid."""
    score = np.zeros((len(axes), len(openings), len(lengths)))
    reflected_score = np.zeros_like(score)
    sine, cosine = np.sin(openings), np.cos(openings)
    maximum = lengths[-1]
    for j in range(len(positions)):
        px, py, pz = positions[j]
        w = weight[j]
        if w < 1e-7:
            continue
        dx, dy = px - origin[0], py - origin[1]
        for reflected in (0, 1):
            if reflected and pz <= z:
                continue
            dz = 2.0 * z - pz - origin[2] if reflected else pz - origin[2]
            distance2 = dx * dx + dy * dy + dz * dz
            for i in range(len(axes)):
                ax, ay, az = axes[i]
                along = ax * dx + ay * dy + az * dz
                r = np.sqrt(max(distance2 - along**2, 1e-10))
                for k in range(len(sine)):
                    s = along - r * cosine[k] / sine[k]
                    sigma = max(28.0, 0.056 * r / sine[k] ** 2)
                    if s < -3.0 * sigma or s > maximum + 3.0 * sigma:
                        continue
                    for a, span in enumerate(lengths):
                        at = min(span, max(0.0, s))
                        edge = (s - at) / sigma
                        if abs(edge) > 3.0:
                            continue
                        x, y, sz = origin[0] + at * ax, origin[1] + at * ay, origin[2] + at * az
                        if reflected:
                            if sz <= z:
                                continue
                            u = (sz - z) / (sz + pz - 2.0 * z)
                            xx, yy = x + u * (px - x), y + u * (py - y)
                            if xx * xx + yy * yy > (1.25 * radius) ** 2:
                                continue
                        elif (sz - z) * (pz - z) < 0.0:
                            u = (z - sz) / (pz - sz)
                            xx, yy = x + u * (px - x), y + u * (py - y)
                            if xx * xx + yy * yy <= radius**2:
                                continue
                        flight = (
                            at * N_PHASE * cosine[k]
                            + N_GROUP * np.sqrt(max(distance2 + at * at - 2.0 * at * along, 1e-10))
                        ) / C
                        delay = (time[j] - flight - clock) / 1.7
                        hit = w * np.exp(-0.5 * (delay**2 + edge**2))
                        score[i, k, a] += hit
                        if reflected:
                            reflected_score[i, k, a] += hit
    return score, reflected_score


def parents(event, geometry, port, window):
    """Reserve distinct short hypotheses near the physical aperture."""
    origin = np.array([0.0, 0.0, 1000.0 * port.z])
    clock = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    selected = []
    for length in (0.45 * window.radius, 0.90 * window.radius, 1.8 * window.radius):
        best = None
        for axis in entrance_axes(geometry, port)[:3]:
            for angle in np.deg2rad([26.0, 31.0, 35.5, 38.8, 41.2]):
                path = Geometry(origin, axis, length, angle)
                vote, flight = radar(event.pmt_positions_mm, path)
                first = q * vote
                if first.sum() < 24.0:
                    continue
                birth = weighted_quantile(clock - flight, first, [0.5])[0]
                membership = vote * np.exp(-0.5 * ((clock - flight - birth) / 1.7) ** 2)
                rank = q @ membership
                if best is None or rank > best[0]:
                    best = rank, path, birth, membership
        if best is not None and best[0] >= 23.0:
            selected.append(best[1:])
    return selected


def nominate(event, model, geometry, port, window):
    """Trace measured pulses backwards from the window to propose the turn."""
    q = np.minimum(event.charges_pe, 2.0)
    positions = event.pmt_positions_mm
    time = event.times_ns - np.median(event.times_ns)
    grid = directions()
    openings = np.deg2rad([24.0, 29.0, 33.5, 37.5, 41.0])
    spans = np.array([110.0, 260.0, 480.0])
    modes = []
    render = fast_ridges
    for incoming, clock, owned in parents(event, geometry, port, window):
        end = incoming.origin + incoming.length * incoming.axis
        elapsed = incoming.length * N_PHASE * np.cos(incoming.opening) / C
        backward = grid[(grid[:, 2] < -0.1) & (grid @ incoming.axis < 0.0)]
        scores, reflected = render(
            positions,
            time,
            q * (1.0 - 0.8 * owned),
            end,
            backward,
            openings,
            spans,
            clock + elapsed,
            window.z,
            window.radius,
        )
        eligible = np.where(reflected >= 12.0, scores / np.sqrt(1.0 + spans / 240.0), -np.inf)
        seen = []
        for j in np.argsort(-eligible.ravel(), kind="stable"):
            i, k, a = np.unravel_index(j, eligible.shape)
            if not np.isfinite(eligible[i, k, a]):
                break
            axis = backward[i]
            if any((axis @ previous > 0.96 for previous in seen)):
                continue
            seen.append(axis)
            start = Geometry(end, axis, spans[a], openings[k])
            modes.append((eligible[i, k, a], incoming, start, float(reflected[i, k, a])))
            if len(seen) == 2:
                break
    return sorted(modes, key=lambda row: -row[0])[:4]


def setup(event, model, paths, order, window, reflectivity, clock, *, render=None):
    """Render one yield per charged leg and one clock shared by both legs."""
    columns, times, variances = [model.bg], [], []
    amplitude = [0]
    norms = []
    cores = []
    past = 0.0
    gate = np.asarray(event.timing_selection["prompt_window_ns"]) - np.median(event.times_ns)
    for j, p in enumerate(paths, 1):
        if render is None:
            rate, at, var = finite(
                model, p.origin, p.axis, p.length, p.opening, np.deg2rad(2.0), order, window, all_times=True
            )
        else:
            saved, at, var = render(tuple(p.origin), tuple(p.axis), p.length, p.opening, order)
            rate = saved.copy()
        rate[1] *= reflectivity
        center = at + past + clock
        spread = np.sqrt(var + 0.85**2)
        pass_gate = ndtr((gate[1] - center) / spread) - ndtr((gate[0] - center) / spread)
        # The long late path is often absent from measured charge. Apply
        # its approximate window visibility when predicting hit tubes.
        # Timing still uses the original gate and unshifted path moments.
        rate *= pass_gate.T
        total = rate.sum()
        if total < 1e-16:
            return None
        norms.append(total)
        cores.append(rate.sum(0) / total)
        for i, r in enumerate(rate):
            if r[model.ix].sum() <= 1e-15:
                continue
            columns.append(r / total)
            times.append(at[model.ix, i] + past)
            variances.append(var[model.ix, i])
            amplitude.append(j)
        past += p.length * N_PHASE * np.cos(p.opening) / C
    groups = np.array(amplitude)
    basis = np.vstack([model.bg, *cores])
    return (
        basis,
        np.array(norms),
        groups,
        np.vstack(columns),
        np.column_stack(times),
        np.column_stack(variances),
    )


def measure(
    event, model, paths, background, order, window, reflection, *, nuisance=None, render=None, iterations=32
):
    """Profile PE yields and the common event time on the original prompt hits."""
    clock = estimate_clock(event, paths[0])
    result = setup(event, model, paths, order, window, reflection, clock, render=render)
    if result is None:
        return None
    shape, expected, groups, optical, time, variance = result
    expected *= model.response
    # Reject a vanishing selected light component before profiling a yield.
    if np.any(expected < np.exp(-18.0)) or not np.isfinite(expected).all():
        return None
    amp = np.r_[background, expected]
    if nuisance is not None:
        column, center, spread = nuisance
        shape = np.vstack([shape, column])
        groups = np.r_[groups, len(amp)]
        optical = np.vstack([optical, column])
        time = np.column_stack([time, center])
        variance = np.column_stack([variance, spread])
        amp = np.r_[amp, 40.0]
    lower = {j: (0.30 * n, 2.5 * n) for j, n in enumerate(expected, 1)}
    clock_group = np.zeros(time.shape[1], dtype=int)
    if nuisance is not None:
        clock_group[-1] = 1
    trial = fit_mixture(
        event,
        model,
        shape,
        amp,
        time,
        amplitude_bounds=lower,
        return_detail=True,
        clock_groups=clock_group,
        timing_components=(groups, optical),
        variance=variance,
        iterations=iterations,
    )
    if trial is None:
        return None
    joint, a, at = trial
    mean = a @ shape
    hit = mean[model.ix]
    direct = (a[1 : 1 + len(paths), None] * shape[1 : 1 + len(paths), model.ix] / hit).T
    ownership = np.column_stack([1.0 - direct.sum(1), direct])
    rates = optical[:, model.ix].T * a[groups]
    return joint, ownership, mean, rates, at, variance, groups, a


def estimate_clock(event, path):
    weight, flight = radar(event.pmt_positions_mm, path)
    time = event.times_ns - np.median(event.times_ns)
    weight *= np.minimum(event.charges_pe, 2.0)
    if weight.sum() <= 1e-10:
        return float(np.median(time - flight))
    return weighted_quantile(time - flight, weight, [0.5])[0]


def chart(parent, child):
    """Move the two directions, lengths and openings around a shared junction."""
    frames = []
    for path in (parent, child):
        d = path.axis
        cross = np.cross(d, np.eye(3)[np.argmin(np.abs(d))])
        cross /= np.linalg.norm(cross)
        frames.append(np.column_stack([cross, np.cross(d, cross)]))

    def paths(x):
        point = parent.origin + np.r_[22.0 * x[:2], 0.0]
        tracks = []
        for j, p in enumerate((parent, child)):
            a, b, s, t = x[2 + 4 * j : 6 + 4 * j]
            axis = p.axis + 0.12 * frames[j] @ np.array([a, b])
            axis /= np.linalg.norm(axis)
            length = p.length * np.exp(0.45 * s)
            opening = np.clip(p.opening + 0.065 * t, 0.28, np.deg2rad(41.8))
            tracks.append(Geometry(point, axis, length, opening))
            point = point + length * axis
        return tracks

    return paths


def in_water(paths, port, *, check_modules=False):
    """Check the active water prism and the volume occupied by the pipe."""
    begin = paths[0].origin
    vertices = np.array([begin, paths[1].origin, paths[1].origin + paths[1].length * paths[1].axis])
    if (UPPER - vertices @ NORMALS.T / 1000.0).min() <= 0.0 or begin[:2] @ begin[:2] > (
        1000.0 * port.radius
    ) ** 2:
        return False
    # A returning particle may pass upstream beside the pipe. Reject a
    # line whose segment would pass back through its face or side wall.
    second = paths[1]
    if second.axis[2] >= 0.0:
        return False
    first = max((1000.0 * port.z - second.origin[2]) / second.axis[2], 0.0)
    if first < second.length:
        xy, u = second.origin[:2], second.axis[:2]
        cross = np.clip(-(xy @ u) / max(u @ u, 1e-20), first, second.length)
        closest = xy + cross * u
        if closest @ closest < (1000.0 * port.radius) ** 2:
            return False
    if not check_modules:
        return True
    # The qualified endpoint must also clear the mPMT domes.
    for path in paths:
        length = path.length / 1000.0
        probe = np.r_[path.origin / 1000.0, path.axis, np.zeros(3), length, 1.0, -3.5]
        if hardware_margin(probe) <= 1e-5:
            return False
    return True


def improve(
    event,
    model,
    start,
    background,
    port,
    window,
    budget,
    *,
    nuisance=None,
    initial=None,
    root=None,
    order=8,
    render=None,
    iterations=32,
):
    """Refine the shortlisted path using bounded small coordinate moves."""
    _, parent, daughter, _ = start
    coordinates = chart(parent, daughter) if root is None else root
    best = -np.inf, None, None
    calls = 0

    # Search the return geometry first, then allow the incoming support
    # and aperture point to move during the same bounded refinement.
    active = np.array([6, 7, 8, 9, 4, 5, 2, 3, 0, 1])

    def objective(y):
        nonlocal best, calls
        calls += 1
        x = np.zeros(10)
        x[active] = y
        paths = coordinates(x)
        if not in_water(paths, port):
            return 1e6
        candidate = measure(
            event,
            model,
            paths,
            background,
            order,
            window,
            window.reflectivity,
            nuisance=nuisance,
            render=render,
            iterations=iterations,
        )
        if candidate is None:
            return 1e6
        joint = candidate[0]
        if joint > best[0]:
            best = joint, paths, y.copy()
        return -joint

    start = np.zeros(len(active)) if initial is None else initial
    minimize(
        objective,
        start,
        method="Powell",
        bounds=[(-1.3, 1.3)] * len(active),
        options=dict(maxfev=budget, xtol=0.16, ftol=1e-4),
    )
    if best[1] is not None and not in_water(best[1], port, check_modules=True):
        return None, calls, None
    return best[1], calls, best[2]


def fit(event, model, geometry, basis, amplitude, winner, budget, *, bank=None):
    """Promote a second ring only after the full same-clock comparison."""
    if (
        len(basis) != 2
        or winner[2]["count"] != 1
        or event.timing_selection["prompt_window_ns"] is None
        or (geometry.get("emission_model") == "full_water_line")
    ):
        return (None, [])
    port = Aperture()
    window = Window(z=1000.0 * port.z)

    # Cache the exact response of a leg when the optimizer changes only the other leg.
    @lru_cache(maxsize=48)
    def render(origin, axis, length, opening, order):
        return finite(
            model,
            np.array(origin),
            np.array(axis),
            length,
            opening,
            np.deg2rad(2.0),
            order,
            window,
            all_times=True,
        )

    q = np.minimum(event.charges_pe, 2.0)
    if q @ winner[1][:, 0] < 40.0:
        return (None, [])
    proposals = nominate(event, model, geometry, port, window)
    if not proposals:
        return (None, [])
    q = np.minimum(event.charges_pe, 2.0)
    reference_mean = amplitude @ basis
    ref_charge = model.obs.score(reference_mean, model.ix)[0]
    reference_time = arrival(event, geometry)[:, 0]
    nuisance = (basis[1], reference_time, np.full(len(q), 2.0**2))
    ranked = []
    for seed in proposals:
        among = setup(
            event,
            model,
            seed[1:3],
            5,
            window,
            window.reflectivity,
            estimate_clock(event, seed[1]),
            render=render,
        )
        if among is None:
            continue
        light, norm, *_ = among
        nominal = model.response * norm
        if len(nominal) != 2:
            continue
        matrix = np.vstack([light, nuisance[0]])
        start = np.r_[amplitude[0], nominal, 0.65 * amplitude[1]]
        ll, _, error = charge_profile(matrix, model.obs, model.ix, start, budget=12)
        if error <= 0.0001 and ll >= ref_charge - 2.0 * np.sqrt(len(q)):
            ranked.append((ll, seed))
    if not ranked:
        return (None, [])
    comparison = Comparison(event, model)
    record = winner[2]
    old = (
        comparison.physical(record)
        if record.get("particles") and record.get("order")
        else comparison.optical(geometry, basis, amplitude, winner[1], record)
    )
    if old is None:
        return (None, [])
    alternative = None
    if bank:
        candidates = sorted(
            (r for r in bank if r.get("order") and r.get("particles") and (r["count"] >= 2)),
            key=lambda r: -r["gain"],
        )
        for candidate in candidates[:3]:
            try:
                rival = comparison.physical(candidate)
            except (ValueError, IndexError):
                continue
            if rival is not None and (alternative is None or rival.score > alternative.score):
                alternative = rival
    old_mean = amplitude @ basis
    old_shape = conditional(model.active, model.ix, event.charges_pe, old_mean)
    shortlisted = [seed for _, seed in sorted(ranked, key=lambda row: -row[0])[:3]]
    best_score = max((r.score for r in (old, alternative) if r is not None))

    def preview(seed):
        root = chart(*seed[1:3])
        path, calls, initial = improve(
            event,
            model,
            seed,
            amplitude[0],
            port,
            window,
            budget // 4,
            nuisance=nuisance,
            root=root,
            order=5,
            render=render,
            iterations=8,
        )
        if path is None:
            return None
        rough = measure(
            event, model, path, amplitude[0], 8, window, window.reflectivity, nuisance=nuisance, render=render
        )
        if rough is None or 2.0 * rough[0] - 15.0 * comparison.log_n < best_score - 4.0 * np.sqrt(len(q)):
            return None
        return (rough, calls, initial, root)

    if not any((preview(seed) is not None for seed in shortlisted)):
        return (None, [])
    # Refine all shortlisted seeds after at least one passes the low-order preview.
    good, reports = ([], [])
    for seed in shortlisted:
        paths, calls, _ = improve(
            event, model, seed, amplitude[0], port, window, budget, nuisance=nuisance, render=render
        )
        if paths is None:
            continue
        final = measure(
            event,
            model,
            paths,
            amplitude[0],
            24,
            window,
            window.reflectivity,
            nuisance=nuisance,
            render=render,
        )
        coarse = measure(
            event,
            model,
            paths,
            amplitude[0],
            16,
            window,
            window.reflectivity,
            nuisance=nuisance,
            render=render,
        )
        if final is None or coarse is None:
            continue
        ll, fractions, mean, rates, time, var, groups, _ = final
        error = abs(ll - coarse[0])
        report = dict(
            family="beam_window_turn",
            count=2,
            supported=False,
            gain=0.0,
            paths=[np.r_[p.origin, p.axis, p.length, p.opening].tolist() for p in paths],
            quadrature_ll_error=error,
            evaluations=calls,
            initial_reflected_vote=seed[-1],
            turn_deg=float(np.rad2deg(np.arccos(np.clip(paths[0].axis @ paths[1].axis, -1.0, 1.0)))),
        )
        reports.append(report)
        if (
            error >= 0.03
            or report["turn_deg"] <= track_counting.get().track_merge_angle_deg
            or set(fractions.argmax(1)) - {0} != {1, 2}
        ):
            continue
        from gated_photon_time import log_density
        from predictive_score import log_probabilities

        by_tube = log_probabilities(model.obs, mean, model.ix)[0]
        by_tube[model.ix] += log_density(
            comparison.time,
            *comparison.gate,
            time,
            rates,
            comparison.sigma,
            model.obs.offsets,
            model.obs.coefficients,
            variance=var,
        )
        module = np.bincount(comparison.modules, weights=by_tube)
        delta = 2.0 * (module - old.joint)
        penalty = 15.0 * comparison.log_n
        gain = delta.sum() - delta.max(initial=0.0) - (penalty - old.penalty)
        if alternative is not None:
            delta_rival = 2.0 * (module - alternative.joint)
            rival_gap = delta_rival.sum() - delta_rival.max(initial=0.0) - (penalty - alternative.penalty)
            report["bank_rival"] = dict(
                family=alternative.record["family"], count=alternative.record["count"], gain=float(rival_gap)
            )
            gain = min(gain, rival_gap)
        support = count_support(comparison, event, rates, time, var, groups)
        shape = conditional(model.active, model.ix, event.charges_pe, mean) - old_shape
        report.update(
            gain=float(gain),
            pulse_support=support,
            within_module_gain=float(shape.sum() - shape.max(initial=0.0)),
        )
        if (
            gain <= MARGIN
            or min((row["lower_statistical"] for row in support)) < track_counting.get().min_track_pmts
            or min((row["leave_one_module"] for row in support)) < track_counting.get().min_track_pmts
            or (report["within_module_gain"] <= 0.0)
        ):
            continue
        no_reflect = measure(
            event, model, paths, amplitude[0], 24, window, 0.0, nuisance=nuisance, render=render
        )
        if no_reflect is None:
            continue
        report["reflection_evidence"] = float(2.0 * (ll - no_reflect[0]))
        if report["reflection_evidence"] <= MARGIN:
            continue
        report["supported"] = True
        good.append((gain, fractions, report))
    return (max(good, key=lambda row: row[0], default=None), reports)


def count_support(comparison, event, rates, time, variance, groups):
    """Measure conditional first-light support for both charged legs."""
    obs = comparison.model.obs
    masks = (sum(1 << (int(i) - 1) for i in np.flatnonzero(groups == j)) for j in (1, 2))
    return [
        lower_bound(
            event.pmt_ids,
            first_pulse_support(
                comparison.time,
                comparison.gate,
                rates,
                time,
                comparison.sigma,
                -1,
                obs.offsets,
                obs.coefficients,
                excluded_mask=mask,
                variance=variance,
            ),
        )
        for mask in masks
    ]
