"""Screen outgoing finite Cherenkov patterns using hit and unhit PMTs."""

from functools import lru_cache
from math import erf

import numpy as np
from numba import njit


@lru_cache(maxsize=1)
def compile_screen():
    from lookup import cap
    from selected_rate_adjoint import score as gate_score
    from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE

    @njit
    def scan(
        delta,
        normals,
        axes,
        opening,
        spans,
        direction,
        info,
        times,
        charge,
        birth,
        nuisance,
        starts,
        angular_width,
        conditional,
        widths,
        gate,
    ):
        result = np.zeros((len(axes), len(opening), len(spans), 3))
        sqrt2 = np.sqrt(2.0)
        distance2 = (delta * delta).sum(1)
        columns = nuisance.shape[1]
        projected = np.empty((len(spans), columns))
        last = np.max(starts + spans)
        exact_gate = conditional.shape[1] > 0
        for i, a in enumerate(axes):
            for j, theta in enumerate(opening):
                projected[:] = 0.0
                sine, cosine = np.sin(theta), np.cos(theta)
                for p, d in enumerate(delta):
                    z = d[0] * a[0] + d[1] * a[1] + d[2] * a[2]
                    r = np.sqrt(max(distance2[p] - z * z, 1.0))
                    foot = z - r * cosine / sine
                    sigma = max(27.0, angular_width * r / (sine * sine))
                    if foot < -3.8 * sigma or foot > last + 3.8 * sigma:
                        continue
                    previous_gap = -1.0
                    left_edge = 0.0
                    for k, length in enumerate(spans):
                        gap = starts[k]
                        if foot < gap - 3.8 * sigma or foot > gap + length + 3.8 * sigma:
                            continue
                        if gap != previous_gap:
                            left_edge = erf((foot - gap) / (sqrt2 * sigma))
                            previous_gap = gap
                        visible = 0.5 * (left_edge + erf((gap + length - foot) / (sqrt2 * sigma)))
                        if visible <= 1e-9:
                            continue
                        emission = min(max(foot, gap), gap + length)
                        x = d[0] - emission * a[0]
                        y = d[1] - emission * a[1]
                        zz = d[2] - emission * a[2]
                        radius = np.sqrt(max(x * x + y * y + zz * zz, 1.0))
                        n = normals[p]
                        angular = -(x * n[0] + y * n[1] + zz * n[2]) / radius
                        response = visible * cap(angular)[0] * sigma / (radius * radius + 10000.0)
                        slope = direction[p]
                        if exact_gate:
                            crossing = (gap + N_PHASE * cosine * (emission - gap) + N_GROUP * radius) / C
                            derivative = gate_score(
                                birth + crossing, times[p], widths[p], gate, conditional[p], charge[p] > 0.0
                            )
                            bound = 3.0 * info[p]
                            slope += 0.35 * min(max(derivative - slope, -bound), bound)
                        elif slope > 0.0 and charge[p] > 0.0:
                            crossing = (gap + N_PHASE * cosine * (emission - gap) + N_GROUP * radius) / C
                            residual = (times[p] - birth - crossing) / 2.15
                            # A bright first pulse from the parent may arrive
                            # before light from the daughter on the same PMT.
                            if residual >= 0.0:
                                slope *= 0.25 + 0.75 * np.exp(-0.5 * residual * residual)
                            else:
                                slope *= 0.54 + 0.46 * np.exp(-0.5 * residual * residual)
                        result[i, j, k, 0] += response * slope
                        result[i, j, k, 1] += response * response * info[p]
                        result[i, j, k, 2] += response
                        for c in range(columns):
                            projected[k, c] += response * nuisance[p, c]
                for k in range(len(spans)):
                    for c in range(columns):
                        result[i, j, k, 1] -= projected[k, c] ** 2
        return result

    return scan


def beam_variations(model, path, baseline, background, particle):
    """Approximate local changes of the surveyed beam and diffuse light, in PE."""
    from charge_starts import allowed_chord
    from material import material
    from value_only_receiver import receiver_values

    extent = path[9] * path[10]
    diffuse = background * model.bg
    columns = [0.12 * (baseline - diffuse), 0.25 * diffuse]
    material_model = material(particle)
    minimum = extent + 0.001 * material_model["margin_mm"] + 0.005
    maximum = 0.999 * 0.001 * material_model["maximum_range_mm"]
    # Survey shifts, launch tilt, emitting depth, initial range and angular width.
    scales = (0.018, 0.018, 0.025, 0.025, 0.065, 0.08, 0.25)
    for k, scale in enumerate(scales):
        field = []
        for step in (-scale, scale):
            trial = path.copy()
            length = extent
            if k < 2:
                trial[k] += step
            elif k < 4:
                trial[k + 1] += step
                trial[3:6] /= np.linalg.norm(trial[3:6])
            elif k == 4:
                length = max(0.014, length + step)
            elif k == 5:
                if minimum > maximum:
                    break
                trial[9] = np.clip(trial[9] + step, minimum, maximum)
            else:
                trial[11] += step
            chord = allowed_chord(trial[:3], trial[3:6])
            if chord is None or chord[0] > 1e-5:
                break
            length = min(length, 0.997 * chord[1], trial[9] - 0.006)
            if length < 0.012:
                break
            trial[10] = length / trial[9]
            value = receiver_values(model.pos, model.normals, trial, 3, particle).sum(0)
            field.append(model.response * 1000.0 * length * value)
        if len(field) == 2:
            columns.append(0.5 * (field[1] - field[0]))
    return np.column_stack(columns)


def profile(slope, info, variations):
    """Remove score directions that modest changes of the parent can explain."""
    weights = np.sqrt(info)
    u, s, _ = np.linalg.svd(weights[:, None] * variations, full_matrices=False)
    # A unit Gaussian penalty on every field change keeps the information
    # positive and limits projection through weakly identified modes.
    basis = u * (s / np.hypot(1.0, s))
    adjusted = slope - weights * (basis @ (basis.T @ (slope / weights)))
    return adjusted, np.ascontiguousarray(basis * weights[:, None])


def peaks(
    competition,
    path,
    baseline,
    background,
    particle,
    geometries,
    *,
    profile_parent=False,
    selected_gate=False,
):
    from charge_starts import weighted_quantile
    from physical_pulse import wavefront

    model, event = competition.model, competition.event
    axis = path[3:6]
    point = path[:3] + axis * path[9] * path[10]
    observed = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    take = slice(None) if selected_gate else model.ix
    time, _, flight = wavefront(model.pos[take], model.normals[take], path, particle, order=5)
    owned = np.clip(1.0 - background * model.bg[model.ix] / baseline[model.ix], 0.0, 1.0)
    direct = time[0, model.ix] if selected_gate else time[0]
    zero = weighted_quantile(observed - direct, q * owned, [0.5])[0]
    conditional = None
    if selected_gate:
        from scipy.optimize import minimize_scalar
        from selected_rate_adjoint import cache
        from value_only_receiver import receiver_values

        extent = 1000.0 * path[9] * path[10]
        photons = model.response * extent * receiver_values(model.pos, model.normals, path, 3, particle)
        rates = np.column_stack([background * model.bg, photons.T])
        selection = competition.selection
        result = minimize_scalar(
            lambda t: -selection.score(rates, time.T + t)[0].sum(),
            method="bounded",
            bounds=(zero - 1.8, zero + 1.8),
            options=dict(maxiter=13, xatol=0.052),
        )
        zero = float(result.x)
        conditional = cache(selection, rates, time.T + zero)
    birth = zero + flight
    nuisance = beam_variations(model, path, baseline, background, particle) if profile_parent else None
    return point, rank_at(
        competition,
        point,
        birth,
        baseline,
        geometries,
        prior_axis=axis,
        nuisance=nuisance,
        conditional=conditional,
    )


def rank_at(
    competition, point, birth, baseline, geometries, *, prior_axis=None, nuisance=None, conditional=None
):
    return scan_modes(
        competition, point, birth, baseline, geometries, prior_axis, nuisance, conditional, neutral=False
    )


def scan_modes(
    competition, point, birth, baseline, geometries, prior_axis, nuisance, conditional, *, neutral
):
    from charge_starts import allowed_chord
    from residual_search import axis_grid

    model, event = competition.model, competition.event
    observed = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    start, end = np.zeros(len(model.active)), np.zeros(len(model.active))
    start[model.ix], end[model.ix] = observed, q
    _, slope = model.obs.score(baseline, model.ix)
    axes = np.vstack([axis_grid(224), *[np.asarray(g["axes"]) for g in geometries]])
    if prior_axis is not None:
        axes = axes[axes @ prior_axis < np.cos(np.radians(13.0))]
    if neutral:
        openings = np.radians([37.0, 41.0])
        gap = np.repeat([35.0, 120.0, 285.0, 535.0, 850.0], 3)
        spans = np.tile([62.0, 145.0, 300.0], 5)
        width = np.radians(4.4)
    else:
        openings = np.radians([20.0, 26.0, 32.0, 37.0, 41.0])
        spans = np.asarray([52.0, 135.0, 300.0, 600.0])
        gap = np.zeros(len(spans))
        width = np.radians(2.7)
    info = 1.0 / np.maximum(baseline + 0.28, 0.28)
    direction = -slope
    basis = np.empty((len(info), 0))
    if nuisance is not None:
        direction, basis = profile(direction, info, nuisance)
    d = model.pos - 1000.0 * point
    timing = np.empty((len(info), 0)) if conditional is None else conditional
    selection = competition.selection
    scores = compile_screen()(
        d,
        model.normals,
        axes,
        openings,
        spans,
        direction,
        info,
        start,
        end,
        birth,
        basis,
        gap,
        width,
        timing,
        selection.width,
        selection.gate,
    )
    normal = np.maximum(scores[:, :, :, 1], 1e-30)
    values = np.maximum(scores[:, :, :, 0], 0.0) ** 2 / normal
    selected = []
    for flat in np.argsort(values.ravel())[::-1]:
        i, j, k = np.unravel_index(flat, values.shape)
        if values[i, j, k] < 9.0:
            break
        if neutral:
            chord = allowed_chord(point + 0.001 * gap[k] * axes[i], axes[i])
            if chord is None or chord[0] > 1e-5 or chord[1] < 0.026:
                continue
        if any(
            axes[i] @ a > 0.976 and abs(openings[j] - b) < 0.082 and abs(gap[k] - g) < 120.0
            for _, a, b, _, _, g in selected
        ):
            continue
        photons = scores[i, j, k, 0] * scores[i, j, k, 2] / normal[i, j, k]
        if photons < 13.0:
            continue
        selected.append(
            (
                float(values[i, j, k]),
                axes[i],
                float(openings[j]),
                float(min(photons, model.total)),
                spans[k] * 0.001,
                gap[k],
            )
        )
        if len(selected) == (7 if neutral else 4):
            break
    return selected if neutral else [r[:-1] for r in selected]
