"""Compare resolved and unresolved light from compact conversion pairs."""

from itertools import product

import numpy as np
from scipy.optimize import minimize

from charge_starts import allowed_chord
from geometry import hardware_margin
from late_light_fit import intensity_fit
from material import material
from source import raw
from timed_charged_light import photon_time
from timed_optical import arrival


def templates(center, direction, photons):
    """Match observed brightness with two particles from one conversion."""
    electron = material("electron")
    for share, shift, turn in product((0.5, 0.7, 0.88), (-0.12, -0.04, 0.035), (0.0, 0.12)):
        p = center + shift * direction
        water = allowed_chord(p, direction)
        if water is None or water[0] > 0.0 or water[1] < 0.02:
            continue
        energy = np.clip((14.0 + 0.21 * photons) * np.array([share, 1.0 - share]), 2.0, 240.0)
        spans = electron["range_at"](energy) / 1000.0
        if max(spans) >= water[1]:
            continue
        side = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
        side /= np.linalg.norm(side)
        for basis in (side, np.cross(direction, side)) if turn else (side,):
            slopes = direction + turn * np.array([1.0 - share, -share])[:, None] * basis
            slopes /= np.linalg.norm(slopes, axis=1)[:, None]
            paths = [
                np.r_[p, axis, np.zeros(3), span, 1.0, np.log(np.deg2rad(2.0))]
                for axis, span in zip(slopes, spans)
            ]
            if min(hardware_margin(x) for x in paths) > 1e-5:
                yield np.asarray(paths)


def render(model, paths, order):
    """Keep each electron's countable core separate from its tail."""
    parts, gradients = [], []
    for path in paths:
        light, derivative = raw(model.pos, model.normals, path, order, False, "electron")
        length = 1000.0 * path[9]
        parts.append(length * light)
        gradients.append(length * derivative.sum(0)[:, 9] + 1000.0 * light.sum(0))
    return np.asarray(parts), np.asarray(gradients)


def optimize(model, paths, fixed, response, order, budget):
    """Refit stopping ranges under shared, calibrated optical response."""
    water = material("electron")
    low = float(water["range_at"](water["energy"][0] + 1.0)) / 1000.0
    high = []
    for p in paths:
        chord = allowed_chord(p[:3], p[3:6])
        if chord is None or chord[0] > 0.0 or chord[1] <= low:
            return None
        high.append(min(chord[1] * 0.997, water["maximum_range_mm"] / 1000.0 * 0.999))
        if high[-1] <= low:
            return None
    p = paths.copy()

    def value(z):
        p[:, 9] = np.exp(z)
        parts, jac = render(model, p, order)
        ll, slope = model.obs.score(fixed + response * parts.sum((0, 1)), model.ix)
        return -ll, response * (jac @ slope) * p[:, 9]

    limits = np.log(np.column_stack([np.full(len(p), low), high]))
    initial = np.clip(np.log(p[:, 9]), *limits.T)
    fit = minimize(
        value,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=limits,
        options=dict(maxfun=budget, maxiter=budget, ftol=2e-8),
    )
    p[:, 9] = np.exp(fit.x)
    if min(hardware_margin(x) for x in p) <= 1e-5:
        return None
    return p


def estimates(
    event, model, geometry, shapes, amplitudes, section, budget=16, *, refine_arms=False, resolve_times=False
):
    """Fit local electromagnetic light and report how many cores are needed."""
    if event.timing_selection["prompt_window_ns"] is None:
        return []
    old_time = arrival(event, geometry)
    check = intensity_fit(event, model, shapes, amplitudes, old_time, local_clock_ring=section)
    if check is None:
        return []
    index = section + 1
    fixed = amplitudes @ shapes - amplitudes[index] * shapes[index]
    center = np.asarray(geometry["vertices"][section]) / 1000.0
    direction = np.asarray(geometry["axes"][section], dtype=float)
    direction /= np.linalg.norm(direction)
    modes = []
    threshold = material("electron")["margin_mm"] / 1000.0
    for paths in templates(center, direction, amplitudes[index]):
        parts, _ = render(model, paths, 4)
        emission = parts.sum((0, 1))
        if emission.sum() <= 0.0:
            continue
        scale = model.response * emission.sum()
        if 0.35 * amplitudes[index] <= scale <= 2.8 * amplitudes[index]:
            mean = fixed + np.clip(amplitudes[index], 0.8 * scale, 1.2 * scale) * emission / emission.sum()
            ll = model.obs.score(mean, model.ix)[0]
            submargin = int(np.count_nonzero(paths[:, 9] < threshold))
            modes.append((ll, paths, submargin))
    unique = {}
    for _, paths, count in sorted(modes, key=lambda row: row[0], reverse=True):
        unique.setdefault(count, paths)
    results = []
    weight = np.minimum(event.charges_pe, 2.0)
    observed = amplitudes[:, None] * shapes[:, model.ix]
    old = observed[index] / observed.sum(0)
    if refine_arms:
        from conversion_arms import optimize as optimize_pair
    else:
        optimize_pair = optimize
    for paths in unique.values():
        p = optimize_pair(model, paths, fixed, model.response, 8, budget)
        if p is None:
            continue
        low, high = [render(model, p, order)[0] for order in (16, 32)]
        charge = high.sum((0, 1))
        norm = charge.sum()
        if norm <= 0.0:
            continue
        basis = shapes.copy()
        basis[index] = charge / norm
        physical = norm * model.response
        initial = amplitudes.copy()
        initial[index] = physical
        flight = np.array([photon_time(event, model.normals[model.ix], [x], "electron") for x in p])
        timing_components = None
        clock_columns = section
        if resolve_times:
            # First arrival is an order statistic of the two arms. A
            # weighted mean of their flight times does not preserve it.
            # Both start at the conversion, so share a clock and response.
            clock_columns = np.arange(section, section + len(flight))
            time = np.column_stack([old_time[:, :section], flight.T, old_time[:, section + 1 :]])
            groups = np.r_[np.arange(index), np.repeat(index, len(flight)), np.arange(index + 1, len(basis))]
            parts = np.vstack([basis[:index], high.sum(1) / norm, basis[index + 1 :]])
            timing_components = groups, parts
        else:
            signal = high.sum(1)[:, model.ix]
            time = old_time.copy()
            time[:, section] = np.divide(
                (flight * signal).sum(0),
                signal.sum(0),
                out=old_time[:, section].copy(),
                where=signal.sum(0) > 1e-20,
            )
        measured = intensity_fit(
            event,
            model,
            basis,
            initial,
            time,
            amplitude_bounds={index: (0.8 * physical, 1.2 * physical)},
            local_clock_ring=clock_columns,
            timing_components=timing_components,
        )
        if measured is None:
            continue
        ll, amplitude, clock = measured
        scale = amplitude[index] / norm
        mean = amplitude @ basis
        predicted = mean[model.ix]
        convergence = abs(
            model.obs.score(mean + scale * (low.sum((0, 1)) - charge), model.ix)[0]
            - model.obs.score(mean, model.ix)[0]
        )
        if convergence >= 0.01:
            continue
        fractions = (amplitude[:, None] * basis[:, model.ix] / predicted).T
        overlap = float(weight @ np.minimum(old, fractions[:, index]) / max(weight @ old, 1e-20))
        if overlap < 0.6:
            continue
        cores = scale * high[:, 0][:, model.ix] / predicted
        support = np.array(
            [np.sum(1.0 - model.obs.no_source(predicted, np.clip(f, 0.0, 1.0))) for f in cores]
        )
        measured_parts = fractions[:, np.arange(len(shapes)) != index]
        visible = cores[support >= 30.0].T
        measured_parts[:, 0] += fractions[:, index] - visible.sum(1)
        measured_parts = np.column_stack([measured_parts, visible])
        # A one-direction template has two ranges and an explicit spread.
        # Keep the score difference visible for validation on each mode.
        results.append(
            dict(
                ring=section,
                cores=int((support >= 30.0).sum()),
                support=support,
                charge_fractions=measured_parts,
                joint_gain=float(ll - check[0]),
                reference_joint_ll=float(check[0]),
                charge_ll=float(model.obs.score(mean, model.ix)[0]),
                overlap=overlap,
                quadrature_ll_error=float(convergence),
                means=predicted,
                relative_clock_ns=clock,
                paths=p,
            )
        )
    return results
