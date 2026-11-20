"""Profile light along one coherent trajectory before testing residual rings."""

import numpy as np
from scipy.optimize import minimize

from readout import observation
from throughgoing import line_times
from wctering.likelihood.crossing import intervals
from wctering.likelihood.geometry import active_geometry
from wctering.receiver.angular_panels import finite_integrate
from wctering.receiver.finite_source import quadrature
from wctering.seeding.coherent_timing import C, N_GROUP


def profile(shapes, obs, ix, start, budget=72):
    """Fit bounded photon intensities in linear coordinates at fixed geometry."""
    scale = max(obs.q.sum() / obs.gain, 1.0)
    maximum = 1e6 / scale

    def objective(a):
        ll, derivative = obs.score(scale * (a @ shapes), ix)
        return -ll / scale, shapes @ derivative

    result = minimize(
        objective,
        np.asarray(start) / scale,
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-10, maximum)] * len(shapes),
        options=dict(maxiter=budget, ftol=1e-12, gtol=1e-8),
    )
    value, gradient = objective(result.x)
    blocked = ((result.x <= 1.01e-10) & (gradient >= 0)) | ((result.x >= maximum) & (gradient <= 0))
    error = float(np.max(np.abs(gradient[~blocked]), initial=0.0))
    if error > 5e-6:
        # A nearly empty longitudinal bin can make L-BFGS-B stop on its
        # objective tolerance. Restart in the same positive intensity box.
        result = minimize(
            objective,
            result.x,
            jac=True,
            method="SLSQP",
            bounds=[(1e-10, maximum)] * len(shapes),
            options=dict(maxiter=100, ftol=1e-13),
        )
        value, gradient = objective(result.x)
        blocked = ((result.x <= 1.01e-10) & (gradient >= 0)) | ((result.x >= maximum) & (gradient <= 0))
        error = float(np.max(np.abs(gradient[~blocked]), initial=0.0))
    return -value * scale, result.x * scale, error


def profile_chord(event, chord, bg):
    """Allow smooth brightness variation and diffuse light on the same axis."""
    ids, positions, normals = active_geometry()
    ix = np.searchsorted(ids, event.pmt_ids)
    obs = observation(event.charges_pe)
    anchor = np.asarray(chord["anchor_m"])
    axis = np.asarray(chord["axis"])
    theta, width = np.deg2rad([chord["opening_deg"], chord["width_deg"]])
    paths = chord.get("emitting_intervals")
    if paths is None:
        paths = [(lo["t"], hi["t"]) for lo, hi in intervals(anchor, axis)]
    paths = [(lo, hi) for lo, hi in paths if hi - lo >= 0.01]
    raw = []
    for broad in (width, 3 * width):
        for low, high in paths:
            # Brightness over four adjoining sections is a nuisance of one
            # trajectory. The broad halo shares its axis and location.
            edges = np.linspace(low, high, 5 if broad == width else 2)
            for left, right in zip(edges[:-1], edges[1:]):
                mid, length = 0.5 * (left + right), right - left
                x = np.r_[anchor + mid * axis, axis, length, theta, np.log(broad)]
                raw.append(
                    (broad == width, length * finite_integrate(positions, normals, x, *quadrature(8))[0])
                )
    if not raw:
        return None
    core = np.asarray([core for core, _ in raw])
    raw = np.asarray([value for _, value in raw])
    integral = raw.sum(1)
    keep = integral > 1e-300
    core, raw, integral = core[keep], raw[keep], integral[keep]
    if not core.any() or core.all():
        return None
    shapes = np.vstack([bg, raw / integral[:, None]])
    bg_amplitude, total = chord["amplitudes"]
    start = np.r_[bg_amplitude, 0.05 * total * integral / integral.sum()]
    start[1:][core] = 0.95 * total * integral[core] / integral[core].sum()
    ll, amps, error = profile(shapes, obs, ix, start)
    if error > 5e-6:
        return None
    means = amps @ shapes
    ring = amps[1:][core] @ shapes[1:][core]
    fraction = np.clip(ring[ix] / means[ix], 0.0, 1.0)
    support = float(np.sum(1 - obs.no_source(means[ix], fraction)))
    hard_support = int(np.count_nonzero(fraction > 0.5))
    return (
        np.column_stack([1 - fraction, fraction]),
        (shapes, amps, ll),
        dict(
            ll=ll,
            chord_ll=chord["ll"],
            log_likelihood_gain=ll - chord["ll"],
            prompt_components=int(core.sum()),
            projected_gradient=error,
            photon_support=support,
            hard_support=hard_support,
            broad_light_fraction=float(amps[1:][~core].sum() / amps[1:].sum()),
        ),
    )


def timing_evidence(event, chord, vertex, direction, opening, fraction):
    """Compare coherent relative-time residuals for a track and a second cone."""
    q = np.minimum(event.charges_pe, 2.0)
    t = event.times_ns - np.median(event.times_ns)
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355
    main_time, _ = line_times(
        event.pmt_positions_mm,
        np.asarray(chord["anchor_m"]),
        np.asarray(chord["axis"]),
        np.deg2rad(chord["opening_deg"]),
    )
    main = t - main_time
    main = abs((main - np.median(main)) / sigma)
    from charge_starts import weighted_quantile

    def comparison(prediction):
        r = t - prediction
        offset = weighted_quantile(r, q * fraction, [0.5])[0]
        return abs((r - offset) / sigma)

    point = comparison(N_GROUP * np.linalg.norm(event.pmt_positions_mm - 1000 * vertex, axis=1) / C)
    theta = np.clip(opening, np.deg2rad(10.0), np.deg2rad(42.0))
    line = comparison(line_times(event.pmt_positions_mm, vertex, direction, theta)[0])

    # A residual must explain a common subset better under its own timing.
    # Taking the best of two whole source models avoids a per-hit time choice.
    def evidence(residual):
        return float((q * fraction) @ (np.log1p(main**2 / 4) - np.log1p(residual**2 / 4)))

    return max(evidence(point), evidence(line))
