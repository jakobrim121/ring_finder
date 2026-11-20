"""Test a long, coherent water chord before fragmenting its light into rings."""

import numpy as np
from scipy.optimize import minimize

from readout import observation, profile
from wctering.likelihood.crossing import intervals
from wctering.likelihood.geometry import active_geometry, common_background
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def line_times(positions, center, direction, opening):
    """Arrival times up to an event-wide offset for an emitting straight track."""
    delta = positions - center * 1000
    z = delta @ direction
    radius = np.linalg.norm(delta - z[:, None] * direction, axis=1)
    distance = radius / np.sin(opening)
    emission = z - distance * np.cos(opening)
    inverse_beta = N_PHASE * np.cos(opening)
    return (inverse_beta * emission + N_GROUP * distance) / C, emission / 1000


def spectrum_model(positions, normals, center, direction, opening, width, order):
    """Use receiver-aware optical panels for a nearly constant-speed chord."""
    water = intervals(center, direction)
    # A particle can pass through hardware with no water light in the gap.
    # Normalize the pieces by their raw optical integrals, not their lengths.
    from wctering.receiver.angular_panels import finite_integrate
    from wctering.receiver.finite_source import quadrature

    combined = np.zeros(len(positions))
    for lo, hi in water:
        mid = 0.5 * (lo["t"] + hi["t"])
        length = hi["t"] - lo["t"]
        x = np.r_[center + mid * direction, direction, length, opening, np.log(width)]
        raw = finite_integrate(positions, normals, x, *quadrature(order))[0]
        combined += length * raw
    return combined / max(combined.sum(), 1e-300), [(lo["t"], hi["t"]) for lo, hi in water]


def observed_intervals(s, weights, water):
    """Fraction of observed light near each end of its allowed water interval."""
    lo = min(a for a, _ in water)
    hi = max(b for _, b in water)
    span = hi - lo
    return [
        float(weights[(s >= a) & (s <= b)].sum() / max(weights.sum(), 1e-20))
        for a, b in [(lo - 0.08, lo + 0.18 * span), (hi - 0.18 * span, hi + 0.08)]
    ]


def starts(event, bank):
    """Try both line orientations using the already computed detector cones."""
    result = []
    for key in (1, 2, 3):
        geometry = bank.get(key, bank.get(str(key)))
        if geometry is None:
            continue
        for vertex, axis in zip(geometry["vertices"], geometry["axes"]):
            v = np.asarray(vertex, float) / 1000
            a = np.asarray(axis, float)
            a = a / np.linalg.norm(a)
            for direction in (a, -a):
                if any(
                    direction @ b > 0.99 and np.linalg.norm(np.cross(v - u, direction)) < 0.12
                    for u, b in result
                ):
                    continue
                result.append((v, direction))
    # Timing orders starting points by a free-clock wavefront residual.
    # The optical fit can then move the line before doing any count check.
    positions = event.pmt_positions_mm
    t = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)

    def loss(row):
        model, _ = line_times(positions, *row, np.deg2rad(41.5))
        r = t - model
        return q @ np.log1p((r - np.median(r)) ** 2 / 4)

    return sorted(result, key=loss)[:3]


def fit(event, bank, *, allow_stopping=False):
    """Fit a physically distinct throughgoing null and record its diagnostics."""
    minimum = 70 if allow_stopping else 250
    if len(event.pmt_ids) < minimum or (1 not in bank and "1" not in bank):
        return None
    active, positions, normals = active_geometry()
    ix = np.searchsorted(active, event.pmt_ids)
    q = event.charges_pe
    primary = bank.get(1, bank.get("1"))
    bg = common_background(np.asarray(primary["vertices"][0]))
    obs = observation(q)
    t = event.times_ns - np.median(event.times_ns)
    time_sigma = np.sqrt((1.5**2 + 2.1**2 / np.maximum(q, 0.25))) / 2.355
    z = np.maximum(q, 0.1)
    outcomes = []
    for anchor, direction in starts(event, bank):
        unit = np.eye(3)[np.argmin(abs(direction))]
        perpendicular = np.cross(direction, unit)
        perpendicular /= np.linalg.norm(perpendicular)
        basis = np.column_stack([perpendicular, np.cross(direction, perpendicular)])
        # Put the anchor at the line point nearest the tank origin. Its
        # translation along the line is redundant with the event clock.
        anchor = anchor - (anchor @ direction) * direction

        def unpack(x):
            a = direction + basis @ x[2:4]
            return anchor + basis @ x[:2], a / np.linalg.norm(a), x[4], x[5]

        def timing(x):
            c, a, theta, _ = unpack(x)
            mean, s = line_times(event.pmt_positions_mm, c, a, theta)
            residual = t - mean
            clock = np.median(residual)
            error = (residual - clock) / time_sigma
            # Robust timing is a start locator. Scattered photons and dark
            # noise can arrive well after the direct wavefront.
            loss = np.sum(np.log1p(error**2 / 4))
            return loss, error, s

        angle = np.deg2rad([19.0 if allow_stopping else 35.0, 42.5])
        opening = (
            float(np.clip(np.deg2rad(primary["openings_deg"][0]), *angle))
            if allow_stopping
            else np.deg2rad(41.5)
        )
        y0 = np.r_[np.zeros(4), opening, np.deg2rad(2.0)]
        bounds = [(-0.6, 0.6)] * 2 + [(-0.5, 0.5)] * 2 + [tuple(angle), tuple(np.deg2rad([0.8, 5.0]))]
        proposed = minimize(
            lambda x: timing(x)[0], y0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=24, ftol=1e-7)
        ).x
        for x in (y0, proposed):
            c, a, theta, width = unpack(x)
            try:
                shape, water = spectrum_model(positions, normals, c, a, theta, width, 8)
            except (ValueError, ArithmeticError):
                continue
            loss, errors, s = timing(x)
            candidates = [(shape, water, "water_chord")]
            if allow_stopping:
                from wctering.receiver.angular_panels import finite_integrate
                from wctering.receiver.finite_source import quadrature

                strong = np.minimum(z, 2.0) * (abs(errors) < 2.5)
                for low, high in water:
                    in_water = (s >= low - 0.04) & (s <= high + 0.04)
                    if np.count_nonzero(in_water & (strong > 0.25)) < 30:
                        continue
                    from charge_starts import weighted_quantile

                    lo, hi = weighted_quantile(s[in_water], strong[in_water], [0.08, 0.92])
                    lo, hi = max(low, lo - 0.04), min(high, hi + 0.04)
                    if hi - lo < 0.10:
                        continue
                    p = np.r_[c + 0.5 * (lo + hi) * a, a, hi - lo, theta, np.log(width)]
                    raw = finite_integrate(positions, normals, p, *quadrature(8))[0]
                    candidates.append((raw / max(raw.sum(), 1e-300), [(lo, hi)], "timed_finite_track"))
            for shape, water, family in candidates:
                if shape.sum() < 0.9:
                    continue
                outcome = assess(bg, shape, ix, q, z, errors, s, water, c, a, theta, width, obs, loss, family)
                if outcome is not None:
                    outcomes.append(outcome)
    return max(outcomes, key=lambda r: r[0]) if outcomes else None


def assess(bg, shape, ix, q, z, errors, s, water, c, a, theta, width, obs, loss, family):
    posterior, score = profile(np.vstack([bg, shape]), ix, q)
    if not score["qualified"]:
        return None
    inside = np.zeros(len(s), bool)
    for lo, hi in water:
        inside |= (s >= lo - 0.08) & (s <= hi + 0.08)
    amps = np.asarray(score["amplitudes"])
    means = amps @ np.vstack([bg, shape])
    coherent = posterior[1]
    in_time = abs(errors) < 3.0
    direct = np.minimum(z, 2.0) * coherent
    weights = direct * inside * in_time
    coverage = float(weights.sum() / np.minimum(z, 2.0).sum())
    photon_support = float(np.sum(1 - obs.no_source(means[ix], coherent)))
    if photon_support < 30 or not np.any(coherent > 0.5):
        return None
    limits = np.linspace(water[0][0], water[-1][1], 5)
    longitudinal = np.histogram(s, bins=limits, weights=weights)[0]
    summary = dict(
        family=family,
        count=1,
        ll=score["ll"],
        time_loss=float(loss),
        coherent_coverage=coverage,
        core_support=photon_support,
        chord_m=sum(hi - lo for lo, hi in water),
        chord_end_support=observed_intervals(s, weights, water),
        longitudinal_quarter_fractions=(longitudinal / max(longitudinal.sum(), 1e-20)).tolist(),
        coherent_longitudinal_span_m=float(np.diff(np.quantile(s[weights > 0.5], [0.05, 0.95]))[0])
        if np.count_nonzero(weights > 0.5) >= 30
        else 0.0,
        emitting_intervals=water,
        amplitudes=amps.tolist(),
        axis=a.tolist(),
        anchor_m=c.tolist(),
        opening_deg=float(np.rad2deg(theta)),
        width_deg=float(np.rad2deg(width)),
        prompt_relative_only=True,
    )
    return score["ll"], posterior.T, summary
