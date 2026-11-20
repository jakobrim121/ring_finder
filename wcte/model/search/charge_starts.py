"""Estimate emitting lengths before starting the calibrated trajectory fit."""

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import permutations
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, lsq_linear

from chain_soft_search import canonical
from chain_sources import interactions, paths
from material import material
from source import raw
from wctering.likelihood.crossing import intervals
from wctering.likelihood.geometry import BOX
from wctering.receiver.point_source import point_source


_FIXED_RECOIL = ContextVar("fixed_recoil", default=False)


@contextmanager
def fixed_recoil(enabled):
    token = _FIXED_RECOIL.set(enabled)
    try:
        yield
    finally:
        _FIXED_RECOIL.reset(token)


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return np.interp(np.asarray(quantiles) * cdf[-1], cdf, values[order])


def optical_charge(geometry, model):
    """Use cone responsibilities only to initialize each section's yield."""
    position = model.pos[model.ix]
    signal = []
    spans = geometry.get("lengths_mm", [0.0] * len(geometry["axes"]))
    for vertex, axis, opening, width, length in zip(
        geometry["vertices"], geometry["axes"], geometry["openings_deg"], geometry["widths_deg"], spans
    ):
        centre = np.r_[np.asarray(vertex) / 1000, axis]
        optics = [np.deg2rad(opening), np.log(np.deg2rad(width))]
        # Normalize across all PMTs. Fitted finite sources retain their
        # finite light pattern when assigning observed PE to initial legs.
        if length is not None and length > 0.0:
            from wctering.receiver.angular_source import source

            shape, _ = source(model.pos, model.normals, np.r_[centre, length / 1000, optics], 8)
        else:
            shape, _ = point_source(model.pos, model.normals, np.r_[centre, optics])
        signal.append(shape[model.ix])
    weights = np.asarray(geometry["weights"])
    response = weights[:, None] * np.vstack([model.bg[model.ix], signal])
    responsibilities = response[1:] / np.maximum(response.sum(0), 1e-300)
    return position / 1000, responsibilities * model.q / model.obs.gain


def allowed_chord(centre, direction):
    """Take the contiguous water interval closest to the optical centre."""
    try:
        water = intervals(centre, direction)
    except (ValueError, ArithmeticError):
        return None
    a, b = min(water, key=lambda p: max(p[0]["t"], -p[1]["t"], 0.0))
    return a["t"] + 1e-3, b["t"] - 1e-3


def nominal_sections(geometry, model, event=None):
    """Cache detector-derived light and geometry shared by latent masses."""
    centers = np.asarray(geometry["vertices"], float) / 1000
    axes = np.asarray(geometry["axes"], float)
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    position, observed = optical_charge(geometry, model)
    spans = geometry.get("lengths_mm", [None] * len(axes))
    observations = []
    for centre, axis, opening, span, q in zip(centers, axes, geometry["openings_deg"], spans, observed):
        chord = allowed_chord(centre, axis)
        if chord is None or chord[1] - chord[0] < 0.005 or q.sum() <= 0.0:
            return None
        maximum = np.arccos(1 / material("muon")["index"])
        theta = np.clip(np.deg2rad(opening), 0.03, maximum - 0.004)
        # A straight ray hitting at the fitted Cherenkov angle fixes its
        # position of emission along the axis, with no event clock involved.
        delta = position - centre
        along = delta @ axis
        radial = np.linalg.norm(delta - along[:, None] * axis, axis=1)
        emission = along - radial / np.tan(theta)
        weights = np.minimum(q, 5.0)
        if event is not None:
            from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE

            time = 1000 * (N_PHASE * np.cos(theta) * emission + N_GROUP * radial / np.sin(theta)) / C
            t = event.times_ns - time
            clock = weighted_quantile(t, weights, [0.5])[0]
            # A broad time weight locates the spatial support of direct
            # light. It leaves a floor for small or scattered ring segments.
            weights = weights * (0.2 + 0.8 * np.exp(-0.5 * ((t - clock) / 2.8) ** 2))
        start, stop = weighted_quantile(emission, weights, [0.1, 0.9])
        middle = np.clip(0.5 * (start + stop), *chord)
        length = max(stop - start, 0.015, (span or 0.0) / 1000)
        if geometry.get("emission_model") == "full_water_line":
            length = chord[1] - chord[0]
        observations.append((centre, axis, chord, theta, middle, length, q.sum()))

    @lru_cache(maxsize=None)
    def section(j, particle):
        centre, axis, chord, theta, middle, length, photons = observations[j]
        m = material(particle)
        gamma = 1 / np.sqrt(1 - (m["index"] * np.cos(theta)) ** -2)
        energy = np.clip(m["mass"] * (gamma - 1), m["energy"][0] + 1.01, m["energy"][-1])
        midpoint_range = float(m["range_at"](energy)) / 1000
        # The ray inversion spreads timing/angle noise along the line. Use
        # the absolute PE calibration to obtain a second extent estimate.
        sample = np.r_[
            centre + middle * axis, axis, np.zeros(3), midpoint_range, 0.0001, np.log(np.deg2rad(1.3))
        ]
        # The brightness estimate is an initializer. Use the same optical
        # quadrature on a short sample; the optimizer refits all energies.
        power = model.response * raw(model.pos, model.normals, sample, 1, False, particle)[0].sum() * 1000
        from_charge = photons / max(power, 1e-8)
        return centre, axis, chord, middle, length, from_charge, midpoint_range

    return section


def initializations(section, particles, normalization, *, fit_geometry=lsq_linear):
    """Initialize the track with both a PE-derived and optical extent."""
    sections = [section(j, particle) for j, particle in enumerate(particles)]
    for use_charge in (True, False):
        centres, axes, starts, ends, remaining = [], [], [], [], []
        for centre, axis, (low, high), middle, optical, charge, ran in sections:
            length = np.clip(charge if use_charge else optical, 0.005, high - low)
            start = np.clip(middle - length / 2, low, high - length)
            starts.append(centre + start * axis)
            ends.append(centre + (start + length) * axis)
            centres.append((starts[-1] + ends[-1]) / 2)
            axes.append(axis)
            remaining.append(ran)
        n = len(axes)
        # Connected hypotheses first try orderings with the smallest gaps
        # between consecutive optical endpoints. Fit one common polyline.
        ranked = sorted(
            permutations(range(n)),
            key=lambda p: sum(np.linalg.norm(ends[a] - starts[b]) ** 2 for a, b in zip(p, p[1:])),
        )
        for order in ranked[:2]:
            a = np.asarray(axes)[list(order)]
            centres_ordered = np.asarray(centres)[list(order)]
            lengths = np.linalg.norm(np.asarray(ends) - starts, axis=1)[list(order)]
            if n == 1:
                vertex = starts[0]
            else:
                # Use light centroids and noisy length estimates as soft
                # constraints, with a modest regularizer on the lengths.
                matrix = np.zeros((4 * n, 3 + n))
                for j in range(n):
                    matrix[3 * j : 3 * j + 3, :3] = np.eye(3)
                    matrix[3 * j : 3 * j + 3, 3 : 3 + j] = a[:j].T
                    matrix[3 * j : 3 * j + 3, 3 + j] = 0.5 * a[j]
                matrix[3 * n :, 3:] = 0.5 * np.eye(n)
                result = fit_geometry(
                    matrix,
                    np.r_[centres_ordered.ravel(), 0.5 * lengths],
                    bounds=([v[0] + 1e-4 for v in BOX] + [0.005] * n, [v[1] - 1e-4 for v in BOX] + [3.5] * n),
                    tol=1e-7,
                )
                vertex, lengths = result.x[:3], result.x[3:]
            ordered_particles = tuple(particles[i] for i in order)
            x = []
            for j, i in enumerate(order):
                limit = material(particles[i])["maximum_range_mm"] / 1000
                ran = min(max(remaining[i] + 0.5 * lengths[j], lengths[j] * 1.00001), limit)
                if ran <= lengths[j]:
                    break
                x.append(np.r_[a[j], np.zeros(3), ran, lengths[j] / ran, np.log(np.deg2rad(1.3))])
            if len(x) == n:
                yield ordered_particles, np.concatenate([vertex, *x, normalization])


def recoil_start(model, x):
    """Raise incoming ranges just enough to make the same spatial path feasible."""
    x = canonical(model, x)
    if x is None:
        return None
    for j in reversed(range(len(model.particles) - 1)):
        index = 9 + 9 * j
        length = x[index] * x[index + 1]
        cached = _FIXED_RECOIL.get()
        if cached:
            if cached == "local":
                from seed_recoil_value import prepare
            else:
                from recoil_repair import prepare

            evaluate = prepare(x[: model.ng], model.particles, j)

            def surplus(ran):
                x[index : index + 2] = ran, length / ran
                return evaluate(ran, x[index + 1])
        else:

            def surplus(ran):
                x[index : index + 2] = ran, length / ran
                return interactions(paths(x[: model.ng]), model.particles)[0][j] - 1e-7

        start = x[index]
        if surplus(start) >= 0.0:
            continue
        high = model.bounds[index][1] * (1 - 1e-10)
        if high <= start or surplus(high) < 0.0:
            return None
        surplus(brentq(surplus, start, high, xtol=1e-9))
    return x
