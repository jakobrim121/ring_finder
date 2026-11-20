"""Seed missing rings by subtracting a fitted finite-track response."""

import numpy as np

from bare_residual_search import residual_scores
from charge_starts import allowed_chord, weighted_quantile
from chord_profile import profile
from readout import observation
from residual_search import axis_grid
from wctering.likelihood.geometry import active_geometry
from wctering.receiver.angular_panels import finite_integrate
from wctering.receiver.finite_source import quadrature
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def positions_for_search(event, bank, geometry):
    """Try the measured track endpoints and timing-seeded light centers."""
    points = list(np.asarray(geometry["vertices"]) / 1000)
    if "lengths_mm" in geometry:
        for vertex, axis, length in zip(geometry["vertices"], geometry["axes"], geometry["lengths_mm"]):
            if length is None or length < 40:
                continue
            for u in (-0.5, 0.5):
                points.append((np.asarray(vertex) + u * length * np.asarray(axis)) / 1000)
    for key in ("1", "2", "3"):
        seed = bank.get(key)
        if seed is not None:
            points.extend(np.asarray(seed["vertices"]) / 1000)
    result = []
    for p in points:
        if all(np.linalg.norm(p - x) >= 0.15 for x in result):
            result.append(p)
        if len(result) >= 8:
            break
    return result


def screen(event, bank, geometry, mean, score, shapes, ix, *, residual_vertices=False):
    """Use a calibrated likelihood derivative to select residual directions."""
    _, pos, normals = active_geometry()
    axes = axis_grid(320)
    t = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    peaks = []
    points = positions_for_search(event, bank, geometry)
    if residual_vertices:
        from residual_axes import timed_residual_vertices

        points.extend(
            p
            for p in timed_residual_vertices(event, mean[ix], limit=4)
            if all(np.linalg.norm(p - v) > 0.15 for v in points)
        )
    for point in points:
        delta = pos - 1000 * point
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
        rays = delta / distance[:, None]
        polar = np.arccos(np.clip(axes @ rays.T, -1.0, 1.0))
        accept = np.maximum(-np.sum(normals * rays, axis=1), 0.02) / (distance * distance + 10000.0)
        for angle in np.deg2rad([24.0, 31.0, 37.0, 41.5]):
            basis = accept * np.exp(-0.5 * ((polar - angle) / np.deg2rad(4.0)) ** 2)
            basis /= np.maximum(basis.sum(1, keepdims=True), 1e-300)
            derivative, info, _ = residual_scores(basis, mean, score, shapes)
            signal = np.maximum(derivative, 0.0) ** 2 / np.maximum(info, 1e-30)
            for i in np.argsort(signal)[-2:]:
                if signal[i] < 14.0 or info[i] < 1e-15:
                    continue
                w = basis[i, ix] * q / np.maximum(mean[ix], 0.2)
                if not w.any():
                    continue
                axis = axes[i]
                z = delta[ix] @ axis
                radius = np.linalg.norm(delta[ix] - z[:, None] * axis, axis=1)
                s = z - radius / np.tan(angle)
                time = (N_PHASE * np.cos(angle) * s + N_GROUP * radius / np.sin(angle)) / C
                residual = t - time
                clock = weighted_quantile(residual, w, [0.5])[0]
                prompt = np.exp(-0.5 * ((residual - clock) / 2.0) ** 2)
                coherence = float(w @ prompt / w.sum())
                if coherence < 0.38:
                    continue
                weight = w * prompt
                lo, hi = weighted_quantile(s / 1000, weight, [0.15, 0.85])
                peaks.append(
                    (
                        float(signal[i] * coherence),
                        point,
                        axis,
                        angle,
                        float(derivative[i] / info[i]),
                        lo,
                        hi,
                        clock,
                        prompt,
                    )
                )
    unique = []
    budget = 6 if residual_vertices else 3
    for row in sorted(peaks, key=lambda r: r[0], reverse=True):
        if any(np.linalg.norm(row[1] - old[1]) < 0.25 and row[2] @ old[2] > 0.94 for old in unique):
            continue
        unique.append(row)
        if len(unique) == budget:
            break
    return unique


def fit(
    event,
    bank,
    geometry,
    shapes,
    amps,
    reference,
    *,
    prior_fractions,
):
    """Add one finite source and profile all source intensities together."""
    active, pos, normals = active_geometry()
    ix = np.searchsorted(active, event.pmt_ids)
    obs = observation(event.charges_pe)
    q = np.minimum(event.charges_pe, 2.0)
    mean = amps @ shapes
    _, score = obs.score(mean, ix)
    result = []
    existing = np.sum(shapes[1:, ix] * amps[1:, None], axis=0)
    for _, point, axis, angle, pe, lo, hi, clock, prompt in screen(
        event, bank, geometry, mean, score, shapes, ix, residual_vertices=False
    ):
        water = allowed_chord(point, axis)
        if water is None:
            continue
        low, high = max(lo - 0.035, water[0]), min(hi + 0.035, water[1])
        if high - low < 0.015:
            continue
        x = np.r_[point + 0.5 * (low + high) * axis, axis, high - low, angle, np.log(np.deg2rad(2.5))]
        raw = finite_integrate(pos, normals, x, *quadrature(8))[0]
        if raw.sum() <= 0.0:
            continue
        source = raw / raw.sum()
        basis = np.vstack([shapes, source])
        ll, intensity, error = profile(basis, obs, ix, np.r_[amps, min(pe, event.charges_pe.sum())])
        if error > 1e-5 or ll - reference <= 6.0:
            continue
        means = (intensity @ basis)[ix]
        parts = (intensity[:, None] * basis[:, ix] / means).T
        last = parts[:, -1]
        labels = parts.argmax(1)
        if set(labels) - {0} != set(range(1, parts.shape[1])):
            continue
        owns = labels == parts.shape[1] - 1
        n_prompt = int(np.count_nonzero(owns & (prompt > 0.5)))
        support = float(np.sum(1 - obs.no_source(means, last)))
        unique = int(np.count_nonzero(owns & (existing < 0.25 * means)))
        if support < 30.0 or n_prompt < 18 or unique < 12:
            continue
        refined = finite_integrate(pos, normals, x, *quadrature(16))[0]
        if refined.sum() <= 0.0:
            continue
        basis[-1] = refined / refined.sum()
        refined_ll = obs.score(intensity @ basis, ix)[0]
        if abs(refined_ll - ll) >= 0.01:
            continue
        parts = (intensity[:, None] * basis[:, ix] / (intensity @ basis)[ix]).T
        if set(parts.argmax(1)) - {0} != set(range(1, parts.shape[1])):
            continue
        module_support = len(np.unique(event.pmt_ids[owns & (prompt > 0.5)] // 100))
        if module_support < 3:
            continue
        old = prior_fractions.argmax(1)
        new = parts.argmax(1)
        # The fourth cone needs its own light and must retain nearly
        # all three existing rings. A split of a bright old region
        # alone cannot pass this test.
        retained = [
            float(q[(old == j) & (new == j)].sum() / max(q[old == j].sum(), 1e-15))
            for j in range(1, parts.shape[1] - 1)
        ]
        added = new == parts.shape[1] - 1
        new_charge = float(q[added & (old == 0)].sum())
        if min(retained, default=0.0) < 0.7 or new_charge < 12.0:
            continue
        # A split that mostly reassigns existing ring charge is
        # weaker evidence than light supported by a new footprint.
        # Weight by the added wavefront rather than all late light.
        prompt_charge = q * prompt * added
        if 2.0 * prompt_charge[old == 0].sum() < prompt_charge.sum():
            continue
        result.append(
            (
                refined_ll,
                parts,
                dict(
                    observed_support=support,
                    prompt_pmts=n_prompt,
                    distinct_pmts=unique,
                    module_support=module_support,
                    relative_clock_ns=float(clock),
                    prompt_pe=float(q[owns] @ prompt[owns]),
                    start=x.tolist(),
                ),
            )
        )
    return max(result, key=lambda r: r[0]) if result else None
