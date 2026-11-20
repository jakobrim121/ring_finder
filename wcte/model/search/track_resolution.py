"""Test whether the measured light resolves two successive charged legs."""

import numpy as np

from chain_sources import paths
from charge_starts import allowed_chord, weighted_quantile
from chord_profile import profile
from coherent_partition import prediction_time
from geometry import hardware_margin
from material import material
from source import raw


def profile_continuation(model, state, rival, rival_state, index, event):
    """Use the rival's dominant leg to test an earlier or later endpoint.

    A single-track fit can miss the straight extension of its own ring.
    Refit the interval of the rival's dominant leg before treating excess
    light as an independent trajectory. Use the same material light yield.
    """
    x = state["parameters"]
    particle = model.particles[0]
    z = rival_state["parameters"]
    mapping = paths(z[: rival.ng])
    track = mapping[1 - index][0].copy()
    mass = rival.particles[1 - index]

    def candidates():
        yield particle, x[:12]
        yield mass, track
        limit = allowed_chord(track[:3], track[3:6])
        if limit is None:
            return
        length = track[9] * track[10]
        m = material(mass)
        for extend in (0.10, 0.24, 0.42):
            if index == 0:
                extra = min(extend, -limit[0] - 0.002)
                start = track[:3] - extra * track[3:6]
                ran = track[9] + extra
            else:
                extra = min(extend, limit[1] - length - 0.002)
                start = track[:3]
                ran = track[9]
            if extra <= 0.015 or length + extra > ran or 1000 * ran >= m["maximum_range_mm"]:
                continue
            proposed = track.copy()
            proposed[:3], proposed[9], proposed[10] = start, ran, (length + extra) / ran
            yield mass, proposed

    best = None
    for mass, p in candidates():
        if hardware_margin(p) <= 1e-5:
            continue
        length = 1000 * p[9] * p[10]
        light = length * raw(model.pos, model.normals, p, 16, False, mass)[0].sum(0)
        norm = light.sum()
        if norm <= 0.0:
            continue
        shapes = np.vstack([model.bg, light / norm])
        ll, amp, error = profile(shapes, model.obs, model.ix, np.exp(x[-2:])[::-1] * [1.0, norm], budget=36)
        if error <= 1e-5 and (best is None or ll > best[0]):
            best = ll, mass, p, amp
    if best is None:
        return None
    ll, mass, p, amp = best
    length = 1000 * p[9] * p[10]
    light = length * raw(model.pos, model.normals, p, 32, False, mass)[0].sum(0)
    if light.sum() <= 0.0:
        return None
    mean = amp[0] * model.bg + amp[1] * light / light.sum()
    new_ll = model.obs.score(mean, model.ix)[0]
    if abs(new_ll - ll) > 0.01:
        return None
    fraction = amp[1] * light[model.ix] / (light.sum() * mean[model.ix])
    flight, visible = prediction_time(event, np.r_[p, x[-2:]], (mass,), event.charges_pe)
    return new_ll, fraction, flight[:, 0], visible[:, 0], p


def competing_footprint(event, one, two, fractions):
    """Require a resolved short section outside the best one-track wavefront."""
    model, state = one
    rival, rival_state = two
    if fractions.shape[1] != 3 or len(rival.particles) != 2:
        return None
    q = np.minimum(event.charges_pe, 2.0)
    labels = fractions.argmax(1)
    short = int(np.argmin(q @ fractions[:, 1:]))
    null = profile_continuation(model, state, rival, rival_state, short, event)
    if null is None:
        return None
    null_ll, single, flight, visible, path = null
    relative = event.times_ns - np.median(event.times_ns)
    weight = q * single * (visible > 1e-6)
    if weight.sum() < 30.0:
        return None
    residual = relative - flight
    clock = weighted_quantile(residual, weight, [0.5])[0]
    on_line = single * np.exp(-0.5 * ((residual - clock) / 2.0) ** 2)
    times, illumination = prediction_time(event, rival_state["parameters"], rival.particles, event.charges_pe)
    res = relative[:, None] - times
    optical = fractions[:, 1:] * (illumination > 1e-6)
    weight = q[:, None] * optical
    if not weight.any():
        return None
    clock = weighted_quantile(res.ravel(), weight.ravel(), [0.5])[0]
    short_prompt = optical[:, short] * np.exp(-0.5 * ((res[:, short] - clock) / 2.0) ** 2)
    owned = labels == short + 1
    independent = owned & (short_prompt > 0.35) & (on_line < 0.25)
    if np.count_nonzero(independent) < 18:
        return None
    modules = np.unique(event.pmt_ids[independent] // 100, return_inverse=True)[1]
    total = np.bincount(modules, weights=q[independent] * short_prompt[independent])
    if len(total) < 3 or total.sum() - total.max() < 12.0:
        return None
    return dict(
        short_section=short + 1,
        independent_pmts=int(independent.sum()),
        independent_prompt_pe=float(total.sum()),
        single_track_ll=float(null_ll),
        single_track_path_m=path.tolist(),
    )
