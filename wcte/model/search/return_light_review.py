"""Check backward light left by a proposed demotion to two rings."""

import numpy as np

from beam_window_light import Window, project
from chain_comparison import MARGIN
from diffuse_timing import fit_mixture
from gated_photon_time import log_density
from geometry import hardware_margin
from module_supported_ring import lower_bound
from predictive_score import log_probabilities
from resolved_ring_evidence import first_pulse_support
from serial_refinement import leg_starts


def nominations(compare, restored, cones):
    """Prepare a new leg where a measured path turns toward its entrance."""
    model = compare.model
    fit, state, _ = restored
    for section, timing_only, merit, _, point, axis, opening, length, birth in cones:
        if section != 1 or timing_only or axis[2] >= 0.0:
            continue
        mass = fit.particles[-1]
        if mass == "electron":
            continue
        photons = float(np.clip(np.sqrt(merit) * 6.0, 20.0, model.total))
        for child in leg_starts(model, point, axis, opening, photons, length, mass):
            path = np.r_[point, child]
            if hardware_margin(path) > 1e-5:
                yield mass, path, birth


def measure(compare, background, proposal=None, *, order=16):
    """Profile only the omitted source and its direct and reflected PE."""
    model, event = compare.model, compare.event
    basis, groups, pieces, ray_time, amplitude = background
    variances = np.zeros_like(ray_time)
    core_mask = 0
    if proposal is not None:
        mass, missing, birth = proposal
        intensity, centers, variance = project(model, missing, mass, order, window=Window())
        norm = intensity.sum()
        if norm * model.response < 5.0 or intensity[[0, 2]][:, model.ix].sum() <= 1e-16:
            return None
        retain = intensity[:, model.ix].sum(1) > 1e-18
        core_mask = sum(
            1 << (ray_time.shape[1] + j) for j, k in enumerate(np.flatnonzero(retain)) if k % 2 == 0
        )
        groups = np.r_[groups, np.full(retain.sum(), len(basis))]
        basis = np.vstack([basis, intensity.sum(0) / norm])
        pieces = np.vstack([pieces, intensity[retain] / norm])
        ray_time = np.column_stack([ray_time, centers[:, retain] + birth])
        variances = np.column_stack([variances, variance[:, retain]])
        amplitude = np.r_[amplitude, model.response * norm]
    scales = np.exp(np.asarray(model.bounds[-2])) / model.response
    bounds = {j: tuple(amplitude[j] * scales) for j in range(1, len(amplitude))}
    fit = fit_mixture(
        event,
        model,
        basis,
        amplitude,
        ray_time,
        return_detail=True,
        amplitude_bounds=bounds,
        clock_groups=np.zeros(ray_time.shape[1], dtype=int),
        timing_components=(groups, pieces),
        variance=variances,
        clock_bounds=(-0.65, 0.65),
        iterations=26,
    )
    if fit is None:
        return None
    ll, yield_pe, time = fit
    rate = (pieces * yield_pe[groups, None])[:, model.ix].T
    by_tube = log_probabilities(model.obs, yield_pe @ basis, model.ix)[0]
    by_tube[model.ix] += log_density(
        compare.time,
        *compare.gate,
        time,
        rate,
        compare.sigma,
        model.obs.offsets,
        model.obs.coefficients,
        variance=variances,
    )
    scores = np.bincount(compare.modules, weights=by_tube)
    if proposal is None:
        return float(ll), None, 0.0, scores
    support = first_pulse_support(
        compare.time,
        compare.gate,
        rate,
        time,
        compare.sigma,
        -1,
        model.obs.offsets,
        model.obs.coefficients,
        excluded_mask=core_mask,
        variance=variances,
    )
    # A reflected burst has sparse first-PE occupancy. Compare the light
    # over the observed modules as one extended burst and account for
    # the extra local track coordinates. The larger fit is retained
    # when the new light is substantial and independently distributed.
    return float(ll), lower_bound(event.pmt_ids, support), float(yield_pe[-1]), scores


def check(compare, measured, restored, cones, *, clock=None):
    """Recheck intense residual light from a contained backwards section."""
    from transport_count_review import responses

    fit, state, _ = restored
    model = compare.model
    candidates = list(nominations(compare, restored, cones))
    if not candidates:
        return False, []
    response = responses(compare.event, model, measured.record, restored)
    if response is None:
        return False, []
    old_shape = np.zeros((len(fit.particles) + 1, len(model.pos)))
    np.add.at(old_shape, response.groups, response.shape)
    amp = old_shape.sum(1)
    if np.any(amp <= 1e-18):
        return False, []
    timing = response.time
    # Absorb the scalar fitted clock into absolute predictions in the
    # event's relative time frame. The residual starts share this clock.
    if clock is None:
        from physical_pulse import profile

        fit_clock = profile(compare.event, fit, state, separate_knockon=True, timing_only=True)
        if fit_clock is None:
            return False, []
        clock = fit_clock["clock_ns"]
    old = (
        old_shape / amp[:, None],
        response.groups,
        response.shape / amp[response.groups, None],
        timing + clock,
        amp,
    )
    control = measure(compare, old)
    if control is None:
        return False, []
    margin = 6.0 * compare.log_n + MARGIN
    reports = []
    for plan in candidates:
        coarse = measure(compare, old, plan, order=8)
        if coarse is None:
            continue
        _, support, count, scores = coarse
        delta = 2.0 * (scores - control[3])
        report = dict(
            gain=float(delta.sum() - delta.max(initial=0.0) - margin),
            support=support,
            yield_pe=count,
            path=plan[1].tolist(),
        )
        reports.append(report)
        if report["gain"] <= MARGIN or count <= 30.0:
            continue
        high = measure(compare, old, plan, order=16)
        if high is None:
            continue
        confirmed = measure(compare, old, plan, order=32)
        if confirmed is None:
            continue
        error = abs(high[0] - confirmed[0])
        if error >= 0.01:
            high, confirmed = confirmed, measure(compare, old, plan, order=64)
            if confirmed is None:
                continue
            error = abs(high[0] - confirmed[0])
        delta = 2.0 * (confirmed[3] - control[3])
        gain = delta.sum() - delta.max(initial=0.0) - margin
        report.update(
            refined_gain=float(gain),
            refined_support=confirmed[1],
            quadrature_ll_error=error,
            effective_gain_modules=int(np.count_nonzero(delta > MARGIN)),
        )
        # This is a lack-of-fit veto, so it preserves the existing count.
        # It does not create a new counted ring below the 30-PMT floor.
        # A strong residual across multiple modules invalidates the
        # two-section fit even when its additional arc is concentrated.
        if error < 0.01 and gain > MARGIN and report["effective_gain_modules"] >= 3:
            return True, reports
    return False, reports
