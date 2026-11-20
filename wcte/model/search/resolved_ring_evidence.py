"""Check that an extra ring contributes independently timed photoelectrons."""

import math

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment

from gated_photon_time import gaussian, series, wavefront as time_density
from joint_prompt_fit import wavefront


@njit(cache=True)
def first_pulse_support(
    times,
    gate,
    expected,
    arrival,
    sigma,
    extra,
    offsets,
    coefficients,
    truncate_photons=False,
    excluded_mask=0,
    halo_mask=0,
    variance=None,
):
    """P(at least one PE from a source | observed charge and first time)."""
    result = np.zeros(len(times))
    for i in range(len(times)):
        width = gate[1] - gate[0]
        first = expected[i, 0] / width
        remaining = first * (gate[1] - times[i])
        other_first, other_remaining = first, remaining
        for j in range(arrival.shape[1]):
            center, width = arrival[i, j], sigma[i]
            if variance is not None:
                width = math.sqrt(width**2 + variance[i, j])
            if halo_mask & (1 << (j + 1)):
                center += 1.5
                width = math.sqrt(width * width + 9.0)
                density, tail, _ = gaussian(times[i] - center, width)
            else:
                density, tail, _ = time_density(times[i] - center, width)
            if truncate_photons:
                # The charge amplitudes describe photons in the measured
                # window. Condition both the first and the remaining PEs
                # on that same window, separately for each source.
                if halo_mask & (1 << (j + 1)):
                    _, low, _ = gaussian(gate[0] - center, width)
                    _, high, _ = gaussian(gate[1] - center, width)
                else:
                    _, low, _ = time_density(gate[0] - center, width)
                    _, high, _ = time_density(gate[1] - center, width)
                accepted = max(low - high, 1e-12)
                density /= accepted
                tail = min(1.0, max(0.0, tail - high) / accepted)
            early = expected[i, j + 1] * density
            later = expected[i, j + 1] * tail
            first += early
            remaining += later
            if j != extra and not (excluded_mask & (1 << j)):
                other_first += early
                other_remaining += later
        begin, end = offsets[i], offsets[i + 1]
        normal, _ = series(remaining, coefficients, begin, end, 1)
        without, _ = series(other_remaining, coefficients, begin, end, 1)
        # The n candidates for the earliest PE supply the factor n. For
        # the other n-1 PEs use the probability of arriving after it.
        # The measured gate cancels in this conditional probability.
        log_absent = math.log(max(other_first, 1e-300)) - math.log(max(first, 1e-300))
        result[i] = -math.expm1(min(0.0, log_absent + without - normal))
    return result


def resolve(
    event,
    model,
    previous,
    trial,
    *,
    minimum=30.0,
    separated_axis=False,
    novel_for_all=False,
    measured_gate=False,
):
    """Match the extra light against the winning fit and test its PE support."""
    from residual_joint_optics import prompt

    gate = event.timing_selection["prompt_window_ns"]
    if gate is None:
        return None
    q = np.minimum(event.charges_pe, 2.0)
    shares = trial["fractions"][:, 1:]
    old = previous[:, 1:]
    common = np.minimum(old[:, :, None], shares[:, None, :])
    common = np.sum(q[:, None, None] * common, axis=0)
    _, matching = linear_sum_assignment(-common)
    retained = common[np.arange(len(matching)), matching] / np.maximum(q @ old, 1e-20)
    (extra,) = set(range(shares.shape[1])) - set(matching)
    path = trial["paths"][extra]
    if min(retained) < 0.55 or np.exp(path[8]) > np.deg2rad(4.0):
        return None
    if separated_axis:
        # A nearby pair must resolve the 15-degree lineage cut beyond the
        # combined measured angular widths. Use displacement transverse
        # to the path so a genuine detached source can share its axis.
        for prior in trial["paths"][matching]:
            displacement = path[:3] - prior[:3]
            across = displacement - (displacement @ prior[3:6]) * prior[3:6]
            angle = np.arccos(np.clip(path[3:6] @ prior[3:6], -1.0, 1.0))
            width = np.hypot(np.exp(path[8]), np.exp(prior[8]))
            if np.linalg.norm(across) < 0.17 and angle < np.deg2rad(15.0) + 2.0 * width:
                return None
    flight = np.column_stack([wavefront(event.pmt_positions_mm, p)[0] for p in trial["paths"]])
    _, clocks = prompt(event, flight, trial["fractions"])
    center = np.median(event.times_ns)
    t = event.times_ns - center
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    rates = (trial["photons"][:, None] * trial["shapes"][:, model.ix]).T
    probabilities = first_pulse_support(
        t,
        np.asarray(gate) - center,
        rates,
        flight + clocks,
        sigma,
        extra,
        model.obs.offsets,
        model.obs.coefficients,
        bool(measured_gate),
    )
    support = float(probabilities.sum())
    # In a multi-ring event the putative new ring can borrow the late
    # light of an already counted particle. A supported additional core
    # should introduce a measured footprint beyond that incumbent ring.
    competing = float(common[:, extra].max())
    added_charge = float(q @ shares[:, extra])
    novel_fraction = 1.0 - competing / max(added_charge, 1e-20)
    if support < minimum or ((separated_axis or novel_for_all or old.shape[1] > 1) and novel_fraction < 0.50):
        return None
    result = dict(
        extra_component=extra,
        conditional_pmt_support=support,
        incumbent_retention=retained.tolist(),
        independent_charge_fraction=novel_fraction,
    )
    if measured_gate:
        from module_supported_ring import lower_bound

        result["module_support"] = lower_bound(event.pmt_ids, probabilities)
        if measured_gate == "module" and result["module_support"]["lower_statistical"] < minimum:
            return None
    return result
