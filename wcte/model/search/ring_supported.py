"""Inspect an extra physical section for distinct prompt and charge support."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from coherent_partition import prediction_time
from charge_starts import weighted_quantile
from chain_sources import paths


def evidence(event, model, state, fractions, *, compare=None):
    """Measure residual coherence for each unmerged charged section."""
    particles = model.particles
    if fractions.shape[1] != len(particles) + 1:
        return []
    q = event.charges_pe
    predicted, illuminated = prediction_time(event, state["parameters"], particles, q)
    time = event.times_ns - np.median(event.times_ns)
    residual = time[:, None] - predicted
    weights = fractions[:, 1:] * np.minimum(q, 2.0)[:, None] * (illuminated > 1e-6)
    if not np.any(weights):
        return []
    clock = weighted_quantile(residual.ravel(), weights.ravel(), [0.5])[0]
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(q, 0.25)) / 2.355
    widths = abs(residual - clock) / sigma[:, None]
    if compare is not None:
        one_model, one_state, one_fraction = compare
        one_time, one_light = prediction_time(event, one_state["parameters"], one_model.particles, q)
        one_residual = time - one_time[:, 0]
        one_weight = np.minimum(q, 2.0) * one_fraction[:, 1] * (one_light[:, 0] > 1e-6)
        if one_weight.any():
            one_clock = weighted_quantile(one_residual, one_weight, [0.5])[0]
            one_width = abs(one_residual - one_clock) / sigma
        else:
            one_width = np.zeros(len(q))
    geometries = [p for p, _ in paths(state["parameters"][:-2])]
    metrics = []
    for j, p in enumerate(geometries):
        owned = fractions.argmax(1) == j + 1
        prompt = owned & (widths[:, j] <= 3.0) & (illuminated[:, j] > 1e-6)
        support = (1 - model.obs.no_source(state["means_hit"], fractions[:, j + 1])).sum()
        record = dict(
            expected_pmts=float(support),
            prompt_pmts=int(np.count_nonzero(prompt)),
            hard_pmts=int(np.count_nonzero(owned)),
            length_m=float(p[9] * p[10]),
        )
        if compare is not None:
            # A coherent section can overlap the first wavefront of a
            # continuous track. Count evidence needs its own direct light.
            original = np.log1p(one_width**2 / 4)
            extension = np.log1p(widths[:, j] ** 2 / 4)
            w = np.minimum(q, 2.0) * fractions[:, j + 1] * owned
            record["relative_timing_gain"] = float(w @ (original - extension))
            record["distinct_prompt_pmts"] = int(
                np.count_nonzero(prompt & ((one_width > 3.0) | (one_fraction[:, 1] < 0.15)))
            )
        metrics.append(record)
    return metrics


def preserves_prompt(event, current, proposed):
    """Check that a refitted trajectory does not erase a supported wavefront."""
    reference, reference_state, ref_fraction = current
    model, state, fractions = proposed
    q = np.minimum(event.charges_pe, 2.0)
    t = event.times_ns - np.median(event.times_ns)
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355

    def wavefront(candidate, fit):
        # The columns of the charge posterior retain every physical section
        # even when two adjacent sections contribute to one counted ring.
        pred, visible = prediction_time(event, fit["parameters"], candidate.particles, event.charges_pe)
        residual = t[:, None] - pred
        weight = fit["posterior"][1:].T * q[:, None] * (visible > 1e-6)
        if not weight.any():
            return np.zeros(len(q), bool)
        clock = weighted_quantile(residual.ravel(), weight.ravel(), [0.5])[0]
        return np.any((abs(residual - clock) < 3.0 * sigma[:, None]) & (visible > 1e-6), axis=1)

    observed = wavefront(reference, reference_state)
    predicted = wavefront(model, state)
    old, new = ref_fraction.argmax(1), fractions.argmax(1)
    old_count, new_count = ref_fraction.shape[1] - 1, fractions.shape[1] - 1
    if not new_count:
        return False, []
    support = np.bincount(old[observed], minlength=old_count + 1)[1:]
    if new_count < old_count:
        # Use one-to-one ownership for a demotion. Otherwise two distinct
        # well supported wavefronts could be claimed by the same new ring.
        overlap = np.zeros((old_count + 1, new_count + 1), int)
        common = observed & predicted
        np.add.at(overlap, (old[common], new[common]), 1)
        i, j = linear_sum_assignment(-overlap[1:, 1:])
        lost = support.copy()
        lost[i] -= overlap[i + 1, j + 1]
    else:
        lost = np.bincount(old[observed & (~predicted | (new == 0))], minlength=old_count + 1)[1:]
    lost = lost.tolist()
    return max(lost, default=0) < 24, lost
