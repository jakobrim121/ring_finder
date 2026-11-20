"""Preserve independently measured wavefronts during count changes."""

import numpy as np

from chain_sources import paths
from charge_starts import weighted_quantile
from coherent_partition import prediction_time
from source import trajectory


def prompt_weights(event, fractions, times, visible):
    """Describe the charge on each ring's measured direct wavefront.

    Optical centers and true birth points differ along a track. Estimate each
    center's time offset for ownership, then separately check common-clock
    consistency when the hypothesis gives physical interaction points.
    """
    q = np.minimum(event.charges_pe, 2.0)
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355
    residual = (event.times_ns - np.median(event.times_ns))[:, None] - times
    labels = fractions.argmax(1)
    records = []
    for j in range(times.shape[1]):
        ownership = q * fractions[:, j + 1] * visible[:, j]
        if not ownership.any():
            return None
        clock = weighted_quantile(residual[:, j], ownership, [0.5])[0]
        coherent = abs(residual[:, j] - clock) <= 3.0 * sigma
        records.append(ownership * coherent * (labels == j + 1))
    return np.column_stack(records)


def section_mapping(model, state, proposal):
    """Recover the counted columns of a serial trajectory from its cores."""
    cores = state["posterior"][1:].T
    support = [float(np.sum(1.0 - model.obs.no_source(state["means_hit"], f))) for f in cores.T]
    active = {j for j, value in enumerate(support) if value >= 30.0}
    groups = [[j] for j in sorted(active)]
    mapped = paths(state["parameters"][: model.ng])
    threshold = np.cos(np.deg2rad(15.0))
    for j, ((incoming, _), (outgoing, _)) in enumerate(zip(mapped, mapped[1:])):
        if j not in active or j + 1 not in active:
            continue
        direction = trajectory(incoming, np.array([1.0]))[1][0]
        if direction @ outgoing[3:6] < threshold:
            continue
        first = next(r for r in groups if j in r)
        second = next(r for r in groups if j + 1 in r)
        if first is not second:
            first.extend(second)
            groups.remove(second)
    if len(groups) != proposal.shape[1] - 1 or not all(
        np.allclose(cores[:, group].sum(1), proposal[:, j + 1], atol=2e-14, rtol=0)
        for j, group in enumerate(groups)
    ):
        return None
    return groups


def physical_prompt(event, fractions, model, state, *, grouped=False):
    """Measure each visible section's timed charge, even when they merge."""
    time, visible = prediction_time(event, state["parameters"], model.particles, event.charges_pe)
    if time.shape[1] == fractions.shape[1] - 1:
        return prompt_weights(event, fractions, time, visible > 1e-6)
    if not grouped:
        return None
    groups = section_mapping(model, state, fractions)
    if groups is None:
        return None
    charge = event.charges_pe
    q = np.minimum(charge, 2.0)
    sigma = np.sqrt(1.5**2 + 2.1**2 / np.maximum(charge, 0.25)) / 2.355
    residual = (event.times_ns - np.median(event.times_ns))[:, None] - time
    weights = state["posterior"][1:].T * q[:, None] * (visible > 1e-6)
    if not weights.any():
        return None
    clock = weighted_quantile(residual.ravel(), weights.ravel(), [0.5])[0]
    prompt = weights * (abs(residual - clock) <= 3.0 * sigma[:, None])
    labels = fractions.argmax(1)
    return np.column_stack([prompt[:, legs].sum(1) * (labels == j + 1) for j, legs in enumerate(groups)])
