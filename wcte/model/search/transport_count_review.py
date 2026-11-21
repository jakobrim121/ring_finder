"""Test whether calibrated charged tracks explain extra optical bands."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from chain_comparison import MARGIN
from chain_sources import paths
from compare_count_models import Comparison, geometry_key
from diffuse_timing import fit_mixture
from module_optics import conditional
from module_supported_ring import lower_bound
from physical_pulse import components
from source import raw
from timed_two_track import restore
from wavefront_support import section_mapping
from wcte.counting import track_counting


@dataclass
class Response:
    """Separate direct cores, slowing tails and the remaining charge."""

    shape: np.ndarray
    time: np.ndarray
    groups: np.ndarray
    cores: tuple


def responses(event, model, record, restored):
    """Build disjoint light columns with a shared scale per charged track."""
    trial, state, fraction = restored
    mapping = section_mapping(trial, state, fraction)
    if mapping is None or len(mapping) != record["count"]:
        return None
    x = state["parameters"]
    saved = trial.primary_cache
    signal, cores = saved[1:3]
    assigned = np.sum(cores, axis=0)
    if not np.any(assigned):
        return None
    eta, dark = np.exp(x[-2:])
    measured = [dark * model.bg]
    groups = [0]
    arrival_time = []
    group_masks = [0] * len(mapping)
    # Keep the unresolved recoil light in the background. The direct and
    # slowing light share the same amplitude, clock and particle path.
    _, centers = components(event, trial, state, separate_knockon=True)
    for j, ((path, _), mass) in enumerate(zip(paths(x[: trial.ng]), trial.particles)):
        light, _ = raw(model.pos, model.normals, path, 2 * record["order"], False, mass)
        tail = 1000.0 * path[9] * path[10] * light[1]
        measured.append(eta * cores[j])
        groups.append(j + 1)
        arrival_time.append(centers[:, 2 * j])
        for group, legs in enumerate(mapping):
            if j in legs:
                group_masks[group] |= 1 << (len(groups) - 2)
        if tail[model.ix].sum() > 1e-14:
            measured.append(eta * tail)
            groups.append(j + 1)
            arrival_time.append(centers[:, 2 * j + 1])
            assigned += tail
    measured[0] = measured[0] + eta * np.maximum(signal - assigned, 0.0)
    return Response(np.asarray(measured), np.column_stack(arrival_time), np.array(groups), tuple(group_masks))


def fit_physical(event, model, record, restored, comparison, *, rate_free, response=None):
    """Profile track yields under the same time and charge law as the cones."""
    trial, state, _ = restored
    if response is None:
        response = responses(event, model, record, restored)
    if response is None:
        return None
    groups, light = response.groups, response.shape
    if not rate_free:
        groups = np.minimum(groups, 1)
    n = int(groups.max()) + 1
    total = np.zeros((n, light.shape[1]))
    np.add.at(total, groups, light)
    scale = total.sum(1)
    if np.any(scale <= 1e-8):
        return None
    # Allow the existing bounded event response to vary jointly across
    # sections, or independently when testing response calibration.
    old_response = state["parameters"][-2]
    window = np.exp(np.asarray(trial.bounds[-2]) - old_response)
    bounds = {j: tuple(window * scale[j]) for j in range(1, n)}
    shape = light / scale[groups, None]
    basis = total / scale[:, None]
    fit = fit_mixture(
        event,
        model,
        basis,
        scale,
        response.time,
        amplitude_bounds=bounds,
        clock_groups=np.zeros(len(groups) - 1, dtype=int),
        timing_components=(groups, shape),
        return_detail=True,
    )
    if fit is None:
        return None
    _, amp, time = fit
    mean = amp @ basis
    rates = (amp[groups, None] * shape[:, model.ix]).T
    signal = (
        np.column_stack(
            [
                rates[:, [i + 1 for i in range(len(groups) - 1) if mask & (1 << i)]].sum(1)
                for mask in response.cores
            ]
        )
        / mean[model.ix, None]
    )
    f = np.column_stack([1.0 - signal.sum(1), signal])
    if set(f.argmax(1)) - {0} != set(range(1, 1 + record["count"])):
        return None
    dimension = trial.dimension + len(amp) - 1
    measured = comparison.measured(f, record, mean, rates, time, dimension, response.cores)
    return measured, mean, amp / scale


def plausible_dropout(comparison, old, fit, mean, basis, amplitude):
    """Require two retained wavefronts and examine the uncovered third."""
    old_support = comparison.supports(old)
    new_support = comparison.supports(fit)
    common = np.minimum(old_support[:, :, None], new_support[:, None, :]).sum(0)
    old_index, new_index = linear_sum_assignment(-common)
    retained = common[old_index, new_index] / np.maximum(old_support.sum(0)[old_index], 1e-20)
    (rem,) = set(range(old_support.shape[1])) - set(old_index)
    leftover = np.maximum(old_support[:, rem] - new_support.max(1), 0.0)
    missing = lower_bound(comparison.event.pmt_ids, leftover)
    per_core = [lower_bound(comparison.event.pmt_ids, u) for u in new_support.T]
    adequate = min(r["lower_statistical"] for r in per_core) >= track_counting.get().min_track_pmts
    keep = np.min(retained)
    unresolved = missing["pmt_support"] < track_counting.get().min_track_pmts
    model, event = comparison.model, comparison.event
    contrast = conditional(model.active, model.ix, event.charges_pe, mean) - conditional(
        model.active, model.ix, event.charges_pe, amplitude @ basis
    )
    module_gain = contrast.sum() - contrast.max(initial=0.0)
    return adequate and keep >= 0.55 and unresolved and module_gain > 0.0, dict(
        retained=retained.tolist(), missing=missing, new_support=per_core, module_gain=float(module_gain)
    )


def fit(event, model, geometry, basis, amplitude, incumbent, bank, limit, *, rate_free=False):
    """Test saved physical paths with one calibrated timing and charge model."""
    current = incumbent[2]
    count = current["count"]
    if (
        count != 3
        or len(basis) != 4
        or event.timing_selection["prompt_window_ns"] is None
        or geometry.get("emission_model") == "full_water_line"
    ):
        return None, []
    ranking = sorted(
        (
            r
            for r in bank
            if r["count"] in (2, 3) and r.get("particles") and r.get("order") and r["supported"]
        ),
        key=lambda r: -r["gain"],
    )
    if not any(r["count"] == 2 for r in ranking):
        return None, []
    compare = Comparison(event, model)
    released = dict(family="released", count=3)
    null = compare.optical(geometry, basis, amplitude, incumbent[1], released)
    if null is None:
        return None, []
    old = [null]
    if current.get("particles") and current.get("order"):
        ranking.insert(0, current)
    if current.get("paths") and current.get("photons"):
        endpoint = compare.finite(current, incumbent[1])
        if endpoint is not None:
            old.append(endpoint)
    trials, reports = [], []
    seen = {2: set(), 3: set()}
    for r in ranking:
        k = r["count"]
        key = geometry_key(r)
        if key in seen[k] or len(seen[k]) >= limit:
            continue
        seen[k].add(key)
        restored = restore(event, model.bg, r)
        tested = fit_physical(event, model, r, restored, compare, rate_free=rate_free)
        if tested is not None:
            if k == 3:
                old.append(tested[0])
            else:
                trials.append(tested)
    for trial in trials:
        fit, mean, brightness = trial
        relative = [2.0 * (fit.joint - r.joint) for r in old]
        gain = min(fit.score - r.score - d.max(initial=0.0) for r, d in zip(old, relative))
        detail = dict(
            family="transport_count_review",
            count=2,
            gain=float(gain),
            supported=False,
            response_scale=brightness.tolist(),
            source=fit.record["family"],
            particles=fit.record["particles"],
        )
        reports.append(detail)
        if gain <= MARGIN:
            continue
        accepted, evidence = plausible_dropout(compare, null, fit, mean, basis, amplitude)
        detail.update(evidence, supported=bool(accepted))
    choices = [(r["gain"], t[0].fractions, r) for r, t in zip(reports, trials) if r["supported"]]
    return max(choices, key=lambda r: r[0], default=None), reports
