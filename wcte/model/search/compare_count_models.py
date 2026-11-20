"""Recheck competing ring counts with the same charge and first-PE evidence."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from chain_comparison import MARGIN
from diffuse_timing import fit_mixture
from finite_residual_fit import optical_shape
from joint_prompt_fit import wavefront
from physical_pulse import components, profile
from gated_photon_time import log_density
from predictive_score import log_probabilities
from resolved_ring_evidence import first_pulse_support
from timed_optical import arrival
from timed_two_track import restore
from wavefront_support import section_mapping


@dataclass
class MeasuredFit:
    fractions: np.ndarray
    record: dict
    joint: np.ndarray
    penalty: float
    rates: np.ndarray
    times: np.ndarray
    core_masks: tuple

    @property
    def score(self):
        return 2.0 * self.joint.sum() - self.penalty


class Comparison:
    def __init__(self, event, model):
        self.event = event
        self.model = model
        center = np.median(event.times_ns)
        self.time = event.times_ns - center
        self.gate = np.asarray(event.timing_selection["prompt_window_ns"]) - center
        q = np.maximum(event.charges_pe, 0.25)
        self.sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / q) / 2.355**2)
        self.q = np.minimum(event.charges_pe, 2.0)
        self.modules = np.unique(model.active // 100, return_inverse=True)[1]
        self.log_n = np.log(len(model.active))
        self._support = {}

    def measured(self, fractions, record, intensity, rates, time, dimension, masks):
        model = self.model
        by_tube = log_probabilities(model.obs, intensity, model.ix)[0]
        by_tube[model.ix] += log_density(
            self.time, *self.gate, time, rates, self.sigma, model.obs.offsets, model.obs.coefficients
        )
        modules = np.bincount(self.modules, weights=by_tube)
        return MeasuredFit(fractions, record, modules, dimension * self.log_n, rates, time, masks)

    def optical(self, geometry, basis, amplitude, fractions, record):
        time = arrival(self.event, geometry)
        length = geometry.get("lengths_mm", [None] * (len(basis) - 1))
        dimension = sum(10 if value else 9 for value in length) + 1
        return self.fixed_optical(time, basis, amplitude, fractions, record, dimension)

    def finite(self, record, fractions, *, flight_offsets=None):
        paths = np.asarray(record["paths"])
        time = np.column_stack([wavefront(self.event.pmt_positions_mm, p)[0] for p in paths])
        basis = np.vstack([self.model.bg, *[optical_shape(self.model, p, 32)[0] for p in paths]])
        if np.any(basis.sum(1) <= 0.0):
            return None
        return self.fixed_optical(
            time,
            basis,
            record["photons"],
            fractions,
            record,
            10 * len(paths) + 1,
            flight_offsets=flight_offsets,
        )

    def fixed_optical(self, time, basis, amplitude, fractions, record, dimension, *, flight_offsets=None):
        clock_groups = None
        if flight_offsets is not None:
            time = time + flight_offsets
            clock_groups = np.zeros(time.shape[1], dtype=int)
        fit = fit_mixture(
            self.event,
            self.model,
            basis,
            np.asarray(amplitude),
            time,
            return_detail=True,
            clock_groups=clock_groups,
        )
        if fit is None:
            return None
        _, a, time = fit
        means = a @ basis
        rates = (a[:, None] * basis[:, self.model.ix]).T
        masks = tuple(1 << j for j in range(len(basis) - 1))
        return self.measured(fractions, record, means, rates, time, dimension, masks)

    def physical(self, record, *, restored=None):
        from module_optics import shapes

        model, state, fractions = restore(self.event, self.model.bg, record) if restored is None else restored
        if fractions.shape[1] != record["count"] + 1:
            return None
        if set(fractions.argmax(1)) - {0} != set(range(1, fractions.shape[1])):
            return None
        groups = section_mapping(model, state, fractions)
        if groups is None:
            return None
        rates, arrival_time = components(self.event, model, state, separate_knockon=True)
        time = profile(self.event, model, state, precomputed=(rates, arrival_time), timing_only=True)
        if time is None:
            return None
        masks = tuple(sum(1 << (2 * j) for j in group) for group in groups)
        return self.measured(
            fractions,
            record,
            shapes(model, state),
            rates,
            arrival_time + time["clock_ns"],
            model.dimension + 1,
            masks,
        )

    def supports(self, fit):
        model = self.model
        # The fitted rates and clock are fixed during count verification.
        # Some candidates compare the same measured footprint several times.
        key = id(fit)
        if key not in self._support:
            support = np.column_stack(
                [
                    first_pulse_support(
                        self.time,
                        self.gate,
                        fit.rates,
                        fit.times,
                        self.sigma,
                        -1,
                        model.obs.offsets,
                        model.obs.coefficients,
                        excluded_mask=mask,
                    )
                    for mask in fit.core_masks
                ]
            )
            support.setflags(write=False)
            self._support[key] = (fit, support)
        return self._support[key][1]

    def resolved(self, smaller, larger):
        """Find the extra independent wavefront after matching shared rings."""
        old, new = self.supports(smaller), self.supports(larger)
        overlap = np.minimum(old[:, :, None], new[:, None, :]).sum(0)
        _, match = linear_sum_assignment(-overlap)
        (extra,) = set(range(new.shape[1])) - set(match)
        # Where a PMT can receive PEs from either model, only the added
        # footprint that exceeds every old source is distinct evidence.
        novel = np.maximum(new[:, extra] - old.max(1), 0.0)
        by_module = np.bincount(self.modules[self.model.ix], weights=novel)
        support = new[:, extra]
        uncertainty = np.sqrt(support @ (1.0 - support))
        return dict(
            raw_pmts=float(support.sum()),
            lower_pmts=float(support.sum() - 1.645 * uncertainty),
            novel_pmts=float(novel.sum()),
            novel_without_best_module=float(novel.sum() - by_module.max(initial=0.0)),
            extra_ring=int(extra) + 1,
        )

    def separates(self, smaller, larger):
        """Measure a new light ridge even if it shares the old ring's PMTs."""
        old, new = self.supports(smaller), self.supports(larger)
        if old.shape[1] != 1 or new.shape[1] != 2:
            return None
        common = np.minimum(old, new).sum(0)
        main = int(np.argmax(common))
        added = 1 - main
        reference = new[:, main]
        support = new[:, added]
        # PMT occupancy alone can be indistinguishable when the old
        # single cone covers both real lobes. Require the new wavefront
        # to have its own light on tubes where the retained ridge is weak.
        unique = np.maximum(support - reference, 0.0)
        modules = np.bincount(self.modules[self.model.ix], weights=unique)
        variance = support @ (1.0 - support)
        joint = np.minimum(old[:, 0], new.sum(1)).sum()
        return dict(
            independent_pmts=float(unique.sum() - modules.max(initial=0.0)),
            lower_pmts=float(support.sum() - 1.645 * np.sqrt(variance)),
            retained_prompt=float(joint / max(old.sum(), 1e-20)),
            old_to_extra=float(common[added] / max(support.sum(), 1e-20)),
            extra_ring=added + 1,
        )


def geometry_key(record):
    return tuple(record.get("particles", ())), tuple(record.get("parameters", ()))


def use_for_comparison(row):
    return row.get("particles") and row.get("order") and row["supported"]


def fit(event, model, pred, winner, audit, geometry, shapes, reference, limit):
    """Change a count only when a complete rival has distributed evidence."""
    if event.timing_selection["prompt_window_ns"] is None:
        return (None, [])
    if len(np.unique(model.ix)) != len(model.ix):
        return (None, [])
    count = min(winner[2]["count"], 3)
    if count < 1:
        return (None, [])
    current = winner[2]
    base_count = pred["fractions"].shape[1] - 1
    from module_predictive import optical_has_weak_core

    if count != 3 or base_count != 3 or current["family"] != "released":
        return (None, [])
    if not optical_has_weak_core(model, geometry, shapes, reference["amplitudes"]):
        return (None, [])
    # Use fitted endpoints from adjacent multiplicities to test weak optical cores.
    records = [r for r in audit if use_for_comparison(r)]
    adjacent = {min(r["count"], 3) for r in records} | {min(base_count, 3)}
    if not any((abs(k - count) == 1 for k in adjacent)):
        return (None, [])
    from module_predictive import ModulePredictive

    compare = ModulePredictive(event, model, reference["amplitudes"][0])
    reference_fit = compare.optical(geometry, shapes, np.asarray(reference["amplitudes"]), winner[1], current)
    if reference_fit is None:
        return (None, [])
    candidates = [reference_fit]
    for rival_count in (count - 1, count):
        ranking = sorted(
            (r for r in records if min(r["count"], 3) == rival_count), key=lambda r: r["gain"], reverse=True
        )
        seen = {geometry_key(current)}
        for record in ranking:
            key = geometry_key(record)
            if key in seen:
                continue
            seen.add(key)
            trial = compare.physical(record)
            if trial is not None:
                candidates.append(trial)
            if len(seen) > limit:
                break
    if len(candidates) == 1:
        return (None, [])
    base = max((r for r in candidates if min(r.record["count"], 3) == count), key=lambda r: r.score)
    rival_count = count - 1
    rival = max(
        (r for r in candidates if min(r.record["count"], 3) == rival_count),
        key=lambda r: r.score,
        default=None,
    )
    if rival is None:
        return None, []
    delta = 2.0 * (rival.joint - base.joint)
    advantage = float(rival.score - base.score)
    robust_gain = float(advantage - delta.max(initial=0.0))
    report = dict(
        old_count=count,
        rival_count=rival_count,
        current_family=base.record["family"],
        rival_family=rival.record["family"],
        current_score=float(base.score),
        rival_score=float(rival.score),
        score_gain=advantage,
        gain_without_best_module=robust_gain,
    )
    if robust_gain <= MARGIN:
        return None, [report]
    if len(base.core_masks) != len(rival.core_masks) + 1:
        return None, [report]
    evidence = compare.resolved(rival, base)
    report["evidence"] = evidence
    # Keep the third core if even the best two-ring fit leaves
    # a countable first-pulse footprint under the original geometry.
    if evidence["novel_without_best_module"] >= 30.0 or evidence["lower_pmts"] >= 30.0:
        return None, [report]
    if base is not reference_fit:
        origin = compare.resolved(rival, reference_fit)
        report["selected_geometry_evidence"] = origin
        if origin["lower_pmts"] >= 30.0:
            return None, [report]
    report["accepted"] = True
    return (rival.fractions, rival.record, report), [report]
