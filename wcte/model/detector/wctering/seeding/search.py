"""V101's ordered seed search, with explicit banks and no replaced globals.

The original, residual and matched partitions are distinct: foreground
proposals need the original partition; the final count uses matched evidence.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from wctering.seeding.config import RingFinderConfig
from wctering.seeding.io import prepare_event_arrays
from wctering.seeding.hypothesis_bank import fit_hypothesis_bank
from wctering.seeding.partition_support import _final_partition_background_assignment
from wctering.seeding.multiplicity import classify_bank
from wctering.seeding.detector_features import detector_evidence
from wctering.seeding.optical_evidence import fit_bank
from wctering.seeding import optical_seed_fit as optical
from wctering.seeding.event_cache import CachedOptimizer, fresh_event_local_cache
from wctering.seeding.continuation import (
    guarded_optimizer,
    supplement_bank,
    final_incumbents,
)
from wctering.seeding.matched import matched_bank, final_pool


def without_partition(original):
    result = copy.deepcopy(original)
    for name in ("analytic_partition", "legacy_partition_background_assignment"):
        result.pop(name, None)
    return result


def seeded_event(raw, event_id):
    config = RingFinderConfig(mode="high_purity_one_ring")
    event = prepare_event_arrays(
        event_id,
        raw["raw_pmt_ids"],
        raw["charges"],
        np.asarray(raw["times"], float),
        config=config,
        mapped_pmt_ids=raw.get("pmt_id_space") == "wcte",
        prompt_window_ns=raw.get("prompt_window_ns"),
    )
    if (decimals := raw.get("clock_decimals")) is not None and len(event.times_ns):
        # Apply readout clock precision after the prompt cut, in a relative frame.
        origin = float(np.median(event.times_ns))
        times = np.round(event.times_ns - origin, decimals)
        times[times == 0.0] = 0.0
        timing = dict(event.timing_selection)
        peak = timing["prompt_peak_ns"]
        if peak is None:
            timing["prompt_window_ns"] = [bound - origin for bound in timing["prompt_window_ns"]]
        else:
            peak = float(np.round(peak - origin, decimals))
            timing["prompt_peak_ns"] = peak
            timing["prompt_window_ns"] = [peak + timing[side] for side in (
                "lower_offset_from_peak_ns", "upper_offset_from_peak_ns",
            )]
        event = replace(event, times_ns=times, timing_selection=timing)
    times = event.times_ns
    centered = replace(event, times_ns=times - (float(np.median(times)) if len(times) else 0.0))
    inputs = (
        centered.pmt_ids,
        centered.pmt_positions_mm,
        centered.charges_pe,
        centered.times_ns,
    )
    offset = float(np.median(centered.times_ns)) if len(times) else 0.0
    counter_inputs = (*inputs[:3], centered.times_ns - offset)

    with fresh_event_local_cache(), threadpool_limits(limits=1):
        original = fit_hypothesis_bank(*counter_inputs, config, event_id)
        cached = CachedOptimizer(optical.optical_fit)
        # The geometric bank is independent of the counter's interim answers.
        # Construct the three-family evidence once on the final timing inputs.
        # Only the last, matched bank supplies the count used by reconstruction.
        base, base_summaries = fit_bank(*inputs, original, optimizer=cached)
        features = detector_evidence(*inputs, original)

        def classify(bank, summaries):
            return classify_bank(
                without_partition(original), *inputs, config, bank, summaries, detector_features=features
            )

        def attach(diagnostic, candidates, optimizer):
            return optical.apply_analytic_partition(
                diagnostic, *inputs, config, _initial_candidates=candidates, _optimizer=optimizer
            )

        def partition(count, candidates, optimizer):
            diagnostic = without_partition(original)
            diagnostic["predicted_ring_class"] = count
            diagnostic["final_partition_background_assignment"] = _final_partition_background_assignment(
                diagnostic, count
            )
            return attach(diagnostic, candidates, optimizer)

        guarded = guarded_optimizer(cached)
        search = SimpleNamespace(
            optical_fit=guarded,
            axis_angles=optical.axis_angles,
            _residual_proposals=optical._residual_proposals,
        )
        residual, residual_summaries = supplement_bank(search, inputs, base, base_summaries)
        matched_base, matched_summary = fit_bank(*inputs, original, optimizer=guarded)
        bank, summaries = matched_bank(search, inputs, matched_base, matched_summary, residual)
        diagnostic, candidates = classify(bank, summaries)
        count = diagnostic["predicted_ring_class"]

        # The counter depends on fitted banks, not these final partitions.
        # Only its chosen count is consumed by foreground construction and the
        # final incumbent. Build that count's original/residual partitions now.
        # Each retained fit still uses the same inputs and continuation policy.
        original_partition, frozen_partition = {}, {}
        if count in base:
            original_partition = partition(count, base_summaries[count]["_candidates"], cached)
            own = original_partition.get("analytic_partition", {})
            extra = final_incumbents(search, inputs, own)
            frozen_partition = partition(
                count, list(residual_summaries[count]["_candidates"]) + extra, guarded
            )
        extra = final_pool(search, inputs, frozen_partition)
        diagnostic = attach(diagnostic, list(candidates or []) + extra, guarded)
        diagnostic["optical_optimizer_cache"] = dict(hits=cached.hits, misses=cached.misses)
        diagnostic["_foreground_reference"] = original_partition
        diagnostic["_foreground_inputs"] = inputs
    return event, diagnostic, bank, config
