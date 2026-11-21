"""Try the known beam track when an otherwise populated event has no ring."""

import time

import numpy as np

from beam_continuation import Reference, score_candidate
from beam_prefix import fit_prefix
from wcte.counting import track_counting


def experiment(captured, prediction, detail, particle):
    from chain_comparison import MARGIN, counted
    from readout import profile
    from ring_supported import evidence
    from wctering.likelihood.geometry import active_geometry, common_background

    event = captured["event"]
    if (
        particle not in ("pion", "proton")
        or prediction["fractions"].shape[1] != 1
        or len(np.unique(event.pmt_ids)) < track_counting.get().min_unique_pmts
    ):
        return dict(reason="Outside zero-ring hadron scope"), {}
    start = time.process_time()
    bank = captured["bank"]
    position = np.asarray(bank["1"]["vertices"][0]) if "1" in bank else np.zeros(3)
    bg = common_background(position)
    active, _, _ = active_geometry()
    ix = np.searchsorted(active, event.pmt_ids)
    _, null = profile(bg[None, :], ix, event.charges_pe)
    if not null["qualified"]:
        return dict(reason="Diffuse reference did not qualify"), {}
    means = null["amplitudes"][0] * bg
    reference = Reference(bg, bg[None, :], means, null["ll"], 1, {}, "diffuse")
    result, history = fit_prefix(event, particle, means)
    if result is None:
        return dict(reason="No qualified primary", history=history), {}
    model, state = result
    fractions = counted(model, state)
    trial = model, state, fractions, {}
    row, shares = score_candidate(event, trial, reference, "zero_beam_prefix")
    pulse = evidence(event, model, state, fractions) if row["count"] == 1 else []
    selected = bool(
        row["count"] == 1
        and row["eligible"]
        and row["gain"] > MARGIN
        and pulse
        and pulse[0]["prompt_pmts"] >= track_counting.get().scaled_support(18)
        and pulse[0]["prompt_pmts"] >= 0.35 * pulse[0]["hard_pmts"]
    )
    return dict(
        candidate=row,
        prompt=pulse,
        history=history,
        selected={"zero_primary": "finite_beam_prefix"} if selected else {},
        cpu_seconds=time.process_time() - start,
    ), {"zero_primary": shares} if selected else {}
