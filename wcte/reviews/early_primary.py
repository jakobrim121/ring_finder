"""Qualify a fast beam primary on events with a decisive one-ring seed."""

import time

import numpy as np

from light_competition import Competition, optical_template
from short_kink import matched
from wcte.counting import track_counting


def probability(counter):
    values = (counter or {}).get("class_scores")
    if values is None:
        return 0.0
    if isinstance(values, dict):
        return float(values.get("1", values.get(1, 0.0)))
    return float(values[1]) if len(values) > 1 else 0.0


def review(prediction, old, bank, event, particle, *, counter_minimum=0.94):
    from coupled_model import Model
    from optical_chain_search import geometries
    from value_seed_beam_candidate import fit
    from wctering.likelihood.geometry import common_background

    if old.get("chord") is not None or prediction["fractions"].shape[1] != 2:
        return None
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return None
    if probability(old.get("counter")) < counter_minimum:
        return None
    bg = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, bg, ("muon",), 4, 2)
    choice = Competition(event, model, prompt=True)
    size = np.log(len(model.active))

    def evidence(r):
        return 2.0 * (0.65 * r["charge"] + 0.35 * r["joint"]) - size * r["dimension"]

    by_count = {}
    for rings in (1, 2, 3):
        saved = geometries(old, bank, rings)
        options = []
        if saved:
            representatives = saved if len(saved) <= 2 else [saved[0], saved[-1]]
            for i, g in enumerate(representatives):
                optical = optical_template(model, g, f"fast_primary_control_{rings}_{i}")
                if optical is not None and (profiled := choice.fit(optical, iterations=25)) is not None:
                    options.append(profiled)
        if options:
            by_count[rings] = max(options, key=evidence)
    control = by_count.get(1)
    rivals = [r for k, r in by_count.items() if k > 1]
    if control is None or not rivals:
        return None
    margin = evidence(control) - max(map(evidence, rivals))
    if margin < 30.0:
        return None
    original_scale = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        beam, state, _, record = fit(event, bg, particle, budget=56)
        if beam is None or record["count"] != 1:
            return None
        trial = matched(choice, beam, state, ((0,),), "fast_beam_primary")
    finally:
        Model.delta_scale = original_scale
    if trial is None:
        return None
    benefit = evidence(trial) - evidence(control)
    if benefit <= 6.6349 or min(k["gate_pmt_support"] for k in trial["neighborhoods"]) < track_counting.get().scaled_support(50.0):
        return None
    shares = trial["fractions"]["prompt"]
    q = np.minimum(prediction["charges"], 2.0)
    before, after = prediction["fractions"][:, 1], shares[:, 1]
    coverage = float(q @ np.minimum(before, after) / max(q @ before, 1e-12))
    if coverage < 0.86:
        return None
    return shares, dict(
        selected=dict(record, family="fast_beam_primary"),
        candidates=[],
        score_gain=benefit,
        seed_bank_margin=margin,
        initial_counter_scores=old["counter"]["class_scores"],
        retained_charge=coverage,
        truth_used=False,
        physically_qualified=True,
    )


def probe(prediction, optical, seeds, hits, *, particle, receipt):
    """Qualify an early beam primary and record the outcome for the review."""
    start = time.process_time()
    try:
        candidate = review(prediction, optical, seeds, hits, particle)
    except (ArithmeticError, KeyError, ValueError) as error:
        candidate = None
        receipt["candidate_failure"] = str(error)
    receipt.update(applied=candidate is not None, comparison_cpu_seconds=time.process_time() - start)
    return candidate
