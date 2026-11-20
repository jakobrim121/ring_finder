"""Profile a two-ring hadron history before revising charge ownership."""

import numpy as np

from beam_energy_step import comparisons
from light_competition import Competition, optical_template, physical_template
from shared_scoring import match_charge_shares


def review(captured, prior, details, particle, *, selected_history=None):
    from coupled_model import Model
    from optical_chain_search import geometries
    from wctering.likelihood.geometry import common_background

    event, old, bank = (captured[key] for key in ("event", "detail", "bank"))
    if prior["fractions"].shape[1] != 3 or "1" not in bank or old.get("chord") is not None:
        return prior, dict(applied=False, reason="Outside the two-ring energy-sharing scope")
    if not event.timing_selection.get("prompt_window_ns"):
        return prior, dict(applied=False, reason="No prompt gate")
    base = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, base, ("muon",), 4, 2)
    choice = Competition(
        event,
        receiver,
        prompt=True,
        primary_models=captured.get("primary_models"),
        module_charge=selected_history is not None,
    )
    seeds, checked = [], []
    for count in (1, 2, 3):
        groups = geometries(old, bank, count)
        if not groups:
            continue
        for j, g in enumerate(groups[:1] if count != 2 else [groups[0], groups[-1]]):
            template = optical_template(receiver, g, f"optical_{count}_{j}")
            if template is None:
                continue
            seeds.append((0.0, g, template))
            if count == 2 and (r := choice.fit(template)) is not None:
                checked.append(r)
    if not checked:
        return prior, dict(applied=False, reason="No two-ring optical control")
    selected = details.get("selected") or {}
    if selected.get("particles") and selected.get("order") and selected.get("count") == 2:
        physical = physical_template(event, receiver, selected)
        if physical is not None and (r := choice.fit(physical, fixed_ratios=True)) is not None:
            checked.append(r)
    trials, searched = comparisons(choice, particle, seeds)
    logn = np.log(len(receiver.active))

    def information(r):
        return 2.0 * (0.65 * r["charge"] + 0.35 * r["joint"]) - r["dimension"] * logn

    control = max(checked, key=information)
    previous = prior["fractions"]
    q = np.minimum(prior["charges"], 2.0)
    for r in sorted(trials, key=information, reverse=True):
        if r["count"] != 2 or any(p["gate_pmt_support"] < 30.0 for p in r["neighborhoods"]):
            continue
        gain = information(r) - information(control)
        if gain < 6.6349:
            break
        shares = r["fractions"]["prompt"]
        left, right, _ = match_charge_shares(previous, shares[:, 1:].T, q)
        first = previous[:, left + 1]
        later = shares[:, right + 1]
        continuity = (q @ np.minimum(first, later)) / np.maximum(q @ first, 1e-12)
        if len(left) != 2 or min(continuity) < 0.70:
            continue
        shares = shares[:, np.r_[0, 1 + right[np.argsort(left)]]]
        if selected_history is not None:
            selected_history.update(r)
        return dict(prior, fractions=shares), dict(
            applied=True,
            information_gain=gain,
            trials=searched,
            fraction_of_previous_charge=continuity,
            prior_source=control["label"],
            quadrature_error=r["quadrature_error"],
        )
    return prior, dict(applied=False, reason="No supported gain over the two-ring control", trials=searched)
