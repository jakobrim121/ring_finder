"""Review a retained single ring for an independent finite charged continuation."""

import numpy as np

from light_competition import Competition, optical_template, physical_template
from short_kink import experiments


def review(
    captured,
    prediction,
    detail,
    particle,
    *,
    required_visibility=None,
    triplet_search=False,
    matched_controls=False,
):
    from coupled_model import Model
    from optical_chain_search import geometries
    from wctering.likelihood.geometry import common_background

    if prediction["fractions"].shape[1] != 2 or particle not in {"pion", "proton"}:
        return prediction, dict(applied=False, reason="Outside retained single hadron scope")
    event, old, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or event.timing_selection.get("prompt_window_ns") is None:
        return prediction, dict(applied=False, reason="No optical seed or prompt gate")
    # The coherent through-water chord already receives the independent
    # high-order continuation comparison in the retained fitter.
    if old.get("chord") is not None:
        return prediction, dict(applied=False, reason="Coherent through-water comparator")
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    competition = Competition(event, model, prompt=True, primary_models=captured.get("primary_models"))
    if triplet_search:
        from terminal_daughters import trials as terminal_trials

        examined, scanned = terminal_trials(competition, particle, triplets=True, smooth_control=True)
    else:
        examined, scanned = experiments(
            competition, particle, visibility_probability=required_visibility is not None
        )

    def visible(p):
        if required_visibility is None:
            return p["gate_pmt_support"] >= 30.0
        return p["probability_of_30"] >= required_visibility

    preferred = []
    prior = prediction["fractions"][:, 1]
    weights = np.minimum(prediction["charges"], 2.0)
    for trial in examined:
        if trial["count"] != 2 or trial.get("turn_deg", 0.0) < 15.0:
            continue
        if not all(visible(p) for p in trial["neighborhoods"]):
            continue
        if triplet_search:
            shares = trial["fractions"]["prompt"][:, 1:]
            continuity = np.max(weights @ np.minimum(prior[:, None], shares)) / max(weights @ prior, 1e-12)
            if continuity < 0.66:
                continue
        preferred.append(trial)
    if not preferred:
        return prediction, dict(
            applied=False, screened_seeds=scanned, reason="No contained and separately supported pair"
        )

    controls = [r for r in examined if r["count"] == 1]
    modes = geometries(old, bank, 1)
    if len(modes) > 2:
        modes = [modes[0], modes[-1]]
    for i, mode in enumerate(modes):
        alternative = optical_template(model, mode, f"optical_one_{i}")
        if alternative is not None and (checked := competition.fit(alternative)) is not None:
            controls.append(checked)
    chosen = detail.get("selected") or {}
    if chosen.get("count") == 1 and chosen.get("particles") and chosen.get("order"):
        control = physical_template(event, model, chosen)
        if control is not None and (checked := competition.fit(control, fixed_ratios=True)) is not None:
            controls.append(checked)
    logn = np.log(len(model.active))

    def information(r):
        return 2.0 * r["charge"] - r["dimension"] * logn

    incumbent = max(controls, key=information)
    result = max(preferred, key=information)
    margin = information(result) - information(incumbent)
    if triplet_search and matched_controls and margin > 6.6349:
        from triplet_controls import search

        controls.extend(search(competition, old, bank, particle, result))
        incumbent = max(controls, key=information)
        margin = information(result) - information(incumbent)
    supported = margin > 6.6349
    record = dict(
        applied=bool(supported),
        screened_seeds=scanned,
        information_gain=float(margin),
        competitor=incumbent["label"],
        count=2 if supported else 1,
        turn_deg=result["turn_deg"],
        neighborhoods=result["neighborhoods"],
        quadrature_error=result["quadrature_error"],
    )
    if supported:
        return dict(prediction, fractions=result["fractions"]["prompt"]), record
    return prediction, record
