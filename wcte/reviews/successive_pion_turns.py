"""Compare two successive turns within one aperture-constrained pion track."""

from itertools import permutations

import numpy as np

from calibrated_count_review import supported_shares
from finite_chains import factory, refine
from light_competition import Competition
from light_model_bank import measured_bank
from proton_vertex_residual import rank
from short_kink import matched
from wcte.counting import track_counting


def eligible(prediction, original, particle):
    selected = original.get("selected") or {}
    if (
        particle != "pion"
        or prediction["fractions"].shape[1] != 2
        or tuple(selected.get("particles") or ()) != ("pion",)
        or selected.get("gain") is None
    ):
        return False
    return any(
        r["count"] == 2
        and r.get("supported")
        and r.get("parameters")
        and (r.get("gain") or 0.0) - selected["gain"] >= 12.0
        for r in original.get("candidates", ())
    )


def plans(competition, source, optical):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical, split
    from source import trajectory
    from value_only_receiver import seed_log_likelihood

    previous = np.asarray(source["parameters"])
    track = previous[:12]
    direction = track[3:6]
    model = factory(competition.event, competition.model.bg, ("pion",) * 3, 4)
    axes = [np.asarray(axis, float) for _, g, _ in optical for axis in g["axes"]]
    axes = [r for i, r in enumerate(axes) if all(r @ left < 0.994 for left in axes[:i])]
    minimum, maximum = np.cos(np.radians([60.0, track_counting.get().turn_offset(4.0)]))
    plans = []
    for a in (0.19, 0.32, 0.46):
        first = split(track, a, "pion", "pion", 1.0)
        if first is None:
            continue
        leading, following = first
        start, terminal, *_ = trajectory(np.r_[track[:3], leading], np.ones(1))
        for middle_axis, last_axis in permutations(axes, 2):
            if (
                not minimum <= terminal[0] @ middle_axis <= maximum
                or not minimum <= middle_axis @ last_axis <= maximum
                or direction @ last_axis < 0.85
            ):
                continue
            intermediate = following.copy()
            intermediate[:3], intermediate[3:6] = middle_axis, 0.0
            mid = np.r_[0.001 * start[0], intermediate]
            for b in (0.21, 0.38, 0.55):
                second = split(mid, b, "pion", "pion", 1.0)
                if second is None:
                    continue
                remainder, final = second
                final[:3], final[3:6] = last_axis, 0.0
                seed = canonical(model, np.r_[track[:3], leading, remainder, final, previous[-2:]])
                if seed is not None:
                    seed = repair_energy(model, seed)
                if seed is None or not feasible(model, seed) or model.hardware_margin(seed) <= 1.0e-5:
                    continue
                value = seed_log_likelihood(model, seed)
                plans.append((value, model.particles, seed, a))
    chosen = []
    for row in sorted(plans, key=lambda r: -r[0]):
        if any(abs(row[3] - r[3]) < 0.12 and row[2][21:24] @ r[2][21:24] > 0.97 for r in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= 3:
            break
    return chosen


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=eligible(prediction, original, particle), applied=False)
    if not report["eligible"]:
        return prediction, report
    event, detail, bank = (captured[n] for n in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    receiver = Model(
        event.pmt_ids,
        event.charges_pe,
        common_background(np.asarray(bank["1"]["vertices"][0])),
        ("muon",),
        4,
        2,
    )
    choice = Competition(
        event, receiver, prompt=True, module_charge=True, primary_models=captured.get("primary_models")
    )
    trials, optics, _, _ = measured_bank(choice, detail, bank, original, 3, deep=True)
    former = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        beam, state, _, source = choice.finite_primary("pion")
        if beam is None or source["count"] != 1:
            return prediction, report
        controls = [r for r in trials if r["count"] in (1, 2)] + [
            matched(choice, beam, state, ((0,),), "primary_unbroken")
        ]
        possible = plans(choice, source, optics)
    finally:
        Model.delta_scale = former
    trials = [r for job in possible if (r := refine(choice, job, lineage_union=True)) is not None]
    controls = [r for r in (*controls, *trials) if r is not None and r["count"] in (1, 2)]
    report.update(seeds=len(possible), resolved_counts=[t["count"] for t in trials])
    if not controls:
        return prediction, report
    logn = np.log(len(receiver.active))
    maximum = np.max([rank(r, logn) for r in controls], axis=0)
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1]
    for proposal in sorted((r for r in trials if r["count"] == 3), key=lambda r: -rank(r, logn)[1]):
        shares = proposal["fractions"]["prompt"]
        delta = rank(proposal, logn) - maximum
        support = [r["gate_pmt_support"] for r in proposal["neighborhoods"]]
        spread = [r["gate_leave_one_module"] for r in proposal["neighborhoods"]]
        retained = float((q @ np.minimum(old, shares[:, 1:].sum(1))) / max(q @ old, 1.0e-12))
        held = []
        for past in controls:
            delta_t = proposal["joint_by_module"] - past["joint_by_module"]
            delta_q = proposal["charge_by_module"] - past["charge_by_module"]
            difference = 0.65 * delta_q + 0.35 * delta_t
            held.append(
                2.0 * (difference.sum() - difference.max(initial=0.0))
                - logn * (proposal["dimension"] - past["dimension"])
            )
        audit = dict(
            evidence=delta.tolist(), support=support, spread=spread, retained=retained, held=float(min(held))
        )
        report.update(audit)
        if (
            min(delta) >= 35.0
            and min(held) >= 15.0
            and min(support) >= track_counting.get().min_track_pmts
            and min(spread) >= 25.0
            and retained >= 0.88
            and supported_shares(prediction, shares, "pion")
        ):
            return dict(prediction, fractions=shares), dict(report, applied=True)
    return prediction, report
