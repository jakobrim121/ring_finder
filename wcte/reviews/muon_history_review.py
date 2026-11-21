"""Review ambiguous incoming muons with a common finite track and flight clock."""

import numpy as np

from light_competition import Competition
from light_model_bank import measured_bank
from wcte.counting import track_counting


def ambiguous(prediction, detail):
    count = prediction["fractions"].shape[1] - 1
    probabilities = detail.get("initial_counter_scores")
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(str(i), probabilities.get(i, 0.0)) for i in range(4)]
    if probabilities is None or len(probabilities) != 4:
        return False
    if count == 1:
        return sum(probabilities[2:]) >= 0.12
    return count == 2 and probabilities[1] >= 0.12


def retained_light(prediction, proposal):
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1:]
    new = proposal[:, 1:]
    old_mass = q @ old.sum(1)
    foreground = float(q @ np.minimum(old.sum(1), new.sum(1)) / max(old_mass, 1e-12))
    overlap = np.einsum("i,ijk->jk", q, np.minimum(old[:, :, None], new[:, None, :]))
    core = float(np.max(overlap / np.maximum((q @ old)[:, None], 1e-12), initial=0.0))
    return foreground, core


def review(captured, prediction, original):
    from beam_endpoint_grid import compare as endpoint_histories
    from coupled_model import Model
    from short_kink import experiments as kink_histories
    from wctering.likelihood.geometry import common_background

    if prediction["fractions"].shape[1] == 4:
        from muon_exit_response import review as review_exit

        result, report = review_exit(captured, prediction, original)
        if report["eligible"] and not report["applied"]:
            try:
                result, occupancy = review_exit(
                    captured, prediction, original, compare_finite_exit=True, guarded_occupancy=True
                )
            except (ArithmeticError, KeyError, ValueError) as error:
                occupancy = dict(applied=False, failure=str(error))
            if occupancy["applied"]:
                return result, dict(occupancy, long_muon_review=True, charge_time_review=report)
            report["occupancy_review"] = occupancy
        if not report["applied"]:
            from muon_local_exit_test import review as review_water

            refit, local = review_water(captured, prediction, original)
            if local["applied"]:
                return refit, dict(local, long_muon_review=True, charge_time_review=report)
            if local["eligible"]:
                report["local_hardware_water_comparison"] = local
        return result, dict(report, long_muon_review=True)
    report = dict(eligible=False, applied=False)
    if not ambiguous(prediction, original):
        return prediction, report
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    old_count = prediction["fractions"].shape[1] - 1
    report["eligible"] = True
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver, prompt=True, after_first_jitter=True)
    trials, optics, _, notes = measured_bank(choice, prepared, bank, original, old_count, deep=True)
    kink, _ = kink_histories(choice, "muon")
    endpoint, searched = endpoint_histories(
        choice, "muon", optics, receiver_screen=True, preserve_species=True, lineage_union=True
    )
    report.update(searched=searched, bank_messages=notes)
    penalty = np.log(len(receiver.active))

    def evidence(result):
        return 2.0 * (0.65 * result["charge"] + 0.35 * result["joint"]) - penalty * result["dimension"]

    rivals = [r for r in (*trials, *kink, *endpoint) if r["count"] == old_count]
    if not rivals:
        return prediction, report
    incumbent = max(rivals, key=evidence)
    alternatives = [r for r in endpoint if r["count"] == 3 - old_count]
    if not alternatives:
        return prediction, report
    candidate = max(alternatives, key=evidence)
    shares = candidate["fractions"]["prompt"]
    margin = evidence(candidate) - evidence(incumbent)
    foreground, core = retained_light(prediction, shares)
    report.update(
        margin=float(margin),
        retained_signal=foreground,
        retained_core=core,
        previous_count=old_count,
        trial_count=candidate["count"],
        trial_turn_deg=candidate["turn_deg"],
        trial_particles=candidate["particles"],
        support=[r["gate_pmt_support"] for r in candidate["neighborhoods"]],
    )
    support = min(r["gate_leave_one_module"] for r in candidate["neighborhoods"])
    if margin < 6.6349 or support < track_counting.get().min_track_pmts:
        return prediction, report
    if old_count == 1 and foreground < 0.85:
        return prediction, report
    if old_count == 2 and (foreground < 0.75 or core < 0.85):
        return prediction, report
    return dict(prediction, fractions=shares), dict(report, applied=True)
