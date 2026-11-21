"""Compare a retained hadron count with a physically connected beam history."""

import numpy as np

from calibrated_count_review import supported_shares
from light_competition import Competition
from light_model_bank import measured_bank
from merged_charge_support import charge_preservation, physical_merger
from shared_scoring import match_charge_shares
from wcte.counting import track_counting


def histories(competition, particle, optics, *, starts_per_family=2):
    from beam_endpoint_grid import compare as aperture_histories
    from short_kink import experiments as local_histories

    endpoints, seeds = aperture_histories(
        competition,
        particle,
        optics,
        receiver_screen=True,
        preserve_species=True,
        lineage_union=True,
        starts_per_family=starts_per_family,
    )
    local, local_seeds = local_histories(competition, particle)
    return endpoints, local, dict(aperture_multiplicity_seeds=seeds, local_kink_seeds=local_seeds)


def lower_count_evidence(original, count):
    posterior = original.get("initial_counter_scores")
    if posterior is None:
        return 0.0
    if isinstance(posterior, dict):
        return float(posterior.get(str(count - 1), posterior.get(count - 1, 0.0)))
    return float(posterior[count - 1]) if len(posterior) >= count else 0.0


def conservation(prediction, lower):
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"]
    i, j, _ = match_charge_shares(lower, old[:, 1:].T, q)
    kept = q @ np.minimum(lower[:, i + 1], old[:, j + 1])
    total = q @ lower[:, 1:]
    if len(i) != len(total):
        return 0.0
    return float(np.min(kept / np.maximum(total[i], 1e-12), initial=1.0))


def separated_faint_section(trial):
    """Keep a plausible separate section that the lower-count model retains."""
    rules = track_counting.get()
    if trial.get("turn_deg", 0.0) < rules.turn_offset(5.0):
        return False
    counted = {j for group in trial.get("lineage_sections", ()) for j in group}
    for j, support in enumerate(trial.get("individual_prompt_support", ())):
        if j not in counted and rules.scaled_support(15.0) <= support <= rules.scaled_support(45.0):
            return True
    return False


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=False, applied=False)
    count = prediction["fractions"].shape[1] - 1
    if count not in (2, 3) or lower_count_evidence(original, count) < (0.12 if count == 2 else 0.05):
        return (prediction, report)
    event, prepared, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return (prediction, report)
    receiver = Model(
        event.pmt_ids,
        event.charges_pe,
        common_background(np.asarray(bank["1"]["vertices"][0])),
        ("muon",),
        4,
        2,
    )
    choice = Competition(
        event, receiver, prompt=True, after_first_jitter=True, primary_models=captured.get("primary_models")
    )
    prior, optics, _, notes = measured_bank(
        choice, prepared, bank, original, count, deep=True, compare_count=count
    )
    penalty = np.log(len(receiver.active))

    def evidence(result):
        return 2.0 * (0.65 * result["charge"] + 0.35 * result["joint"]) - penalty * result["dimension"]

    def supported(result):
        return min((n["gate_pmt_support"] for n in result["neighborhoods"]), default=0.0) >= track_counting.get().min_track_pmts

    endpoints, local, search = histories(choice, particle, optics)
    report.update(search, eligible=True, bank_messages=notes)
    options = endpoints + local
    same = [r for r in prior + options if r["count"] == count]
    lower = [r for r in options if r["count"] == count - 1 and supported(r)]
    if not same or not lower:
        return (prediction, report)
    before, after = (max(same, key=evidence), max(lower, key=evidence))
    delta = evidence(after) - evidence(before)
    shares = after["fractions"]["prompt"]
    covered = conservation(prediction, shares)
    faint = separated_faint_section(after)
    report.update(
        margin=float(delta),
        confidence=lower_count_evidence(original, count),
        previous_count=count,
        proposed_count=count - 1,
        label=after["label"],
        retained_lower_core=covered,
        separated_faint_section=faint,
    )
    for name in ("turn_deg", "individual_prompt_support", "lineage_sections"):
        if name in after:
            report[name] = after[name]
    if delta < 6.6349:
        return (prediction, report)
    if count == 3 and delta < 20.0:
        return (prediction, dict(report, protected_count_margin=True))
    if covered < 0.66:
        if (
            particle != "proton"
            or faint
            or delta < (20.0 if count == 2 else 60.0)
            or (not physical_merger(after, count))
        ):
            return (prediction, report)
        candidates = charge_preservation(prediction, shares)
        match = next((row for row in candidates if row[0] >= 0.82 and min(row[1:]) >= 0.8), None)
        if match is None:
            return (prediction, report)
        report["merged_charge_retention"] = list(match)
    if not supported_shares(prediction, shares, particle):
        return (prediction, report)
    if faint:
        return (prediction, dict(report, protected_faint_section=True))
    if count == 3:
        from merged_charge_support import lost_ring_signal, preserves_distributed_light

        retained, remaining = lost_ring_signal(prediction, shares, after.get("modeled_charge_share"))
        report.update(old_core_retained=retained, old_core_unexplained_pe=remaining)
        if not preserves_distributed_light(retained, remaining):
            return (prediction, dict(report, protected_old_core=True))
    if count == 2 and particle == "proton":
        from proton_vertex_residual import compare as proton_competition, rank

        options, evaluated = proton_competition(choice, optics, constrained=False)
        proof = rank(after, penalty)
        verified = []
        alternatives = []
        q = np.minimum(prediction["charges"], 2.0)
        old_signal = prediction["fractions"][:, 1:].sum(1)
        for trial in options:
            if trial["count"] != 2:
                continue
            shares2 = trial["fractions"]["prompt"]
            kept = conservation(prediction, shares2)
            matching = match_charge_shares(prediction["fractions"], shares2[:, 1:].T, q)[2]
            linked = q @ np.minimum(old_signal, shares2[:, 1:].sum(1)) / max(q @ old_signal, 1e-10)
            audit = dict(
                label=trial["label"],
                margin=(rank(trial, penalty) - proof).tolist(),
                gate=[r["gate_pmt_support"] for r in trial["neighborhoods"]],
                distributed=[r["gate_leave_one_module"] for r in trial["neighborhoods"]],
                matched_dice=matching.tolist(),
                retained=kept,
                foreground_retained=float(linked),
                incoming_preserved=bool(supported_shares(prediction, shares2, particle)),
            )
            verified.append(audit)
            if (
                audit["incoming_preserved"]
                and linked >= 0.85
                and (min(audit["gate"]) >= 45.0)
                and (min(audit["distributed"]) >= 35.0)
                and (min(audit["margin"]) >= 40.0)
            ):
                alternatives.append((audit["margin"][1], shares2, audit))
        report.update(proton_pair_competitors=verified, evaluated_proton_pair_starts=evaluated)
        if alternatives:
            _, shares, supported = max(alternatives, key=lambda item: item[0])
            return (
                dict(prediction, fractions=shares),
                dict(report, applied=True, count=2, retained_proton_pair=supported),
            )
        from proton_turn_boundary import compare as review_boundary

        pair, boundary = review_boundary(choice, prediction, after)
        report["profiled_proton_turn"] = boundary
        if boundary["applied"]:
            return (dict(prediction, fractions=pair), dict(report, applied=True, count=2))
        from proton_successive_vertices import compare as profile_cascade

        pair, cascade = profile_cascade(choice, prediction, after, optics, prefer_on_tie=True)
        if cascade["eligible"]:
            report["profiled_successive_vertices"] = cascade
        if cascade["applied"]:
            return (dict(prediction, fractions=pair), dict(report, applied=True, count=2))
    return (dict(prediction, fractions=shares), dict(report, applied=True))
