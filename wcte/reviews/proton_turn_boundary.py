"""Profile both sides of an optically resolved proton turn near the count cut."""

import numpy as np

from calibrated_count_review import supported_shares
from finite_chains import factory
from light_competition import Competition
from matched_turn_split import qualify, seed_turn
from occupancy_pulse import create
from proton_vertex_residual import rank
from shared_scoring import match_charge_shares
from short_kink import matched


def qualified_merge(record):
    return (
        tuple(record.get("particles", ())) == ("proton", "proton")
        and bool(record.get("order"))
        and bool(np.asarray(record.get("parameters", ())).size)
        and record.get("lineage_sections") == [(0, 1)]
        and 9.0 <= record.get("turn_deg", 0.0) < 15.0
        and min(record.get("individual_prompt_support", (0.0,))) >= 40.0
    )


def occupancy_fit(choice, trial, groups):
    model = factory(choice.event, choice.model.bg, trial["particles"], 2 * trial["order"])
    state = model(trial["parameters"], True)
    return matched(choice, model, state, groups, "first_pulse_turn")


def compare(competition, incumbent, merge):
    """Test a clearly supported two-section interpretation of the same chord."""
    report = dict(eligible=qualified_merge(merge), applied=False)
    if not report["eligible"]:
        return None, report
    floor = 17.0
    seed = seed_turn(np.asarray(merge["parameters"]), floor + 1.0)
    pair = qualify(competition, merge["particles"], seed, separate=True, aperture=True, turn_floor=floor)
    if pair is None:
        return None, report
    observed = [n["gate_pmt_support"] for n in pair["neighborhoods"]]
    distributed = [n["gate_leave_one_module"] for n in pair["neighborhoods"]]
    shares = pair["fractions"]["prompt"]
    q = np.minimum(incumbent["charges"], 2.0)
    old = incumbent["fractions"][:, 1:]
    i, j, dice = match_charge_shares(incumbent["fractions"], shares[:, 1:].T, q)
    retained = (q @ np.minimum(old[:, i], shares[:, j + 1])) / np.maximum(q @ old[:, i], 1.0e-12)
    report.update(
        turn_deg=pair["turn_deg"],
        support=observed,
        distributed=distributed,
        old_core_dice=dice.tolist(),
        retained=retained.tolist(),
    )
    valid = supported_shares(incumbent, shares, "proton")
    penalty = np.log(len(competition.model.active))
    delta = rank(pair, penalty) - rank(merge, penalty)
    report.update(conserved=bool(valid), evidence=delta.tolist())
    choice = Competition(competition.event, competition.model, prompt=True, after_first_jitter=False)
    choice.selection = create(choice.event, choice.model)
    lower = occupancy_fit(choice, merge, ((0, 1),))
    higher = occupancy_fit(choice, pair, ((0,), (1,)))
    if lower is None or higher is None:
        return None, report
    gain = higher["joint_by_module"] - lower["joint_by_module"]
    reduction = penalty * (higher["dimension"] - lower["dimension"])
    advantage = float(2.0 * gain.sum() - reduction)
    robust = float(advantage - 2.0 * gain.max(initial=0.0))
    report.update(occupancy_margin=advantage, occupancy_without_best_module=robust)
    if (
        not valid
        or min(observed) < 40.0
        or min(distributed) < 30.0
        or min(retained) < 0.55
        or max(retained) < 0.85
        or min(delta) < -6.0
        or robust <= 0.0
    ):
        return None, report
    return shares, dict(report, applied=True)
