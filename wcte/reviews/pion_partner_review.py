"""Refit a measured two-track proposal with the supplied incident pion mass."""

import numpy as np

from aperture_primary_only import measured_entry_overlap
from finite_chains import factory, refine
from light_competition import Competition
from light_model_bank import measured_bank
from proton_vertex_residual import rank


def physical_candidates(original):
    trials = [r for r in original.get("candidates", ()) if r.get("supported")]
    high = [r for r in trials if r["count"] == 2 and len(r.get("particles", ())) == 2 and r.get("parameters")]
    lower = [r["gain"] for r in trials if r["count"] == 1 and r.get("gain") is not None]
    floor = max(lower, default=0.0)
    return sorted([r for r in high if (r.get("gain") or 0.0) - floor >= 20.0], key=lambda r: -r["gain"])


def eligible(prediction, original, particle):
    if particle != "pion" or prediction["fractions"].shape[1] != 2:
        return False
    family = (original.get("selected") or {}).get("particles") or []
    if not family or family[0] == "pion":
        return False
    prior = original.get("initial_counter_scores") or ()
    if isinstance(prior, dict):
        p0 = prior.get("0", prior.get(0, 0.0))
        p1 = prior.get("1", prior.get(1, 0.0))
    else:
        if len(prior) != 4:
            return False
        p0, p1 = prior[:2]
    return 1.0 - p0 - p1 >= 0.12 and bool(physical_candidates(original))


def seed(competition, record):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from material import material
    from matched_turn_split import at_aperture
    from value_only_receiver import seed_log_likelihood

    medium = material("pion")
    particle = record["particles"][0]
    former = material(particle)
    x = np.asarray(record["parameters"], float).copy()
    shape = x.copy()
    length = x[9] * x[10]
    old_energy = float(former["energy_at"](1000.0 * x[9]))
    energy = old_energy * medium["mass"] / former["mass"]
    energy = np.clip(energy, medium["energy"][0], medium["energy"][-1])
    distance = max(length + 0.015, 0.001 * float(medium["range_at"](energy)))
    shape[9:11] = distance, length / distance
    shape = at_aperture(shape)
    if shape is None:
        return None
    particles = ("pion", record["particles"][1])
    prototype = factory(competition.event, competition.model.bg, particles, 4)
    shape = canonical(prototype, shape)
    if shape is not None:
        shape = repair_energy(prototype, shape)
    if shape is None or not feasible(prototype, shape) or prototype.hardware_margin(shape) <= 1.0e-6:
        return None
    return seed_log_likelihood(prototype, shape), particles, shape, 0


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from wctering.likelihood.geometry import common_background

    report = dict(eligible=eligible(prediction, original, particle), applied=False)
    if not report["eligible"]:
        return prediction, report
    event, detail, bank = (captured[k] for k in ("event", "detail", "bank"))
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
    competition = Competition(event, receiver, prompt=True, module_charge=True)
    book, _, _, _ = measured_bank(competition, detail, bank, original, 1, deep=True)
    controls = [r for r in book if r["count"] == 1]
    if not controls:
        return prediction, report
    choices = [t for r in physical_candidates(original)[:2] if (t := seed(competition, r)) is not None]
    logn = np.log(len(receiver.active))
    maximum = np.max([rank(r, logn) for r in controls], axis=0)
    q = np.minimum(prediction["charges"], 2.0)
    old = prediction["fractions"][:, 1]
    mass = q @ old
    report["alternatives"] = []
    for plan in choices:
        t = refine(competition, plan, visibility_probability=True)
        if t is None or t["count"] != 2:
            continue
        shares = t["fractions"]["prompt"]
        support = [n["gate_pmt_support"] for n in t["neighborhoods"]]
        spread = [n["gate_leave_one_module"] for n in t["neighborhoods"]]
        entry = measured_entry_overlap(prediction["pmt_ids"], prediction["charges"], shares[:, 1], particle)
        retained = float((q @ np.minimum(old, shares[:, 1:].sum(1))) / max(mass, 1.0e-12))
        gain = rank(t, logn) - maximum
        held = []
        for base in controls:
            delta = 0.65 * (t["charge_by_module"] - base["charge_by_module"])
            delta += 0.35 * (t["joint_by_module"] - base["joint_by_module"])
            held.append(
                float(
                    2.0 * (delta.sum() - delta.max(initial=0.0)) - logn * (t["dimension"] - base["dimension"])
                )
            )
        observed = dict(
            evidence=gain.tolist(),
            robust=min(held),
            entry=entry,
            retained=retained,
            support=support,
            distributed=spread,
        )
        report["alternatives"].append(observed)
        # An incumbent ring can include charge from two separate tracks. Permit
        # a larger change in foreground when both new footprints are bright
        # and the timing gain remains strong outside the favored module.
        retained_light = retained >= 0.90 or (
            retained >= 0.80 and min(spread) >= 80.0 and min(gain) >= 80.0 and min(held) >= 60.0
        )
        if (
            retained_light
            and entry >= 0.60
            and min(gain) >= 30.0
            and min(held) >= 20.0
            and min(support) >= 30.0
            and min(spread) >= 20.0
        ):
            return dict(prediction, fractions=shares), dict(report, applied=True, selected=observed)
    return prediction, report
