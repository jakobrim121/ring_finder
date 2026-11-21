"""Split an incumbent light mixture using an already fitted beam history."""

import numpy as np

from light_competition import Competition
from light_model_bank import measured_bank
from proton_vertex_residual import rank
from wcte.counting import track_counting


def measured_pair_records(prediction, original, particle):
    from chain_sources import paths
    from matched_turn_split import at_aperture
    from staged_physics.primary import Aperture

    if prediction["fractions"].shape[1] != 2 or particle not in ("pion", "proton"):
        return []
    aperture = Aperture()
    candidates = [
        r for r in original.get("candidates", ()) if r.get("supported") and r.get("gain") is not None
    ]
    lone = max((r["gain"] for r in candidates if r["count"] == 1), default=0.0)
    compatible = []
    for rec in candidates:
        chain = rec.get("particles") or ()
        parameters = rec.get("parameters")
        if (
            rec["count"] != 2
            or len(chain) < 2
            or chain[0] != particle
            or not rec.get("order")
            or parameters is None
            or len(parameters) != 5 + 9 * len(chain)
            or rec["gain"] < lone + 75.0
        ):
            continue
        saved = np.asarray(parameters)
        origin, axis = saved[:6].reshape(2, 3)
        gap = origin[2] - aperture.z
        if not -0.20 <= gap <= 0.16 or axis[2] < 0.92:
            continue
        crosses = origin[:2] - gap * axis[:2] / axis[2]
        if np.linalg.norm(crosses) > aperture.radius + 0.015:
            continue
        # A fit can contain a short, unobserved upstream leg. Retain the
        # next section at the entrance instead of preserving that leg.
        sections = paths(saved[:-2])
        if (
            len(chain) == 3
            and chain[1] == particle
            and saved[9] * saved[10] < 0.025
            and sections[1][0][2] < aperture.z
        ):
            saved = np.r_[sections[1][0][:3], saved[12:]]
            chain = chain[1:]
        start = at_aperture(saved)
        if start is not None:
            compatible.append((rec, tuple(chain), start))
    return sorted(compatible, key=lambda r: -r[0]["gain"])[:2]


def eligible(prediction, original, particle):
    return bool(measured_pair_records(prediction, original, particle))


def comparative_evidence(trial, controls, logn):
    delta = rank(trial, logn) - np.max([rank(r, logn) for r in controls], axis=0)
    held = []
    for rival in controls:
        q = trial["charge_by_module"] - rival["charge_by_module"]
        t = trial["joint_by_module"] - rival["joint_by_module"]
        gain = 0.65 * q + 0.35 * t
        held.append(
            2.0 * (gain.sum() - gain.max(initial=0.0)) - logn * (trial["dimension"] - rival["dimension"])
        )
    return delta, float(min(held))


def review(captured, prediction, original, particle):
    from coupled_model import Model
    from finite_chains import refine
    from light_competition import Template, physical_template
    from wctering.likelihood.geometry import common_background

    records = measured_pair_records(prediction, original, particle)
    report = dict(eligible=bool(records), applied=False)
    if not records:
        return prediction, report
    event, detail, bank = (captured[k] for k in ("event", "detail", "bank"))
    if "1" not in bank or not event.timing_selection.get("prompt_window_ns"):
        return prediction, report
    model = Model(
        event.pmt_ids,
        event.charges_pe,
        common_background(np.asarray(bank["1"]["vertices"][0])),
        ("muon",),
        4,
        2,
    )
    competition = Competition(event, model, prompt=True, module_charge=True)
    models, _, _, _ = measured_bank(competition, detail, bank, original, 1, deep=True)
    controls = [
        r
        for r in models
        if r["count"] == 1
        or r["count"] == 3
        and min(n["gate_leave_one_module"] for n in r["neighborhoods"]) >= track_counting.get().min_track_pmts
    ]
    if not controls:
        return prediction, report
    logn = np.log(len(model.active))
    q = competition.q
    old = prediction["fractions"][:, 1]
    mass = max(float(q @ old), 1.0e-12)
    report["models"] = []
    for record, chain, start in records:
        trial = refine(competition, (0.0, chain, start, 0), physical_coverage=True)
        if trial is None or trial["count"] != 2:
            continue
        template = physical_template(
            event, model, dict(record, particles=chain, parameters=trial["parameters"], order=trial["order"])
        )
        if template is None or 1 not in template.core_groups[0]:
            continue
        shares = trial["fractions"]["prompt"]
        gain, held = comparative_evidence(trial, controls, logn)
        support = [n["gate_pmt_support"] for n in trial["neighborhoods"]]
        spread = [n["gate_leave_one_module"] for n in trial["neighborhoods"]]
        retained = float(q @ np.minimum(old, shares[:, 1:].sum(1)) / mass)
        portions = (q @ np.minimum(old[:, None], shares[:, 1:])) / mass
        info = dict(
            particles=list(record["particles"]),
            family=record["family"],
            gain=gain.tolist(),
            held=held,
            support=support,
            distributed=spread,
            retained=retained,
            portions=portions.tolist(),
        )
        report["models"].append(info)
        if (
            min(gain) < 75.0
            or held < 45.0
            or min(support) < track_counting.get().min_track_pmts
            or min(spread) < track_counting.get().scaled_support(40.0)
            or retained < 0.90
            or min(portions) < 0.18
        ):
            continue
        drop = []
        for group in template.core_groups:
            keep = [0, *(k for core in group for k in (core, core + 1))]
            single = Template(
                template.label + "_omitted",
                template.shapes[keep],
                template.center[:, np.asarray(keep[1:]) - 1],
                template.amplitudes[keep],
                (tuple(range(1, len(keep), 2)),),
                np.zeros(len(keep) - 1, int),
                trial["dimension"] - 7 * (len(chain) - len(group)),
                {},
            )
            drop.append(competition.fit(single, iterations=32, fixed_ratios=True))
        if any(r is None for r in drop):
            continue
        gain, held = comparative_evidence(trial, drop, logn)
        info.update(conditional_gain=gain.tolist(), conditional_held=held)
        if min(gain) >= 55.0 and held >= 40.0:
            return dict(prediction, fractions=shares), dict(report, applied=True, count=2)
    return prediction, report
