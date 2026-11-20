"""Use timed finite optics to refine a measured charged-track split."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from chain_comparison import MARGIN
from compare_count_models import Comparison
from gated_optical_fit import fit as fit_optical
from module_optics import conditional, shapes
from module_supported_ring import lower_bound
from residual_joint_optics import path_from_geometry


def verify(compare, old, tested, mean, old_mean, *, spread_support=False):
    """Find the raw evidence for an omitted third ring."""
    model, event = compare.model, compare.event
    ref, rep = compare.supports(old), compare.supports(tested)
    if ref.shape[1] != 3 or rep.shape[1] != 2:
        return False, {}
    common = np.minimum(ref[:, :, None], rep[:, None, :]).sum(0)
    i, j = linear_sum_assignment(-common)
    (missed,) = set(range(3)) - set(i)
    overlap = common[i, j] / np.maximum(ref.sum(0)[i], 1e-20)
    stronger = rep.max(1)
    independent = lower_bound(event.pmt_ids, np.maximum(ref[:, missed] - stronger, 0.0))
    support = [lower_bound(event.pmt_ids, row) for row in rep.T]
    contrast = conditional(model.active, model.ix, event.charges_pe, mean)
    contrast -= conditional(model.active, model.ix, event.charges_pe, old_mean)
    within = contrast.sum() - contrast.max(initial=0.0)
    # The retained cores have to be independently visible. Any third
    # optical band on 30 additional PMTs keeps the larger hypothesis.
    accepted = (
        min(overlap) >= 0.62
        and independent["pmt_support"] < 30.0
        and min(r["lower_statistical"] for r in support) >= 30.0
        and within > 0.0
    )
    data = dict(
        retained=overlap.tolist(), independent=independent, core_support=support, module_gain=float(within)
    )
    if spread_support:
        data["omitted_index"] = int(missed)
    return accepted, data


def review(event, model, geometry, basis, amplitude, baseline, winner, bank_two, archive, budget):
    """Refine both counts with the same finite-path geometry and first-time model."""
    if (
        winner[2]["count"] != 3
        or len(basis) != 4
        or event.timing_selection["prompt_window_ns"] is None
        or (geometry.get("emission_model") == "full_water_line")
    ):
        return (None, [])
    from aperture_geometry_fit import starts as aperture_starts

    # Require distinct measured wavefronts at a feasible join before proposing a two-track rival.
    candidates = aperture_starts(event, model, geometry, amplitude, bank_two, limit=2)
    if not candidates:
        return (None, [])
    comparison = Comparison(event, model)
    original = comparison.optical(geometry, basis, amplitude, baseline, dict(family="released", count=3))
    if original is None:
        return (None, [])
    best, reports, old = (None, [], None)
    optical_rival = None
    predictive = None
    for _, start, name in candidates:
        from aperture_optical_track import fit as fit_track

        result = fit_track(event, start, model.bg, budget)
        if result is None:
            continue
        paths, fractions, info = result
        offsets = np.array([0.0, info["offset_ns"]])
        record = dict(
            family="reassigned_optics",
            source=name,
            count=2,
            paths=paths.tolist(),
            photons=info["photons"].tolist(),
            supported=False,
            evaluations=info["evaluations"],
        )
        record.update(flight_offsets_ns=offsets.tolist(), angle_deg=info["angle_deg"])
        if "quadrature_ll_error" in info:
            record["quadrature_ll_error"] = info["quadrature_ll_error"]
        proposal = comparison.finite(record, fractions, flight_offsets=offsets)
        if proposal is None:
            continue
        from aperture_optical_track import transports

        record["start_score_margin"] = float(proposal.score - original.score)
        if record["start_score_margin"] < -4.0 * np.sqrt(len(comparison.q)):
            reports.append(dict(record, gain=0.0))
            continue
        considered = list(
            transports(event, model, paths, info["photons"], budget, limit=3, module_weight=1.0)
        )
        record["transport_candidates"] = [check for check, _ in considered]
        certified = [
            (comparison.physical(check, restored=restored), check, restored)
            for check, restored in considered
            if restored is not None
        ]
        valid = [row for row in certified if row[0] is not None]
        if not valid:
            reports.append(dict(record, gain=0.0))
            continue
        proposal, check, physical = max(valid, key=lambda row: row[0].score)
        mean = shapes(*physical[:2])
        record["transport_check"] = check
        fractions = proposal.fractions
        record["seed_paths"], record["seed_photons"] = (record.pop("paths"), record.pop("photons"))
        record.update(check, family="reassigned_optics")
        gain = proposal.score - original.score - max(0.0, 2.0 * np.max(proposal.joint - original.joint))
        record["gain"] = float(gain)
        reports.append(record)
        if gain <= MARGIN:
            continue
        accepted, detail = verify(comparison, original, proposal, mean, amplitude @ basis)
        record.update(detail)
        if not accepted:
            continue
        if old is None:
            control = fit_optical(event, geometry, model.bg, budget, strength=1.0)
            if control is None:
                break
            _, share, info3 = control
            try:
                paths3 = [path_from_geometry(info3["geometry"], j) for j in range(3)]
            except (ArithmeticError, ValueError):
                break
            old = comparison.finite(
                dict(family="refitted_optical", count=3, paths=paths3, photons=info3["photons"]), share
            )
            if old is None:
                break
            rivals = sorted(
                (
                    r
                    for r in archive
                    if r.get("particles") and r.get("order") and (r["count"] == 3) and r.get("supported")
                ),
                key=lambda r: -r["gain"],
            )
            if winner[2].get("particles") and winner[2].get("order"):
                rivals.insert(0, winner[2])
            controls = [old]
            for r in rivals[:2]:
                checked = comparison.physical(r)
                if checked is not None:
                    controls.append(checked)
            picked = winner[2]
            if picked.get("paths") and picked.get("photons"):
                checked = comparison.finite(picked, winner[1])
                if checked is not None:
                    controls.append(checked)
        gain = min(
            (proposal.score - r.score - max(0.0, 2.0 * np.max(proposal.joint - r.joint)) for r in controls)
        )
        record["gain"] = float(gain)
        if gain <= MARGIN:
            continue
        accepted, details = verify(comparison, old, proposal, mean, amplitude @ basis, spread_support=True)
        if accepted:
            from remaining_wavefront import remains

            j = details["omitted_index"]
            record["remaining_seed"] = dict(
                path=paths3[j].tolist(), photons=float(old.record["photons"][j + 1])
            )
            blocked, residual = remains(
                comparison,
                proposal,
                mean,
                paths3[j],
                old.record["photons"][j + 1],
                budget // 2,
                separate_yields=False,
            )
            record["remaining_wavefront"] = residual
            if blocked:
                continue
            from module_predictive import ModulePredictive

            if predictive is None:
                predictive = ModulePredictive(event, model, amplitude[0])
                optical_rival = predictive.finite(old.record, old.fractions)
            charged = predictive.physical(record, restored=physical)
            if charged is None or optical_rival is None:
                continue
            delta = 2.0 * (charged.joint - optical_rival.joint)
            gap = float(charged.score - optical_rival.score - delta.max(initial=0.0))
            record["module_predictive_gain"] = gap
            if gap <= MARGIN:
                continue
            from third_charged_section import check

            clock = None
            from additional_track_scan import endpoint_cones
            from physical_pulse import profile

            timing = profile(event, *physical[:2], separate_knockon=True, timing_only=True)
            if timing is not None:
                clock = timing["clock_ns"]
            cones = list(endpoint_cones(comparison, physical, shared=clock))
            blocked, third = check(
                comparison,
                proposal,
                physical,
                paths3[j],
                old.record["photons"][j + 1],
                budget,
                scan=False,
                timed=True,
                cones=cones,
            )
            record["additional_charged_section"] = third
            if blocked:
                continue
            from return_light_review import check as reflected

            blocked, reflected_trials = reflected(comparison, proposal, physical, cones, clock=clock)
            record["return_light_trials"] = reflected_trials
            if blocked:
                continue
        record["supported"] = bool(accepted)
        if accepted and (best is None or gain > best[0]):
            best = (float(gain), fractions, record)
    return (best, reports)
