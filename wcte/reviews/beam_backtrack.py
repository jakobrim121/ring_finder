"""Find connected fits by tracing measured downstream axes to the beam."""

import time

import numpy as np

from beam_continuation import Reference, reference_model, refine_continuation, score_candidate


def replay_reference(captured, detail):
    reference = reference_model(captured, detail)
    if reference is not None:
        return reference
    from timed_two_track import restore
    from wctering.likelihood.geometry import common_background

    event, bank = captured["event"], captured["bank"]
    pick = detail["selected"]
    particles = pick.get("particles", ())
    coordinates = pick.get("parameters", ())
    if not particles or len(coordinates) != 5 + 9 * len(particles) or not pick.get("order"):
        return None
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    model, state, fractions = restore(event, background, pick)
    if fractions.shape[1] != 2 or state["ll"] != state["log_joint"]:
        return None
    light = model.primary_cache[1]
    response, diffuse = np.exp(state["parameters"][-2:])
    means = response * light + diffuse * background
    basis = np.stack((background, light / max(light.sum(), 1e-300)))
    geom = dict(vertices=[state["parameters"][:3] * 1000])
    return Reference(background, basis, means, state["ll"], model.dimension, geom, "charged")


def measured_axes(bank, cached):
    """Collect detector seeds before fitting any new track."""
    seen = []
    for geometry in [*(bank[k] for k in ("2", "1", "3") if k in bank), *cached]:
        for pos, axis, angle in zip(geometry["vertices"], geometry["axes"], geometry["openings_deg"]):
            point, direction = np.asarray(pos) / 1000, np.asarray(axis, dtype=float)
            direction = direction / np.linalg.norm(direction)
            if not any(direction @ a > 0.998 and np.linalg.norm(point - p) < 0.22 for p, a, _ in seen):
                seen.append((point, direction, angle))
    return seen


def starts(event, particle, captured, reference, *, range_diversity=False):
    """Bound the seed search and reuse identical optical integrals."""
    from balanced_search import feasible
    from chain_soft_search import canonical
    from charge_starts import allowed_chord
    from coupled_model import Model
    from optical_chain_search import physical_start
    from staged_physics.primary import Aperture
    from value_only_receiver import seed_log_likelihood
    from wctering.seeding.geometry import wcte_contains_vertex

    port = Aperture()
    entry = np.array([0.0, 0.0, port.z])
    nominal = np.array([0.0, 0.0, 1.0])
    event_bank = captured["bank"]
    old = captured["detail"]
    cached = [
        r["geometry"]
        for r in sorted(old.get("fits", []), key=lambda r: r.get("score", -np.inf), reverse=True)
        if r.get("qualified") and r.get("geometry")
    ][:2]
    if old.get("chord"):
        chord = old["chord"]["chord"]
        cached.append(
            dict(
                vertices=[1000.0 * np.asarray(chord["anchor_m"])],
                axes=[chord["axis"]],
                openings_deg=[chord["opening_deg"]],
            )
        )
    picks = measured_axes(event_bank, cached)
    models = {
        tail: Model(event.pmt_ids, event.charges_pe, reference.background, (particle, tail), 4, 4)
        for tail in (particle,)
    }
    order = []
    tried = 0
    for point, outgoing, opening in picks[:12]:
        solution, *_ = np.linalg.lstsq(np.column_stack((nominal, -outgoing)), point - entry, rcond=None)
        # An unconstrained optical point usually lies somewhere along the
        # downstream ray. Reserve depths even when near-parallel rays make
        # the inferred intersection ill-conditioned.
        near = np.clip(solution[0], 0.18, 2.4)
        depths = np.unique(
            np.r_[near, np.clip(near + np.array([-0.25, 0.25]), 0.18, 2.4), 0.35, 0.75, 1.2, 1.75, 2.25]
        )
        for depth in depths:
            middle = entry + depth * nominal
            if not wcte_contains_vertex(middle * 1000):
                continue
            water = allowed_chord(middle, outgoing)
            if water is None or water[0] > 0.004 or water[1] < 0.10:
                continue
            _, end = water
            normalized_charge = event.charges_pe / max(np.sum(event.charges_pe), 1.0)
            delta = (event.pmt_positions_mm - middle * 1000) / 1000.0
            along = delta @ outgoing
            radius = np.linalg.norm(delta - along[:, None] * outgoing, axis=1)
            theta = np.deg2rad(np.clip(opening, 12.0, 41.5))
            emitting = along - radius / np.tan(theta)
            visible = (emitting > 0) & (emitting < end)
            if not np.any(visible):
                continue
            from charge_starts import weighted_quantile

            length = float(weighted_quantile(emitting[visible], normalized_charge[visible], [0.85])[0])
            length = np.clip(length, 0.10, end)
            for tail, model in models.items():
                normalization = np.log([model.response, max(0.08 * model.total, 1.0)])
                p = (particle, tail)
                tail_openings = (
                    (opening, 39.5 if particle == "pion" else 32.0) if range_diversity else (opening,)
                )
                for outgoing_length in {length, end}:
                    for tail_opening in tail_openings:
                        tried += 1
                        opens = [max(opening, 39.0 if particle == "pion" else 31.0), tail_opening]
                        initial = physical_start(
                            entry, [nominal, outgoing], [depth, outgoing_length], opens, p, 1.0, normalization
                        )
                        if initial is None:
                            continue
                        initial = canonical(model, initial)
                        if (
                            initial is None
                            or not feasible(model, initial)
                            or model.hardware_margin(initial) <= 1e-5
                        ):
                            continue
                        score = seed_log_likelihood(model, initial)
                        order.append((score, initial, depth, tail_opening))
    return sorted(order, key=lambda row: row[0], reverse=True), tried


def experiment(captured, prediction, detail, particle, *, fine_start=False):
    """Compare the reverse starts and the smooth one-track explanation."""
    from chain_comparison import MARGIN
    from coupled_model import Model
    from smooth_beam_core import fit as fit_smooth
    from value_seed_beam_candidate import fit as fit_beam

    if particle not in {"pion", "proton"} or prediction["fractions"].shape[1] != 2:
        return dict(reason="Outside one-ring hadron scope"), {}
    start = time.process_time()
    event = captured["event"]
    reference = replay_reference(captured, detail)
    if reference is None:
        return dict(reason="Unsupported reference"), {}
    previous = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        ranked, attempted = starts(event, particle, captured, reference, range_diversity=fine_start)
        seed_seconds = time.process_time() - start
        rows = []
        seed_audit = []
        seeds = ranked[:1]
        if seeds:
            first = seeds[0]
            alternate = next(
                (
                    r
                    for r in ranked[1:]
                    if abs(r[2] - first[2]) > 0.30 or r[1][12:15] @ first[1][12:15] < 0.95
                ),
                None,
            )
            if alternate is not None:
                seeds.append(alternate)
        for value, point, depth, opening in seeds:
            orders = (32, 64) if fine_start else (8, 16, 32)
            trial, history = refine_continuation(event, reference.background, particle, point, orders=orders)
            seed_audit.append(dict(seed_ll=value, depth_m=depth, opening_deg=opening, history=history))
            if trial is not None:
                rows.append(score_candidate(event, trial, reference, "backtracked_continuation"))
        same = max([0.0] + [r["gain"] for r, _ in rows if r["eligible"] and r["count"] == 1])
        eligible = [
            (r, f) for r, f in rows if r["count"] == 2 and r["eligible"] and r["gain"] > same + MARGIN
        ]
        if eligible:
            primary = fit_beam(event, reference.background, particle)
            if primary[0] is not None:
                rows.append(score_candidate(event, primary, reference, "finite_incoming"))
                smooth = fit_smooth(event, reference.background, primary[3])
                if smooth[0] is not None:
                    rows.append(score_candidate(event, smooth, reference, "smooth_incoming"))
            same = max([0.0] + [r["gain"] for r, _ in rows if r["eligible"] and r["count"] == 1])
        eligible = [(r, f) for r, f in eligible if r["gain"] > same + MARGIN]
        winner = max(eligible, key=lambda pair: pair[0]["gain"]) if eligible else None
        candidates = [r for r, _ in rows]
    finally:
        Model.delta_scale = previous
    return dict(
        reference=reference.kind,
        reference_ll=reference.ll,
        seed_cpu_seconds=seed_seconds,
        feasible_starts=len(ranked),
        attempted_starts=attempted,
        checks=seed_audit,
        candidates=candidates,
        same_count_gain=same,
        selected={"backtrack": winner[0]["family"]} if winner else {},
        cpu_seconds=time.process_time() - start,
    ), {"backtrack": winner[1]} if winner else {}
