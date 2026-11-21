"""Compare both sides of the lineage turn cut using the same charged history."""

import numpy as np

from kink_boundary import turn_cosine
from lineage_support import compare as measured_lineage
from wcte.counting import track_counting
from wcte.counting import track_counting


def turn_degrees(x):
    cosine, _ = turn_cosine(x)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def model_at(competition, particles, order, separate, *, aperture=False, turn_floor=None):
    from coupled_model import Model

    rules = track_counting.get()
    # Leave 0.02 degrees on each side to keep solver tolerance off the cut.
    boundary = (turn_floor or rules.turn_offset(0.02)) if separate else rules.turn_offset(-0.02)
    threshold = np.cos(np.radians(boundary))
    sign = -1.0 if separate else 1.0

    e, m = competition.event, competition.model
    if aperture:
        from finite_chains import factory

        model = factory(e, m.bg, particles, order)
    else:
        model = Model(e.pmt_ids, e.charges_pe, m.bg, particles, 4, order)
    control = model.all_constraints

    def all_constraints(x):
        values, derivative = control(x)
        cosine, gradient = turn_cosine(x)
        value = sign * (cosine - threshold)
        return np.r_[values, value], np.vstack([derivative, sign * gradient])

    model.all_constraints = all_constraints
    return model


def seed_turn(x, degrees):
    from source import trajectory

    x = x.copy()
    incoming = trajectory(x[:12], np.ones(1))[1][0]
    tangent = x[12:15] - (x[12:15] @ incoming) * incoming
    if np.linalg.norm(tangent) < 1.0e-8:
        tangent = np.cross(incoming, np.eye(3)[np.argmin(abs(incoming))])
    theta = np.radians(degrees)
    x[12:15] = np.cos(theta) * incoming + np.sin(theta) * tangent / np.linalg.norm(tangent)
    x[15:18] -= (x[15:18] @ x[12:15]) * x[12:15]
    return x


def qualify(competition, particles, x, *, separate, aperture=False, turn_floor=None):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from track_coordinates import fit

    if separate and track_counting.get().track_merge_angle_deg == 180.0:
        return None
    nfev = 0
    x = x.copy()
    for order in (8, 16, 32):
        model = model_at(competition, particles, order, separate, aperture=aperture, turn_floor=turn_floor)
        x = canonical(model, x)
        if x is None or (x := repair_energy(model, x)) is None:
            return None
        outcome, x = fit(model, x, 30, 43, full_likelihood=True, check_initial=True)
        nfev += int(outcome.nfev)
        fine = model_at(competition, particles, 2 * order, separate, aperture=aperture, turn_floor=turn_floor)
        if not feasible(fine, x) or fine.hardware_margin(x) <= 1.0e-5:
            return None
        state = fine(x, True)
        error = abs(state["ll"] - model(x, True)["ll"])
        if error > 0.01:
            continue
        value = measured_lineage(competition, fine, state)
        if value is not None and value["count"] == (2 if separate else 1):
            value.update(
                label="matched_turn_split_" + ("separate" if separate else "merged"),
                turn_deg=turn_degrees(x),
                parameters=x,
                particles=particles,
                quadrature_error=error,
                fit_calls=nfev,
                order=order,
                aperture=aperture,
            )
            return value
        return None
    return None


def at_aperture(x):
    from staged_physics.primary import Aperture
    from source import trajectory

    port = Aperture()
    position, tangent, _, _ = trajectory(x[:12], np.ones(1))
    p = position[0] / 1000.0
    ray = x[3:6]
    if ray[2] < 0.35 or p[2] <= port.z + 0.022:
        return None
    entry = x[:3] + (port.z - x[2]) * ray / ray[2]
    entry[:2] *= min(1.0, 0.84 * port.radius / max(np.linalg.norm(entry[:2]), 1.0e-15))
    a = p - entry
    length = np.linalg.norm(a)
    axis = a / length
    q = x.copy()
    q[:3], q[3:6] = entry, axis
    q[6:9] = (tangent[0] - axis) - axis * ((tangent[0] - axis) @ axis)
    q[9] = max(q[9], length + 0.010)
    q[10] = length / q[9]
    return q


def compare(competition, record, *, aperture=False):
    from balanced_search import feasible
    from bare_residual_search import repair_energy
    from chain_soft_search import canonical
    from short_kink import matched
    from timed_two_track import restore

    found = record.get("selected")
    if not (
        found and found.get("count") == 1 and len(found.get("particles", ())) == 2 and found.get("order")
    ):
        return [], dict(reason="No fitted two-section incumbent")
    x = np.asarray(found["parameters"], float)
    angle = turn_degrees(x)
    rules = track_counting.get()
    if angle > rules.track_merge_angle_deg:
        return [], dict(reason="Incumbent's turn is already resolved", old_turn=angle)
    if rules.track_merge_angle_deg == 180.0:
        return [], dict(reason="No separate turn exceeds the requested angle", old_turn=angle)
    parts = tuple(found["particles"])
    model, state, _ = restore(competition.event, competition.model.bg, found)
    sections = matched(competition, model, state, ((0,), (1,)), "unmerged_incumbent_sections")
    if sections is None or min(r["gate_pmt_support"] for r in sections["neighborhoods"]) < rules.scaled_support(28.0):
        return [], dict(reason="Insufficient measured light in two sections", old_turn=angle)
    seed = at_aperture(x) if aperture else x
    if seed is None:
        return [], dict(reason="No leg crosses the surveyed entry plane", old_turn=angle)
    coarse = model_at(competition, parts, 4, True, aperture=aperture)
    rank = []
    for shift in (0.78, 1.0, 1.24):
        for turn in (rules.turn_offset(offset) for offset in (1.5, 8.0, 18.0)):
            q = seed_turn(seed, turn)
            q[10] = np.clip(q[10] * shift, 0.001, 0.998)
            q = canonical(coarse, q)
            if q is None or (q := repair_energy(coarse, q)) is None:
                continue
            if feasible(coarse, q) and coarse.hardware_margin(q) > 1.0e-5:
                rank.append((float(coarse(q, True)["log_joint"]), q))
    result = []
    for _, q in sorted(rank, key=lambda r: -r[0])[:2]:
        fit = qualify(competition, parts, q, separate=True, aperture=aperture)
        if fit is not None:
            old = sections["fractions"].get("prompt", sections["fractions"]["timed"])[:, 1:]
            new = fit["fractions"].get("prompt", fit["fractions"]["timed"])[:, 1:]
            weight = competition.q
            retained = (weight @ np.minimum(old, new)) / np.maximum(weight @ old, 1.0e-12)
            fit["retained_each_incumbent_section"] = retained.tolist()
            fit["retained_both_incumbent_sections"] = float(
                (weight @ np.minimum(old.sum(1), new.sum(1))) / max(weight @ old.sum(1), 1.0e-12)
            )
            result.append(fit)
    if result:
        unchanged = seed_turn(seed, min(turn_degrees(seed), rules.turn_offset(-0.25))) if aperture else seed
        rival = qualify(competition, parts, unchanged, separate=False, aperture=aperture)
        if rival is not None:
            result.append(rival)
    return result, dict(old_turn=angle, tested_starts=len(rank))
