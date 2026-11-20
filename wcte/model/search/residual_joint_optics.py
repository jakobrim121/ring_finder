"""Recover short rings by moving both the residual and the existing cones."""

from itertools import chain

import numpy as np
from scipy.optimize import linear_sum_assignment

from analog_optical import Analog, contain_finite, endpoint
from charge_starts import weighted_quantile
from chord_profile import profile
from finite_residual_fit import optical_shape
from joint_prompt_fit import wavefront
from timed_optical import arrival
from wctering.likelihood.charge import MAX_OPENING_RAD
from wctering.seeding.coherent_timing import C, N_PHASE


def path_from_geometry(geometry, j):
    """Read a cone into the existing finite light coordinate system."""
    direction = np.asarray(geometry["axes"][j], dtype=float)
    direction /= np.linalg.norm(direction)
    lengths = geometry.get("lengths_mm", [None] * len(geometry["axes"]))
    p = np.r_[
        np.asarray(geometry["vertices"][j]) / 1000.0,
        direction,
        (lengths[j] or 60.0) / 1000.0,
        np.clip(np.deg2rad(geometry["openings_deg"][j]), 0.12, MAX_OPENING_RAD),
        np.log(np.deg2rad(np.clip(geometry["widths_deg"][j], 1.2, 16.0))),
    ]
    contain_finite(p)
    return p


def starts(event, model, bank, geometry, shapes, amp):
    """Shortlist both the optical bank and newly observed residual axes."""
    from timed_small_circles import starts as residual_starts

    count = len(shapes) - 1
    try:
        retained = [path_from_geometry(geometry, j) for j in range(count)]
    except ValueError:
        return []
    options = [(retained, amp, "incumbent")]
    other = bank.get(str(count + 1))
    if other is not None:
        try:
            paths = [path_from_geometry(other, j) for j in range(count + 1)]
            options.append((paths, np.asarray(other["weights"]) * amp.sum(), "count_bank"))
        except ValueError:
            pass
    # Use the excess after fitting all current light, including unhit PMTs.
    # No cut on previous background ownership is applied. A real short
    # leg can be hidden in the foreground of a wide optical cone.
    for path, strength in residual_starts(event, model, bank, geometry, shapes, amp):
        try:
            contain_finite(path)
        except ValueError:
            continue
        options.append(([*retained, path], np.r_[amp, strength], "timed_small_circle"))
    ranked = []
    for paths, initial, kind in options:
        basis = np.vstack([shapes[0], *[optical_shape(model, p, 4)[0] for p in paths]])
        if np.any(basis[1:].sum(1) <= 0.0):
            continue
        ll, photons, error = profile(basis, model.obs, model.ix, initial, budget=24)
        if error <= 1e-5:
            ranked.append((ll, np.asarray(paths), photons, kind))
    return ranked


def solve(event, prototype, paths, photons, budget):
    """Refine every axis and light yield together under one charge likelihood."""
    k = len(paths)
    seed = dict(
        vertices=1000.0 * paths[:, :3],
        axes=paths[:, 3:6].copy(),
        openings_deg=np.rad2deg(paths[:, 7]),
        widths_deg=np.rad2deg(np.exp(paths[:, 8])),
        weights=photons / photons.sum(),
    )
    try:
        model = Analog(event, seed, prototype.bg, "finite", 8)
    except (ValueError, AssertionError):
        return None
    model.x = np.r_[
        paths.ravel(), np.clip(np.log(photons[1:] / photons[0]), -8.0, 8.0), np.log(photons.sum())
    ]
    for j in range(k):
        model.bounds[9 * j + 8] = (np.log(np.deg2rad(1.2)), np.log(np.deg2rad(16.0)))
    x, calls = endpoint(model, budget)
    if x is None:
        return None
    paths = x[: 9 * k].reshape(k, 9)
    from scipy.special import softmax

    photons = np.exp(x[-1]) * softmax(np.r_[0.0, x[9 * k : -1]])
    basis = np.vstack([prototype.bg, *[optical_shape(prototype, p, 16)[0] for p in paths]])
    ll, photons, error = profile(basis, model.obs, model.ix, photons, budget=32)
    if error > 1e-5:
        return None
    high = np.vstack([prototype.bg, *[optical_shape(prototype, p, 32)[0] for p in paths]])
    total = photons @ high
    score = model.obs.score(total, model.ix)[0]
    error = abs(score - ll)
    if error >= 0.01:
        return None
    fractions = (photons[:, None] * high[:, model.ix] / total[model.ix]).T
    support = [float(np.sum(1.0 - model.obs.no_source(total[model.ix], f))) for f in fractions.T[1:]]
    if min(support) < 30.0 or set(fractions.argmax(1)) - {0} != set(range(1, k + 1)):
        return None
    return dict(
        ll=float(score),
        paths=paths,
        photons=photons,
        fractions=fractions,
        shapes=high,
        observed_support=support,
        evaluations=calls,
        quadrature_ll_error=float(error),
    )


def prompt(event, time, fractions):
    """Retain charge near the measured wavefront, with a relative clock."""
    q = np.minimum(event.charges_pe, 2.0)
    residual = (event.times_ns - np.median(event.times_ns))[:, None] - time
    weight = q[:, None] * fractions[:, 1:]
    clocks = [weighted_quantile(r, w, [0.5])[0] for r, w in zip(residual.T, weight.T)]
    return weight * np.exp(-0.5 * ((residual - clocks) / 1.65) ** 2), np.asarray(clocks)


def physical_start(event, model, modes, shapes, amp):
    """Profile the best distinct cone set from fitted charged alternatives."""
    from track_promotion import optical_starts

    best = None
    for track, state, fractions, _ in modes:
        if fractions.shape[1] != len(amp) + 1:
            continue
        paths = optical_starts(track, state, fractions)
        if paths is None:
            continue
        try:
            for p in paths:
                contain_finite(p)
        except (ValueError, ArithmeticError):
            continue
        basis = np.vstack([shapes[0], *[optical_shape(model, p, 4)[0] for p in paths]])
        if np.any(basis[1:].sum(1) <= 0.0):
            continue
        q = np.minimum(event.charges_pe, 2.0)
        yields = q @ fractions
        initial = amp.sum() * yields / yields.sum()
        ll, photons, error = profile(basis, model.obs, model.ix, initial, budget=20)
        if error <= 1e-5 and (best is None or ll > best[0]):
            best = ll, paths, photons, "physical_sections"
    return best


def supports(event, old_time, incumbent, result):
    """Require separate prompt light and prevent splitting one smooth core."""
    shares, paths = result["fractions"], result["paths"]
    old, _ = prompt(event, old_time, incumbent)
    times = np.column_stack([wavefront(event.pmt_positions_mm, p)[0] for p in paths])
    new, clocks = prompt(event, times, shares)
    q = np.minimum(event.charges_pe, 2.0)
    overlap = (incumbent[:, 1:].T * q) @ shares[:, 1:]
    a, b = linear_sum_assignment(-overlap)
    count = incumbent.shape[1] - 1
    k = len(paths)
    kept = np.minimum(old[:, a], new[:, b]).sum(0) / np.maximum(old[:, a].sum(0), 1e-20)
    if k == count:
        return dict(matching=b, matched_prompt_fraction=kept.tolist()) if min(kept) >= 0.55 else None
    (extra,) = set(range(k)) - set(b)
    old_charge = old.sum(1)
    new_charge = new.sum(1)
    # Soft footprints allow a stronger candidate to divide a broad old ring
    # while preserving the actual prompt charge recorded on those receivers.
    kept_total = np.minimum(old, new_charge[:, None]).sum(0) / np.maximum(old.sum(0), 1e-20)
    distinct = (shares.argmax(1) == extra + 1) & (new[:, extra] >= 0.22 * q)
    separated = distinct & (np.max(np.delete(shares[:, 1:], extra, axis=1), axis=1) < 0.35)
    if (
        min(kept_total) < 0.65
        or min(kept) < 0.38
        or separated.sum() < 18
        or len(np.unique(event.pmt_ids[separated] // 100)) < 3
        or np.ptp(clocks) > 9.0
    ):
        return None
    # For nearby cones, opening and phase can be nearly degenerate with a
    # single slowing track. Widely separated vertices can be conversions.
    p = paths[extra]
    # A free wide cone may fit residual background while its axis and
    # center are poorly constrained. Require a resolved optical ridge.
    if np.exp(p[-1]) > np.deg2rad(4.0):
        return None
    # Two almost saturated, short cones can partition one bent
    # electromagnetic track. Prefer the unchanged count in this case
    # until the coupled shower competitor has enough evidence.
    if k == 2 and np.all(paths[:, 7] > np.deg2rad(41.0)) and np.max(paths[:, 6]) < 0.20:
        return None
    links = []
    for j in b:
        other = paths[j]
        axis_dot = float(p[3:6] @ other[3:6])
        delta = p[:3] - other[:3]
        transverse = np.linalg.norm(delta - (delta @ other[3:6]) * other[3:6])
        if axis_dot > np.cos(np.deg2rad(15.0)) and transverse < 0.25:
            return None
        for incoming, outgoing in ((extra, j), (j, extra)):
            x, y = paths[incoming], paths[outgoing]
            after = x[:3] + 0.5 * x[6] * x[3:6]
            before = y[:3] - 0.5 * y[6] * y[3:6]
            gap = np.linalg.norm(after - before)
            delay = 500.0 * (x[6] * np.cos(x[7]) + y[6] * np.cos(y[7]))
            flight = N_PHASE * delay / C
            mismatch = abs(clocks[outgoing] - clocks[incoming] - flight)
            links.append((float(gap), float(mismatch)))
    if not any(gap <= 0.24 and mismatch <= 2.3 for gap, mismatch in links):
        return None
    novel = np.minimum(np.maximum(new_charge - old_charge, 0.0), new[:, extra])
    if novel[separated].sum() < 8.0:
        return None
    return dict(
        matching=np.r_[b, extra],
        matched_prompt_fraction=kept.tolist(),
        retained_prompt_fraction=kept_total.tolist(),
        separated_pmts=int(separated.sum()),
        added_prompt_pe=float(novel[separated].sum()),
        relative_clocks_ns=clocks.tolist(),
        optical_join_checks=links,
    )


def fit(event, model, bank, geometry, shapes, amp, incumbent, budget, *, current, competing_modes=None):
    """Compare joint finite-cone fits with the refitted incumbent."""
    from resolved_ring_evidence import resolve

    original = model.obs.score(amp @ shapes, model.ix)[0]
    count = incumbent.shape[1] - 1
    old_time = arrival(event, geometry)
    null = None
    improved = original
    options = starts(event, model, bank, geometry, shapes, amp)
    for _, paths, photons, kind in options:
        if kind != "incumbent":
            continue
        null = solve(event, model, paths, photons, min(budget, 28))
        if null is not None:
            improved = max(improved, null["ll"])

    # Give the extra ring and the current count the same movable geometry.
    # Count changes pay for the new coordinates before their timing check.
    penalty = 9.0 * np.log(len(model.active))
    expanded = [row for row in options if len(row[1]) == count + 1]
    finalists = sorted(expanded, key=lambda row: row[0], reverse=True)[:2]
    track_modes = None
    accepted = False

    def fallback():
        nonlocal track_modes
        if not accepted and competing_modes is not None and count == 1:
            if track_modes is None:
                track_modes = competing_modes()
            option = physical_start(event, model, track_modes, shapes, amp)
            if option is not None:
                yield option
        if not accepted and null is not None:
            from junction_residual_seeds import start as junction_start

            option = junction_start(
                event, model, null["paths"], null["fractions"], null["shapes"], null["photons"]
            )
            if option is not None:
                yield option

    for _, paths, photons, kind in chain(finalists, fallback()):
        trial = solve(event, model, paths, photons, budget)
        if trial is None or 2.0 * (trial["ll"] - improved) <= penalty:
            continue
        check = supports(event, old_time, incumbent, trial)
        if check is None:
            continue
        # The physical and junction seeds share geometry with a prior fit.
        # Require a resolved turn before counting a new optical ridge.
        evidence = resolve(
            event,
            model,
            current,
            trial,
            separated_axis=kind in ("physical_sections", "junction_residual"),
        )
        if evidence is None:
            continue
        trial["conditional_support"] = evidence
        if competing_modes is not None and (count == 1 or kind == "physical_sections"):
            from track_promotion import corroborated

            if track_modes is None:
                track_modes = competing_modes()
            proof = corroborated(event, trial, evidence, track_modes)
            trial["mode_agreement"] = proof
            if proof is None:
                from soft_light_partition import keep_main

                # Reassign diffuse light if the extra optical cone lacks
                # its own corroborating charged trajectory.
                soft = keep_main(
                    event, model, geometry, shapes, amp, incumbent, trial, evidence["extra_component"]
                )
                if soft is not None:
                    ll, shares, details = soft
                    yield ll, shares, dict(details, seed=kind, null_ll=improved, penalty=penalty)
                continue
        matching = check.pop("matching")
        share = trial.pop("fractions")[:, np.r_[0, 1 + matching]]
        trial.pop("shapes")
        trial.update(check, seed=kind, null_ll=improved, penalty=penalty)
        accepted = True
        yield trial["ll"], share, trial
