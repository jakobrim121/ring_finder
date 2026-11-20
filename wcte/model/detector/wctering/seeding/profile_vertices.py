"""Detector-only contained vertex-profile foreground experiment; no file I/O."""

from __future__ import annotations
import copy
import numpy as np
from wctering.seeding.optical_seed_fit import circle_seed, seeds
from wctering.seeding.cone_seeds import _best_coherence_cone
from wctering.seeding.geometry import (
    wcte_contains_vertex,
    WCTE_PRISM_FACE_NORMALS_XZ,
    WCTE_PRISM_APOTHEM_MM,
    WCTE_PRISM_Y_MIN_MM,
    WCTE_PRISM_Y_MAX_MM,
)

LOG_BAND = float(np.log(20.0))
GROUP_VELOCITY = 299.792458 / 1.388


def canonical_axis(axis):
    a = np.asarray(axis, float).copy()
    norm = np.linalg.norm(a)
    if not np.isfinite(a).all() or norm < 1.0e-12:
        raise ValueError("Invalid profile axis")
    a /= norm
    if a[int(np.argmax(np.abs(a)))] < 0:
        a = -a
    return a


def contained_line_limits(vertex, axis):
    """Exact line/prism intersections; no detector-origin or clock assumption."""
    v = np.asarray(vertex, float)
    a = canonical_axis(axis)
    if not wcte_contains_vertex(v, tolerance_mm=1.0e-5):
        raise ValueError("Profile anchor outside detector")
    normals = np.zeros((len(WCTE_PRISM_FACE_NORMALS_XZ) + 2, 3))
    normals[:-2, 0] = WCTE_PRISM_FACE_NORMALS_XZ[:, 0]
    normals[:-2, 2] = WCTE_PRISM_FACE_NORMALS_XZ[:, 1]
    normals[-2, 1] = 1.0
    normals[-1, 1] = -1.0
    bounds = np.r_[
        np.repeat(WCTE_PRISM_APOTHEM_MM, len(normals) - 2),
        WCTE_PRISM_Y_MAX_MM,
        -WCTE_PRISM_Y_MIN_MM,
    ]
    slack = bounds - normals @ v
    rate = normals @ a
    lo, hi = -np.inf, np.inf
    for s, r in zip(slack, rate):
        if r > 1.0e-14:
            hi = min(hi, s / r)
        elif r < -1.0e-14:
            lo = max(lo, s / r)
    if not np.isfinite([lo, hi]).all() or lo > 1.0e-4 or hi < -1.0e-4:
        raise ValueError("Invalid line intersection")
    return float(lo), float(hi), a


def vertex_proposals(vertex, axis, covariance):
    lo, hi, a = contained_line_limits(vertex, axis)
    v = np.asarray(vertex, float)
    cov = np.asarray(covariance, float)
    sigma = np.sqrt(max(float(a @ cov @ a), 0.0))
    result = []
    for mult in (-3.0, -1.0, 1.0, 3.0):
        s = mult * sigma
        if lo + 1.0e-5 < s < hi - 1.0e-5:
            result.append(("local", s, v + s * a))
    for fraction in (0.35, 0.75, 0.95):
        result.extend(
            [
                ("contained", fraction * lo, v + fraction * lo * a),
                ("contained", fraction * hi, v + fraction * hi * a),
            ]
        )
    for _, _, point in result:
        if not wcte_contains_vertex(point, tolerance_mm=1.0e-5):
            raise RuntimeError("Uncontained profile proposal")
    return sorted(result, key=lambda x: (x[0], *np.round(x[2], 8)))


def free_time_profile(positions, times, charges, vertex, mask):
    m = np.asarray(mask, bool)
    if not m.any():
        m = np.ones(len(times), bool)
    residual = (
        np.asarray(times)[m] - np.linalg.norm(np.asarray(positions)[m] - vertex, axis=1) / GROUP_VELOCITY
    )
    weights = np.minimum(np.asarray(charges)[m], 2.0)
    order = np.argsort(residual, kind="stable")
    cum = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cum, 0.5 * cum[-1])), len(order) - 1)
    t0 = float(residual[order[index]])
    sigma = max(0.8, float(np.sqrt(np.average((residual - t0) ** 2, weights=weights))))
    return t0, sigma


def profile_seed_bank(diagnostics, k, positions, charges, times, old):
    original = _seed_bank(diagnostics, k, positions, charges)
    bank = [dict(name=family, scope="original", seed=seed) for family, seed in original]
    if k == 1:
        for other in (2, 3):
            extra = seeds(diagnostics, other, positions, charges, "independent")
            if extra:
                bank.extend(
                    dict(
                        name=f"h{other}_component{j + 1}",
                        scope="additional_family",
                        seed=[r],
                    )
                    for j, r in enumerate(extra)
                )
    anchor = seeds(diagnostics, k, positions, charges, "independent")
    h = (
        diagnostics.get("full_event_hypotheses", {})
        if k < 3
        else diagnostics.get("three_ring_hypothesis_test", {}).get("full_event_hypotheses", {})
    )
    fits = [h.get("h1_fit", {})] if k == 1 else h.get("h2_fits" if k == 2 else "h3_fits", [])
    if anchor and len(fits) == k:
        for j, fit in enumerate(fits):
            if not fit.get("valid") or "covariance_xyz_mm2" not in fit:
                continue
            v = np.asarray(fit["vertex_mm"], float)
            if not wcte_contains_vertex(v, tolerance_mm=1.0e-5):
                continue
            axis = fit.get("covariance_major_axis", fit.get("information_weak_axis"))
            for scope, s, point in vertex_proposals(v, axis, fit["covariance_xyz_mm2"]):
                rings = copy.deepcopy(anchor)
                rings[j]["vertex"] = point
                mask = np.asarray(old) == j + 1
                if mask.sum() >= 10:
                    rings[j]["axis"], rings[j]["opening"] = circle_seed(
                        positions[mask], charges[mask], point, rings[j]["axis"]
                    )
                bank.append(
                    dict(
                        name=f"{scope}_ring{j + 1}_s{s:+.8f}",
                        scope=scope,
                        profile_ring=j + 1,
                        displacement_mm=s,
                        seed=rings,
                    )
                )
    # Duplicate proposals retain their earliest scope (original before profile).
    seen = set()
    result = []
    for entry in bank:
        rings = copy.deepcopy(entry["seed"])
        if len(rings) != k or any(not wcte_contains_vertex(r["vertex"], tolerance_mm=1.0e-5) for r in rings):
            continue
        for j, r in enumerate(rings):
            r["vertex"] = np.asarray(r["vertex"], float)
            r["axis"] = np.asarray(r["axis"], float)
            r["t0"], r["sigma_t"] = free_time_profile(
                positions, times, charges, r["vertex"], np.asarray(old) == j + 1
            )
        code = tuple(
            np.round(
                np.concatenate([np.r_[r["vertex"], r["axis"], r["opening"]] for r in rings]),
                8,
            )
        )
        if code in seen:
            continue
        seen.add(code)
        result.append(dict(entry, seed=rings))
    return result


def _seed_bank(diagnostics, k, pos, q):
    result = []
    for family in ("legacy", "circle", "independent"):
        rings = seeds(diagnostics, k, pos, q, family)
        if rings is not None:
            result.append((family, rings))
    previous = diagnostics.get("analytic_partition", {})
    if previous.get("applied") and previous.get("count") == k and len(previous.get("components", [])) == k:
        result.append(
            (
                "frozen_optical",
                [
                    dict(
                        vertex=np.array(c["vertex_mm"]),
                        axis=np.array(c["direction"]),
                        opening=c["opening_deg"],
                        t0=0.0,
                        sigma_t=1.0,
                    )
                    for c in previous["components"]
                ],
            )
        )
    if k == 1 and result:
        ring = dict(result[0][1][0])
        delta = pos - ring["vertex"]
        u = delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-10)
        ring["axis"], ring["opening"] = _best_coherence_cone(u, np.minimum(q, 2.0))
        result.append(("coherence", [ring]))
    seen = set()
    unique = []
    for family, rings in result:
        if any(not wcte_contains_vertex(r["vertex"], tolerance_mm=1e-5) for r in rings):
            continue
        code = tuple(
            np.round(
                np.concatenate([np.r_[r["vertex"], r["axis"], r["opening"]] for r in rings]),
                7,
            )
        )
        if code not in seen:
            seen.add(code)
            unique.append((family, rings))
    return unique
