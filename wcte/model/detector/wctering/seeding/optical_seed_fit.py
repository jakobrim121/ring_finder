"""Analytic optical partition refinement after the frozen ring-count decision.

No classifier, trained weights, ROOT truth, particle identity, beam energy or
source filename enters this module. For each event, maximize a conditional
all-active-PMT light likelihood for K cones plus diffuse background. Event-local
multistart and residual Hough proposals repair biased partial-arc seeds.

The angular Gaussian is an effective cone profile, not a full photon-transport
simulation. A fixed 2 PE cap limits the influence of bright PMTs when this
approximate profile is imperfect. All geometry parameters and mixture weights
are fitted anew within the event. See docs/V2_25_ANALYTIC_PARTITION.md.
"""

import numpy as np
from scipy.optimize import minimize
from wctering.logmath import logsumexp
from functools import lru_cache
from wctering.seeding.io import active_wcte_ids
from wctering.seeding.detector import load_wcte_physical_geometry
from wctering.seeding.geometry import wcte_contains_vertex
from wctering.seeding.cone_seeds import _best_localized_cone, _best_coherence_cone
import time


@lru_cache(maxsize=1)
def active_geometry():
    g = load_wcte_physical_geometry()
    ids = np.asarray(sorted(active_wcte_ids()))
    ix = np.asarray([g.id_to_index[int(i)] for i in ids])
    return ids, g.positions_mm[ix], g.inward_normals[ix]


def unit(x):
    return np.asarray(x, float) / max(np.linalg.norm(x), 1e-15)


def axis_angles(a):
    a = unit(a)
    return np.array([np.arccos(np.clip(a[2], -1, 1)), np.arctan2(a[1], a[0])])


def axis_derivatives(theta, phi):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    a = np.array([st * cp, st * sp, ct])
    da = np.array([[ct * cp, ct * sp, -st], [-st * sp, st * cp, 0.0]])
    return a, da


def circle_seed(pos, q, vertex, seed_axis):
    u = pos - vertex
    u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-12)
    w = np.minimum(q, 2)
    axis = unit(seed_axis)
    for _ in range(4):
        mean = np.average(u, axis=0, weights=w + 1e-10)
        cov = (u - mean).T @ ((u - mean) * w[:, None]) / max(w.sum(), 1e-12)
        _, vec = np.linalg.eigh(cov)
        new = vec[:, 0]
        if new @ mean < 0:
            new = -new
        opening = np.arccos(np.clip(new @ mean, -1, 1))
        if not np.deg2rad(10) <= opening <= np.deg2rad(55):
            break
        axis = new
        residual = np.arccos(np.clip(u @ axis, -1, 1)) - opening
        w = np.minimum(q, 2) * np.exp(-0.5 * (residual / np.deg2rad(10)) ** 2)
    opening = np.average(np.arccos(np.clip(u @ axis, -1, 1)), weights=w + 1e-10)
    return axis, float(np.clip(np.rad2deg(opening), 12, 55))


def seeds(d, k, positions, charges, family="legacy"):
    full = d.get("full_event_hypotheses", {})
    if k == 1:
        rings = [full.get("h1_partition_ring")]
        labels = np.ones(len(charges), int)
        fits = [full.get("h1_fit", {})]
    else:
        h = full if k == 2 else (d.get("three_ring_hypothesis_test") or {}).get("full_event_hypotheses", {})
        rings = h.get("partition_rings", [])
        labels = np.asarray(h.get("partition_labels", []), int)
        fits = h.get("h2_fits" if k == 2 else "h3_fits", [])
    if len(rings) != k or any(r is None for r in rings) or len(labels) != len(charges):
        return None
    out = []
    for i, r in enumerate(rings):
        v = np.asarray(r["vertex"], float)
        axis = np.asarray(r["direction"], float)
        opening = float(r["theta_c"])
        if family == "independent" and len(fits) == k and fits[i].get("valid", False):
            v = np.asarray(fits[i]["vertex_mm"], float)
        if family in ["circle", "independent"]:
            m = labels == i + 1
            if m.sum() >= 10:
                axis, opening = circle_seed(positions[m], charges[m], v, axis)
        out.append(dict(vertex=v, axis=axis, opening=opening, t0=r["t0"], sigma_t=r["sigma_t"]))
    return out


from numba import njit


@njit(cache=True)
def shape_factors(angle, cosine, axis_jacobian, log_exposure, opening, width, variance):
    """Fuse elementwise factors while retaining their original operation order.

    Transcendentals, BLAS products, normalization and reductions stay in NumPy.
    Variance is deliberately evaluated by NumPy outside this loop: its scalar
    power can round differently from Numba's multiplication for exponent two.
    This loop removes temporary arrays; it does not use fast-math or threads.
    """
    logkernel = np.empty(len(angle))
    jacobian = np.empty((len(angle), 4))
    for index in range(len(angle)):
        residual = angle[index] - opening
        squared = (residual / width) ** 2
        logkernel[index] = log_exposure[index] - 0.5 * squared
        scale = residual / (variance * np.sqrt(1 - cosine[index] ** 2))
        jacobian[index, 0] = scale * axis_jacobian[index, 0]
        jacobian[index, 1] = scale * axis_jacobian[index, 1]
        jacobian[index, 2] = residual / variance
        jacobian[index, 3] = squared
    return logkernel, jacobian


def optical_fit(
    pmt_ids,
    positions,
    charges,
    times,
    seed,
    *,
    charge_cap=2.0,
    width_deg=8.0,
    maxiter=75,
    _debug=False,
    _initial_parameters=None,
):
    ids, allpos, normals = active_geometry()
    indices = np.searchsorted(ids, pmt_ids)
    assert np.array_equal(ids[indices], pmt_ids)
    k = len(seed)
    weights = np.minimum(charges, charge_cap)
    total = weights.sum()
    views = []
    for r in seed:
        delta = allpos - r["vertex"]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        u = delta / distance[:, None]
        inc = np.clip(np.sum(normals * (-u), axis=1), 0, 1)
        exposure = np.maximum(inc / (distance**2 + 100.0**2), 1e-30)
        views.append((u, distance, exposure))
    bgview = np.mean([v[2] / v[2].sum() for v in views], axis=0)
    background = 0.5 * bgview + 0.5 / len(ids)
    initial = []
    bounds = []
    for r in seed:
        theta, phi = axis_angles(r["axis"])
        initial.extend(
            [
                theta,
                phi,
                np.deg2rad(np.clip(r["opening"], 12, 55)),
                np.log(np.deg2rad(width_deg)),
            ]
        )
        bounds.extend(
            [
                (1e-5, np.pi - 1e-5),
                (-4 * np.pi, 4 * np.pi),
                (np.deg2rad(12), np.deg2rad(55)),
                (np.log(np.deg2rad(2.5)), np.log(np.deg2rad(20))),
            ]
        )
    initial.extend([0.0] * k)
    bounds.extend([(-8.0, 8.0)] * k)
    initial = np.asarray(initial)
    if _initial_parameters is not None:
        initial = np.asarray(_initial_parameters, dtype=float).copy()
    # Vertices and receiver exposure are fixed throughout this seed fit.
    # Hoist their logarithms; keep the original likelihood operation order.
    log_exposures = [np.log(view[2]) for view in views]
    # Resolve NumPy's own loops once per fit. Unsupported NumPy builds keep
    # the ordinary expression below; both paths use the same arithmetic.
    from wctering.logmath import _vector_backend
    from wctering.seeding.seed_component import component

    vector_backend = _vector_backend()
    previous_geometry = None
    previous_shapes = None
    previous_derivatives = None

    def evaluate(x, detail=False):
        nonlocal previous_geometry, previous_shapes, previous_derivatives
        shapes = [background]
        derivatives = []
        geometry = x[: 4 * k]
        unchanged = previous_geometry is not None and np.array_equal(geometry, previous_geometry)
        for i, (u, distance, exposure) in () if unchanged else enumerate(views):
            theta, phi, opening, logwidth = x[4 * i : 4 * i + 4]
            width = np.exp(logwidth)
            axis, da = axis_derivatives(theta, phi)
            if vector_backend is None:
                cosine = np.clip(u @ axis, -1 + 1e-10, 1 - 1e-10)
                angle = np.arccos(cosine)
                logkernel, jac = shape_factors(
                    angle, cosine, u @ da.T, log_exposures[i], opening, width, width**2
                )
                shape = np.exp(logkernel - logsumexp(logkernel))
                jac -= shape @ jac
            else:
                _, function, loops = vector_backend
                shape, jac = component(
                    function,
                    loops,
                    u,
                    axis,
                    da,
                    log_exposures[i],
                    opening,
                    width,
                    width**2,
                )
            shapes.append(shape)
            derivatives.append(jac[indices])
        if unchanged:
            shapes, derivatives = previous_shapes, previous_derivatives
        else:
            previous_geometry = geometry.copy()
            previous_shapes, previous_derivatives = shapes, derivatives
        logits = np.concatenate(([0.0], x[4 * k :]))
        pi = np.exp(logits - logsumexp(logits))
        shapes = np.asarray(shapes)
        terms = pi[:, None] * np.maximum(shapes[:, indices], 1e-300)
        density = terms.sum(axis=0)
        resp = terms / np.maximum(density, 1e-300)
        ll = float(weights @ np.log(np.maximum(density, 1e-300)))
        grad = np.zeros_like(x)
        for i, jac in enumerate(derivatives):
            grad[4 * i : 4 * i + 4] = -((weights * resp[i + 1]) @ jac)
        grad[4 * k :] = -((resp[1:] @ weights) - pi[1:] * total)
        if detail:
            return dict(
                ll=ll,
                posterior=resp,
                shapes=shapes,
                weights=pi,
                parameters=x,
                axes=np.asarray([axis_derivatives(x[4 * i], x[4 * i + 1])[0] for i in range(k)]),
                openings_deg=np.rad2deg(x[np.arange(k) * 4 + 2]),
                widths_deg=np.rad2deg(np.exp(x[np.arange(k) * 4 + 3])),
                vertices=np.asarray([r["vertex"] for r in seed]),
                seed=seed,
            )
        return -ll, grad

    if _debug:
        return evaluate, initial, bounds
    fit = minimize(
        evaluate,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5, "maxls": 20},
    )
    result = evaluate(fit.x, True)
    result.update(converged=bool(fit.success), iterations=int(fit.nit))
    result["labels"] = result["posterior"].argmax(axis=0)
    return result


SCHEMA_VERSION = "wcte-analytic-optical-partition-v225-preserve-foreground"


def _residual_proposals(positions, charges, fit):
    """Replace one cone using unexplained light; choose by the same likelihood."""
    proposals = []
    for j in range(len(fit["seed"])):
        residual = np.minimum(charges, 2.0) * (fit["posterior"][0] + fit["posterior"][j + 1])
        if residual.sum() < 15:
            continue
        vertex = fit["vertices"][j]
        u = positions - vertex
        u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-10)
        for function in (_best_localized_cone, _best_coherence_cone):
            axis, opening = function(u, residual)
            proposal = [dict(r) for r in fit["seed"]]
            for i in range(len(proposal)):
                proposal[i].update(axis=fit["axes"][i], opening=fit["openings_deg"][i])
            proposal[j].update(axis=axis, opening=opening)
            proposals.append(proposal)
    return proposals


def _admissible(fit, k, n):
    """Require finite probabilities and preserve every counted component."""
    p = fit["posterior"]
    return (
        p.shape == (k + 1, n)
        and np.all(np.isfinite(p))
        and np.isfinite(fit["ll"])
        and set(map(int, fit["labels"])) - {0} == set(range(1, k + 1))
    )


def _legacy_labels(diagnostics, k, n):
    policy = diagnostics.get("final_partition_background_assignment") or {}
    if policy.get("background_assignment_applied"):
        labels = np.asarray(policy.get("partition_labels", []), dtype=np.int64)
        if labels.shape == (n,):
            return labels
    if k == 1:
        return np.ones(n, dtype=np.int64)
    full = (
        (diagnostics.get("full_event_hypotheses") or {})
        if k == 2
        else ((diagnostics.get("three_ring_hypothesis_test") or {}).get("full_event_hypotheses") or {})
    )
    return np.asarray(full.get("partition_labels", []), dtype=np.int64)


def preserve_foreground_posterior(posterior, legacy_labels):
    """Condition optical ring membership on the existing foreground mask.

    This changes only the ring-to-ring assignment. It cannot remove or add a
    foreground pulse. Background certainty here is inherited from that mask,
    not inferred as a calibrated optical probability.
    """
    posterior = np.asarray(posterior, dtype=float)
    foreground = np.asarray(legacy_labels) > 0
    ring = posterior[1:]
    mass = ring.sum(axis=0)
    conditioned = np.divide(
        ring,
        mass[None, :],
        out=np.full_like(ring, 1.0 / len(ring)),
        where=mass[None, :] > 0,
    )
    out = np.zeros_like(posterior)
    out[0, ~foreground] = 1.0
    out[1:, foreground] = conditioned[:, foreground]
    return out


def refine_partition(
    pmt_ids,
    positions,
    charges,
    times,
    diagnostics,
    count,
    *,
    _initial_candidates=None,
    _optimizer=None,
):
    """Return a detector-only refinement, or an explicit legacy fallback.

    Input rows are the retained prompt pulses. The count is an already fixed
    detector prediction. It is never inferred again or changed here.
    """
    optimizer = optical_fit if _optimizer is None else _optimizer
    started = time.perf_counter()
    k = int(count)
    base = dict(
        schema=SCHEMA_VERSION,
        model="analytic_v225",
        count=k,
        applied=False,
        charge_cap_pe=2.0,
        background_policy="preserve",
    )
    ids = np.asarray(pmt_ids, dtype=np.int64)
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(ids)
    if k == 0 or n == 0:
        base.update(reason="zero_count_or_empty_event", runtime_s=0.0)
        return base
    if k == 1:
        base.update(reason="single_ring_partition_preserved", runtime_s=0.0)
        return base
    if k not in (1, 2, 3):
        raise ValueError("count must be 0, 1, 2, or 3")
    if positions.shape != (n, 3) or charges.shape != (n,) or times.shape != (n,):
        raise ValueError("prepared partition inputs must have aligned shapes")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(charges))
        or not np.all(np.isfinite(times))
        or np.any(charges <= 0)
    ):
        base.update(reason="invalid_detector_values", runtime_s=time.perf_counter() - started)
        return base
    active, allpos, _ = active_geometry()
    ix = np.searchsorted(active, ids)
    if np.any(ix >= len(active)) or not np.array_equal(active[np.minimum(ix, len(active) - 1)], ids):
        base.update(reason="unrecognized_active_pmt", runtime_s=time.perf_counter() - started)
        return base
    if not np.allclose(positions, allpos[ix], rtol=0, atol=1e-6):
        base.update(reason="nonstandard_pmt_geometry", runtime_s=time.perf_counter() - started)
        return base
    fits = []
    numerical_failures = []

    def attempt(seed, family):
        if seed is None:
            return
        if any(not wcte_contains_vertex(r["vertex"], tolerance_mm=1e-5) for r in seed):
            return
        try:
            fit = optimizer(ids, positions, charges, times, seed, charge_cap=2.0)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            numerical_failures.append(type(exc).__name__)
            return
        fit["family"] = family
        if np.isfinite(fit["ll"]) and np.all(np.isfinite(fit["posterior"])):
            fits.append(fit)

    if _initial_candidates is None:
        for family in ("legacy", "circle", "independent"):
            attempt(seeds(diagnostics, k, positions, charges, family), family)
    else:
        # v2.26 already evaluated these exact 75-iteration starting fits to
        # form counter evidence. Reuse the unpolished candidates so residual
        # proposals and the final partition retain v2.25 numerical behavior.
        fits.extend(_initial_candidates)
    if fits and k > 1:
        best = max(fits, key=lambda f: f["ll"])
        for proposal in _residual_proposals(positions, charges, best):
            attempt(proposal, "residual")
    admissible = [f for f in fits if _admissible(f, k, n)]
    base.update(
        candidates=len(fits),
        rejected_collapsed_candidates=len(fits) - len(admissible),
        numerical_failures=numerical_failures,
    )
    if not admissible:
        base.update(reason="no_admissible_optical_fit", runtime_s=time.perf_counter() - started)
        return base
    best = max(admissible, key=lambda f: f["ll"])
    polished = False
    if not best["converged"]:
        candidate = optimizer(
            ids,
            positions,
            charges,
            times,
            best["seed"],
            charge_cap=2.0,
            maxiter=150,
            _initial_parameters=best["parameters"],
        )
        if _admissible(candidate, k, n) and candidate["ll"] >= best["ll"]:
            candidate["family"] = best["family"]
            candidate["iterations"] += best["iterations"]
            best = candidate
            polished = True
    posterior = best["posterior"]
    old = _legacy_labels(diagnostics, k, n)
    if old.shape != (n,) or not np.all(np.isin(old, range(k + 1))):
        base.update(
            reason="no_aligned_legacy_foreground",
            runtime_s=time.perf_counter() - started,
        )
        return base
    posterior = preserve_foreground_posterior(posterior, old)
    labels = posterior.argmax(axis=0)
    if set(map(int, labels)) - {0} != set(range(1, k + 1)):
        base.update(
            reason="foreground_preserving_partition_collapsed",
            runtime_s=time.perf_counter() - started,
        )
        return base
    base.update(
        applied=True,
        reason="maximum_likelihood_admissible_fit",
        selected_seed_family=best["family"],
        log_likelihood=best["ll"],
        convergence_polish_applied=polished,
        optimizer_converged=best["converged"],
        optimizer_iterations=best["iterations"],
        partition_labels=labels.tolist(),
        membership_probabilities=posterior.T.tolist(),
        mixture_weights=best["weights"].tolist(),
        components=[
            dict(
                label=i + 1,
                vertex_mm=best["vertices"][i].tolist(),
                direction=best["axes"][i].tolist(),
                opening_deg=float(best["openings_deg"][i]),
                angular_width_deg=float(best["widths_deg"][i]),
                assigned_unique_pmts=int(len(np.unique(ids[labels == i + 1]))),
            )
            for i in range(k)
        ],
        runtime_s=time.perf_counter() - started,
    )
    return base


def apply_analytic_partition(
    diagnostics,
    pmt_ids,
    positions,
    charges,
    times,
    config,
    *,
    _initial_candidates=None,
    _optimizer=None,
):
    """Attach the final partition after all counter decisions have finished."""
    result = refine_partition(
        pmt_ids,
        positions,
        charges,
        times,
        diagnostics,
        diagnostics["predicted_ring_class"],
        _initial_candidates=_initial_candidates,
        _optimizer=_optimizer,
    )
    diagnostics["analytic_partition"] = result
    if result["applied"]:
        old = diagnostics.get("final_partition_background_assignment", {})
        diagnostics["legacy_partition_background_assignment"] = old
        diagnostics["final_partition_background_assignment"] = dict(
            background_assignment_applied=True,
            policy=SCHEMA_VERSION,
            partition_labels=result["partition_labels"],
            reason="optical ring competition with preserved legacy foreground",
        )
    return diagnostics
