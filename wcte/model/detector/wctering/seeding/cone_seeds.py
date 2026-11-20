"""Fit timing vertices and propose single-cone directions.

These are initial hypotheses, not final ring counts. The candidate sphere
search, angular widths and charge weights retain their frozen definitions."""

from __future__ import annotations


import math

import numpy as np

from scipy.optimize import least_squares

from wctering.seeding.geometry import (
    ENFORCE_WCTE_VERTEX_CONTAINMENT,
    contained_point_source_refit,
    wcte_contains_vertex,
)


from wctering.seeding.event_cache import _cached_call, _cached_readonly_array_call
from wctering.seeding.seed_constants import (
    _CAND,
    _COH_SIGMA,
    _SIGMA_THETA,
    _THETA_C0,
    _THETA_C_MAX,
    _VG,
)


def _fit_vertex_t0_uncached(pos, t, w):
    """Charge-weighted point-source timing fit -> (vertex, t0).

    An analytic Jacobian and tight tolerances reduce sensitivity to numerical
    derivative errors. Track-like events can still be ill-conditioned, and a
    different solver build can change the local endpoint. The validation report
    records the actual numerical environment; cross-build identity is not assumed.
    """
    sw = np.sqrt(np.maximum(w, 1e-9))
    x0 = np.r_[np.average(pos, 0, weights=np.maximum(w, 1e-9)), np.median(t)]

    def resid(x):
        d = np.maximum(np.linalg.norm(pos - x[:3], axis=1), 1e-9)
        return sw * (t - x[3] - d / _VG)

    def jac(x):
        vecs = pos - x[:3]
        d = np.maximum(np.linalg.norm(vecs, axis=1), 1e-9)
        J = np.empty((len(pos), 4))
        J[:, :3] = sw[:, None] * vecs / d[:, None] / _VG
        J[:, 3] = -sw
        return J

    s = least_squares(resid, x0, jac=jac, max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    parameters = np.asarray(s.x, dtype=float)
    if ENFORCE_WCTE_VERTEX_CONTAINMENT and not wcte_contains_vertex(parameters[:3]):
        contained = contained_point_source_refit(
            pos,
            t,
            sw,
            _VG,
            parameters,
            time_bound_ns=1000.0,
            max_iterations=300,
        )
        if not bool(contained.get("valid", False)):
            raise RuntimeError("partitioner could not form a contained WCTE vertex")
        parameters = np.asarray(contained["parameters"], dtype=float)
    return parameters[:3], float(parameters[3])


def _fit_vertex_t0(pos, t, w):
    return _cached_call("fit_vertex_t0", _fit_vertex_t0_uncached, pos, t, w)


def _cone_angle_matrix_uncached(unit_directions):
    unit_directions = np.asarray(unit_directions, dtype=float)
    angles = unit_directions @ _CAND.T
    np.clip(angles, -1.0, 1.0, out=angles)
    np.arccos(angles, out=angles)
    np.degrees(angles, out=angles)
    return angles


def _cone_angle_matrix(unit_directions):
    """Candidate-axis angles, reused exactly within one event fit."""

    return _cached_readonly_array_call("cone_angle_matrix", _cone_angle_matrix_uncached, unit_directions)


def _ring_params(pos, t, q, resp):
    """Fit one ring's parameters from charge*responsibility-weighted hits."""
    w = q * resp + 1e-9
    V, t0 = _fit_vertex_t0(pos, t, w)
    d = pos - V
    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    uh = d / dn[:, None]
    ax = np.average(uh, 0, weights=w)
    ax /= np.linalg.norm(ax)
    ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
    thc = float(np.average(ang, weights=w))
    dt = t - t0 - dn / _VG
    sigt = max(
        float(np.sqrt(np.average((dt - np.average(dt, weights=w)) ** 2, weights=w))),
        1.0,
    )
    return dict(
        vertex=V,
        t0=t0,
        direction=ax,
        theta_c=thc,
        sigma_t=min(sigt, 5.0),
        charge=float(w.sum()),
    )


def fit_dominant_ring(pos, t, q, theta0=_THETA_C0, sig=_SIGMA_THETA, near_deg=18.0, iters=3):
    """Fit the dominant ring with a de-contaminated vertex.

    Charge-weighted Hough vote for the cone axis, then RE-MULTILATERATE the vertex
    using ONLY the on-cone hits and refit. Iterating this keeps the dominant-ring
    axis accurate (~9 deg vs ~21 deg if the vertex is fit on the blended hit set),
    which is what makes the downstream peeling work.
    """
    V, t0 = _fit_vertex_t0(pos, t, q)
    rp = None
    for _ in range(iters):
        d = pos - V
        dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        uh = d / dn[:, None]
        ang = _cone_angle_matrix(uh)
        shell = np.exp(-0.5 * ((ang - theta0) / sig) ** 2)
        vote = q @ shell
        u = _CAND[np.argmax(vote)]
        a = np.degrees(np.arccos(np.clip(uh @ u, -1, 1)))
        near = np.abs(a - theta0) < near_deg
        if near.sum() < 6:
            near = np.abs(a - np.average(a, weights=q)) < near_deg
        if near.sum() < 6:
            break
        rp = _ring_params(pos[near], t[near], q[near], np.ones(int(near.sum())))
        V = rp["vertex"]
    return rp if rp is not None else _ring_params(pos, t, q, np.ones(len(q)))


def _azimuthal_coverage(uh_on, axis, w, nbin=12):
    """Fraction of azimuthal sectors around `axis` carrying charge (ring completeness)."""
    a0 = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, a0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi = np.degrees(np.arctan2(uh_on @ e2, uh_on @ e1)) % 360.0
    sectors = np.minimum((phi // (360.0 / nbin)).astype(int), nbin - 1)
    occ = np.bincount(sectors, weights=w, minlength=nbin)
    return float((occ > 0.01 * w.sum()).sum()) / nbin


def _fit_dominant_sharp_uncached(pos, t, q, ntop=40, near_deg=14.0, spread_floor=6.0):
    """Dominant-ring fit that picks the brightest *coherent* cone, not the highest vote.

    Re-ranks the top `ntop` Hough candidates by oncharge * azimuthal_coverage /
    max(angular_spread, spread_floor), all evaluated at one common vertex.
    Falls back to fit_dominant_ring if no candidate qualifies.
    """
    pos = np.asarray(pos, float)
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    V, t0 = _fit_vertex_t0(pos, t, q)
    d = pos - V
    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    uh = d / dn[:, None]
    angc = _cone_angle_matrix(uh)
    shell = np.exp(-0.5 * ((angc - _THETA_C0) / _SIGMA_THETA) ** 2)
    vote = q @ shell
    order = np.argsort(-vote)[:ntop]
    # Profile all top candidates in one hit-by-candidate block.  This is the
    # same weighted-median seed and four fixed-point shell updates as the
    # scalar loop below used historically, but it removes hundreds of tiny
    # NumPy dispatches in the H3 recursive partition bank.
    top_angles = angc[:, order]
    physical = (top_angles >= 12.0) & (top_angles <= 62.0)
    physical_charge = q @ physical
    physical_weight = np.where(physical, q[:, None], 0.0)
    sorted_indices = np.argsort(top_angles, axis=0)
    sorted_angles = np.take_along_axis(top_angles, sorted_indices, axis=0)
    sorted_weights = np.take_along_axis(physical_weight, sorted_indices, axis=0)
    cumulative_weight = np.cumsum(sorted_weights, axis=0)
    median_indices = np.argmax(
        cumulative_weight >= 0.5 * cumulative_weight[-1][None, :],
        axis=0,
    )
    theta = sorted_angles[median_indices, np.arange(len(order))]
    updating = physical_charge >= 0.08 * q.sum()
    weighted_angles = top_angles * q[:, None]
    for _ in range(4):
        near_matrix = np.abs(top_angles - theta[None, :]) < near_deg
        near_charge = q @ near_matrix
        valid_update = updating & (near_charge >= 1.0e-6)
        numerator = np.sum(np.where(near_matrix, weighted_angles, 0.0), axis=0)
        theta = np.where(
            valid_update,
            numerator / np.maximum(near_charge, 1.0e-300),
            theta,
        )
        updating = valid_update
    on_matrix = np.abs(top_angles - theta[None, :]) < near_deg
    best_axis = None
    best_thc = None
    best_score = -np.inf
    for column, i in enumerate(order):
        if physical_charge[column] < 0.08 * q.sum():
            continue
        a = top_angles[:, column]
        thc = float(theta[column])
        on = on_matrix[:, column]
        if on.sum() < 6:
            continue
        spread = float(np.sqrt(np.average((a[on] - thc) ** 2, weights=q[on])))
        cov = _azimuthal_coverage(uh[on], _CAND[i], q[on])
        score = q[on].sum() * cov / max(spread, spread_floor)
        if score > best_score:
            best_score = score
            best_axis = _CAND[i]
            best_thc = thc
    if best_axis is None:
        return fit_dominant_ring(pos, t, q)
    a = np.degrees(np.arccos(np.clip(uh @ best_axis, -1, 1)))
    near = np.abs(a - best_thc) < near_deg
    if near.sum() < 6:
        return fit_dominant_ring(pos, t, q)
    return _ring_params(pos[near], t[near], q[near], np.ones(int(near.sum())))


def fit_dominant_sharp(pos, t, q, ntop=40, near_deg=14.0, spread_floor=6.0):
    """Dominant sharp-ring seed, shared by H2 and H3 within one event."""

    return _cached_call(
        "fit_dominant_sharp",
        _fit_dominant_sharp_uncached,
        pos,
        t,
        q,
        ntop,
        near_deg,
        spread_floor,
    )


def _best_localized_cone_uncached(uh, q, cap=_THETA_C_MAX):
    """Find the brightest tight cone (axis, theta_c in [8, cap]) at the shared vertex.

    A charge-weighted Hough vote over candidate axes at a grid of opening angles;
    the axis with the most concentrated charge wins and theta_c is refined within a
    window, clipped to cap. Used to RE-SEED a ring whose ordinary mean-direction fit
    came out unphysical (theta_c > cap) -- the symptom of an axis pointing at a
    diffuse/contaminated hit-set rather than a real ring.
    """
    uh = np.asarray(uh, float)
    q = np.asarray(q, float)
    angc = _cone_angle_matrix(uh)  # Nhit x Ncand
    best_v = -1.0
    bi = 0
    bthc = 30.0
    opening_grid = np.arange(14.0, cap + 0.1, 4.0)
    step = 4.0
    sigma2 = float(_SIGMA_THETA) ** 2
    kernel = np.exp(-0.5 * ((angc - float(opening_grid[0])) / _SIGMA_THETA) ** 2)
    # For a uniform opening grid, update the same Gaussian shell exactly by
    # its analytic ratio instead of evaluating a fresh exponential for every
    # opening.  Candidate and opening traversal order remains unchanged.
    growth = np.exp(angc * step / sigma2)
    for grid_index, thc0 in enumerate(opening_grid):
        v = q @ kernel
        i = int(np.argmax(v))
        if v[i] > best_v:
            best_v = v[i]
            bi = i
            bthc = thc0
        if grid_index + 1 < len(opening_grid):
            from wctering.seeding.cone_moments import advance_kernel

            factor = math.exp(-float(thc0) * step / sigma2 - step**2 / (2.0 * sigma2))
            advance_kernel(kernel, growth, factor)
    ax = _CAND[bi]
    a = angc[:, bi]
    on = np.abs(a - bthc) < 14.0
    thc = float(np.clip(np.average(a[on], weights=q[on]) if q[on].sum() > 0 else bthc, 8.0, cap))
    return ax, thc


def _best_localized_cone(uh, q, cap=_THETA_C_MAX):
    return _cached_call("best_localized_cone", _best_localized_cone_uncached, uh, q, cap)


def _best_coherence_cone_uncached(uh, q, sig=_COH_SIGMA, cap=_THETA_C_MAX):
    """Find a cone (axis, theta_c) maximizing a TIGHTNESS-weighted charge vote.

    Where _best_localized_cone maximizes the charge in a shell, this divides the
    shell charge by sqrt(1 + radial variance), so a faint but TIGHT, coherent ring
    outscores a brighter but diffuse blob or spread-out noise. That is exactly what
    lets it lock onto a weak second ring buried under a dominant ring's outer tail
    and heavy noise -- the case (e.g. 500 MeV ev8) where the charge-only vote is won
    by the dominant tail and the faint ring's axis is never found.
    """
    uh = np.asarray(uh, float)
    q = np.asarray(q, float)
    angc = _cone_angle_matrix(uh)  # Nhit x Ncand
    best = -1.0
    bi = 0
    bthc = 30.0
    opening_grid = np.arange(16.0, cap + 0.1, 3.0)
    step = 3.0
    sigma2 = float(sig) ** 2
    kernel = np.exp(-0.5 * ((angc - float(opening_grid[0])) / float(sig)) ** 2)
    growth = np.exp(angc * step / sigma2)
    for grid_index, thc0 in enumerate(opening_grid):
        # Accumulate the same two sums in one pass over hits. This avoids
        # materializing weight, residual and squared-residual matrices.
        from wctering.seeding.cone_moments import shell_moments

        shell_charge, squared_residual = shell_moments(angc, q, kernel, thc0)
        sw = shell_charge + 1e-9
        var = squared_residual / sw
        score = shell_charge / np.sqrt(1.0 + var)  # reward tight (low-variance) shells
        i = int(np.argmax(score))
        if score[i] > best:
            best = score[i]
            bi = i
            bthc = thc0
        if grid_index + 1 < len(opening_grid):
            from wctering.seeding.cone_moments import advance_kernel

            factor = math.exp(-float(thc0) * step / sigma2 - step**2 / (2.0 * sigma2))
            advance_kernel(kernel, growth, factor)
    return _CAND[bi], bthc


def _best_coherence_cone(uh, q, sig=_COH_SIGMA, cap=_THETA_C_MAX):
    return _cached_call("best_coherence_cone", _best_coherence_cone_uncached, uh, q, sig, cap)
