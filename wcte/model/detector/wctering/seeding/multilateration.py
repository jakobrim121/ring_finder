"""Charge-weighted timing vertices and their information axes.

Small events use the reference SciPy solve. Larger events use the same bounded
Levenberg–Marquardt arithmetic in C (or NumPy if compilation is unavailable).
The 200-hit switch is part of the frozen counter; it is not retuned here.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares
from wctering.seeding import native_point
from wctering.seeding.timing_precision import precision_weights
from wctering.seeding.geometry import (
    ENFORCE_WCTE_VERTEX_CONTAINMENT,
    contained_point_source_refit,
    wcte_contains_vertex,
    wcte_vertex_margin_mm,
)

FAST_MIN_HITS = int(os.environ.get("WCTE_ANISOTROPY_FAST_LM_MIN_N", "200"))


def weighted_median(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if len(finite) else np.nan

    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if len(finite) else np.nan

    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, 0.5)])


def initial_guess(p: np.ndarray, t: np.ndarray, w: np.ndarray, vc: float) -> np.ndarray:
    """Charge-weighted starting point matching the original scipy path."""
    loc0 = np.average(p, axis=0, weights=w)
    eps0 = weighted_median(t - np.linalg.norm(p - loc0[None, :], axis=1) / vc, w)
    return np.array([loc0[0], loc0[1], loc0[2], eps0], dtype=float)


def solve_numpy(
    p: np.ndarray,
    t: np.ndarray,
    sqrtw: np.ndarray,
    vc: float,
    x0: np.ndarray,
    max_nfev: int,
):
    """Pure-NumPy fallback for environments without the native-C backend."""
    lower = np.array([-5000.0, -5000.0, -5000.0, -1000.0], dtype=float)
    upper = np.array([5000.0, 5000.0, 5000.0, 1000.0], dtype=float)
    x = np.minimum(np.maximum(np.asarray(x0, dtype=float), lower + 1e-6), upper - 1e-6)

    def cost_h_g(xx):
        vecs = p - xx[:3][None, :]
        d = np.maximum(np.linalg.norm(vecs, axis=1), 1e-12)
        r = sqrtw * (t - xx[3] - d / vc)
        J = np.column_stack((sqrtw[:, None] * vecs / d[:, None] / vc, -sqrtw))
        return float(np.sum(r * r)), J.T @ J, J.T @ r

    cost, H, g = cost_h_g(x)
    lam = 1e-3
    n_eval = 1
    for _ in range(int(max_nfev)):
        if np.max(np.abs(g)) < 2e-6:
            break
        old_cost = cost
        accepted = False
        step = np.zeros(4, dtype=float)
        for _trial in range(12):
            H_lm = H.copy()
            diag = np.maximum(np.diag(H_lm), 1e-12)
            H_lm.flat[::5] += lam * diag
            try:
                step = np.linalg.solve(H_lm, -g)
            except np.linalg.LinAlgError:
                H_lm.flat[::5] += (lam + 1.0) * 10.0
                step = np.linalg.solve(H_lm, -g)
            xt = np.minimum(np.maximum(x + step, lower + 1e-6), upper - 1e-6)
            trial_cost, _, _ = cost_h_g(xt)
            n_eval += 1
            if trial_cost <= cost:
                x = xt
                cost, H, g = cost_h_g(x)
                lam = max(lam * 0.33, 1e-12)
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if np.linalg.norm(step) < 2e-6 * (2e-6 + np.linalg.norm(x)):
            break
        if abs(old_cost - cost) < 2e-6 * max(old_cost, 1.0):
            break
        if not accepted:
            break
    return x, cost, H, n_eval, True


def solve_bounded_lm(*args):
    if native_point.LIBRARY is not None:
        try:
            return native_point.solve(*args)
        except Exception:
            pass
    return solve_numpy(*args)


def backend_label():
    return "fast_bounded_lm_" + ("native_c" if native_point.LIBRARY is not None else "numpy")


class PointSourceFitter:
    """Charge-weighted point-source timing fit used by the counter."""

    def __init__(
        self,
        pmt_locs: np.ndarray,
        hit_times: np.ndarray,
        hit_charges: np.ndarray,
        pmt_ids: np.ndarray,
        *,
        refraction_index: float = 1.333,
        group_index: float | None = None,
        beta: float = 1.0,
        sigma_t_ns: float = 1.0,
        charge_weight_power: float = 1.0,
    ) -> None:
        input_positions = np.asarray(pmt_locs, dtype=float)
        input_times = np.asarray(hit_times, dtype=float)
        input_charges = np.asarray(hit_charges, dtype=float)
        input_ids = np.asarray(pmt_ids, dtype=int)
        self.input_N = int(len(input_ids))
        good = (
            np.isfinite(input_positions).all(axis=1)
            & np.isfinite(input_times)
            & np.isfinite(input_charges)
            & np.isfinite(input_ids)
        )
        self.pmt_locs = input_positions[good]
        self.hit_times = input_times[good]
        self.hit_charges = input_charges[good]
        self.pmt_ids = input_ids[good]
        if self.pmt_locs.ndim != 2 or self.pmt_locs.shape[1] != 3:
            raise ValueError("pmt_locs must have shape (N, 3)")
        if not (len(self.pmt_locs) == len(self.hit_times) == len(self.hit_charges) == len(self.pmt_ids)):
            raise ValueError("Input arrays must all have the same length")
        self.c_light = 299.792458
        self.n = float(refraction_index)
        self.n_group = float(group_index) if group_index is not None else float(refraction_index)
        self.vc = self.c_light / self.n_group
        self.beta = float(beta)
        self.v_particle = self.beta * self.c_light
        self.sigma_t_ns = float(sigma_t_ns)
        self.charge_weight_power = float(charge_weight_power)
        self._point_fit_cache: Dict[Tuple, Dict] = {}

    def _scipy_fit(
        self,
        hit_mask: np.ndarray,
        guess: Optional[np.ndarray] = None,
        max_nfev: int = 35,
        *,
        position_bound_mm: Optional[float] = None,
        time_bound_ns: Optional[float] = None,
    ) -> Dict:
        hit_mask = np.asarray(hit_mask, dtype=bool)
        cache_key = None
        if guess is None:
            cache_key = (
                int(max_nfev),
                None if position_bound_mm is None else float(position_bound_mm),
                None if time_bound_ns is None else float(time_bound_ns),
                hit_mask.tobytes(),
            )
            cached = self._point_fit_cache.get(cache_key)
            if cached is not None:
                return cached
        p = self.pmt_locs[hit_mask]
        t = self.hit_times[hit_mask]
        q = self.hit_charges[hit_mask]
        n_hits = len(t)
        if n_hits < 6:
            return {"valid": False, "reason": "too few hits"}

        w = precision_weights(
            q,
            self.charge_weight_power,
            self.sigma_t_ns,
        )
        sqrtw = np.sqrt(w)

        if guess is None:
            loc0 = np.average(p, axis=0, weights=w)
            eps0 = weighted_median(t - np.linalg.norm(p - loc0[None, :], axis=1) / self.vc, w)
            guess = np.array([loc0[0], loc0[1], loc0[2], eps0], dtype=float)

        position_bound = 5000.0 if position_bound_mm is None else float(position_bound_mm)
        time_bound = 1000.0 if time_bound_ns is None else float(time_bound_ns)
        if not np.isfinite(position_bound) or position_bound <= 0.0:
            raise ValueError("position_bound_mm must be finite and positive")
        if not np.isfinite(time_bound) or time_bound <= 0.0:
            raise ValueError("time_bound_ns must be finite and positive")
        lower = np.array([-position_bound] * 3 + [-time_bound])
        upper = np.array([position_bound] * 3 + [time_bound])
        guess = np.minimum(np.maximum(guess, lower + 1e-6), upper - 1e-6)

        def rho(pars: np.ndarray) -> np.ndarray:
            loc = pars[:3]
            eps = pars[3]
            dists = np.linalg.norm(p - loc[None, :], axis=1)
            return sqrtw * (t - eps - dists / self.vc)

        def jac(pars: np.ndarray) -> np.ndarray:
            loc = pars[:3]
            vecs = p - loc[None, :]
            dists = np.maximum(np.linalg.norm(vecs, axis=1), 1e-12)
            jac_xyz = sqrtw[:, None] * vecs / dists[:, None] / self.vc
            jac_eps = -sqrtw
            return np.column_stack([jac_xyz, jac_eps])

        try:
            res = least_squares(
                rho,
                guess,
                jac=jac,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                xtol=2e-6,
                ftol=2e-6,
                gtol=2e-6,
            )
            pars = np.asarray(res.x, dtype=float)
            solver = "scipy_reference"
            containment_refit = None
            if ENFORCE_WCTE_VERTEX_CONTAINMENT and not wcte_contains_vertex(pars[:3]):
                containment_refit = contained_point_source_refit(
                    p,
                    t,
                    sqrtw,
                    self.vc,
                    pars,
                    time_bound_ns=time_bound,
                    max_iterations=max(300, int(max_nfev)),
                )
                if not bool(containment_refit.get("valid", False)):
                    return {
                        "valid": False,
                        "reason": "no contained WCTE point-source solution",
                        "unconstrained_vertex": pars[:3],
                        "containment_refit": containment_refit,
                    }
                pars = np.asarray(containment_refit["parameters"], dtype=float)
                solver = "scipy_contained_slsqp"
            J = jac(pars)
            cov = np.linalg.pinv(J.T @ J, rcond=1e-12)
            pos_cov = 0.5 * (cov[:3, :3] + cov[:3, :3].T)
            eigvals, eigvecs = np.linalg.eigh(pos_cov)
            eigvals = np.maximum(eigvals, 0.0)
            major = eigvecs[:, -1]
            major = major / np.linalg.norm(major)
            anis = float(np.sqrt(eigvals[-1]) / np.sqrt(np.sum(eigvals))) if np.sum(eigvals) > 0 else 0.0
            pulls = rho(pars)
            chi2 = float(np.sum(pulls**2))
            ndof = n_hits - 4
            out = {
                "valid": True,
                "success": bool(
                    res.success if containment_refit is None else containment_refit.get("success", False)
                ),
                "pars": pars,
                "vertex": pars[:3],
                "eps": float(pars[3]),
                "cov": cov,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "major_axis": major,
                "anisotropy": anis,
                "chi2": chi2,
                "ndof": ndof,
                "red_chi2": chi2 / max(ndof, 1),
                "n_hits": n_hits,
                "result": res,
                "solver": solver,
                "vertex_containment": {
                    "enforced": bool(ENFORCE_WCTE_VERTEX_CONTAINMENT),
                    "was_constrained": bool(containment_refit is not None),
                    "margin_mm": float(wcte_vertex_margin_mm(pars[:3])),
                    "unconstrained_vertex_mm": (
                        None if containment_refit is None else np.asarray(res.x[:3], dtype=float)
                    ),
                },
            }
            if cache_key is not None:
                self._point_fit_cache[cache_key] = out
            return out
        except Exception as exc:
            return {"valid": False, "reason": repr(exc)}

    def point_source_fit(
        self,
        hit_mask: np.ndarray,
        guess: Optional[np.ndarray] = None,
        max_nfev: int = 35,
        *,
        position_bound_mm: Optional[float] = None,
        time_bound_ns: Optional[float] = None,
    ) -> Dict:
        """Fast bounded LM replacement for PointSourceFitter.point_source_fit."""
        # Exact-path fallback for small/sparse events. These are already fast with
        # scipy, and their downstream pair ranking can be sensitive to tiny solver
        # drift because there are few hits per ring.
        if (
            int(getattr(self, "input_N", 0)) < FAST_MIN_HITS
            or position_bound_mm is not None
            or time_bound_ns is not None
        ):
            return self._scipy_fit(
                hit_mask,
                guess=guess,
                max_nfev=max_nfev,
                position_bound_mm=position_bound_mm,
                time_bound_ns=time_bound_ns,
            )
        hit_mask = np.asarray(hit_mask, dtype=bool)
        cache_key = None
        if guess is None:
            cache_key = ("fast_lm", int(max_nfev), hit_mask.tobytes())
            cached = self._point_fit_cache.get(cache_key)
            if cached is not None:
                return cached

        p = self.pmt_locs[hit_mask]
        t = self.hit_times[hit_mask]
        q = self.hit_charges[hit_mask]
        n_hits = len(t)
        if n_hits < 6:
            return {"valid": False, "reason": "too few hits"}

        w = precision_weights(
            q,
            self.charge_weight_power,
            self.sigma_t_ns,
        )
        sqrtw = np.sqrt(w)
        if guess is None:
            guess = initial_guess(p, t, w, self.vc)
        lower = np.array([-5000.0, -5000.0, -5000.0, -1000.0], dtype=float)
        upper = np.array([5000.0, 5000.0, 5000.0, 1000.0], dtype=float)
        guess = np.minimum(np.maximum(np.asarray(guess, dtype=float), lower + 1e-6), upper - 1e-6)

        try:
            pars, chi2, H, n_eval, success = solve_bounded_lm(
                p, t, sqrtw, float(self.vc), guess, int(max_nfev)
            )
            pars = np.asarray(pars, dtype=float)
            containment_refit = None
            if ENFORCE_WCTE_VERTEX_CONTAINMENT and not wcte_contains_vertex(pars[:3]):
                unconstrained_vertex = pars[:3].copy()
                containment_refit = contained_point_source_refit(
                    p,
                    t,
                    sqrtw,
                    self.vc,
                    pars,
                    time_bound_ns=1000.0,
                    max_iterations=max(300, int(max_nfev)),
                )
                if not bool(containment_refit.get("valid", False)):
                    return {
                        "valid": False,
                        "reason": "no contained WCTE point-source solution",
                        "unconstrained_vertex": unconstrained_vertex,
                        "containment_refit": containment_refit,
                    }
                pars = np.asarray(containment_refit["parameters"], dtype=float)
                chi2 = float(containment_refit["chi2"])
                H = np.asarray(containment_refit["normal_matrix"], dtype=float)
                success = bool(containment_refit.get("success", False))
            else:
                unconstrained_vertex = None
            cov = np.linalg.pinv(H, rcond=1e-12)
            pos_cov = 0.5 * (cov[:3, :3] + cov[:3, :3].T)
            eigvals, eigvecs = np.linalg.eigh(pos_cov)
            eigvals = np.maximum(eigvals, 0.0)
            major = eigvecs[:, -1]
            norm_major = np.linalg.norm(major)
            if norm_major > 0:
                major = major / norm_major
            anis = float(np.sqrt(eigvals[-1]) / np.sqrt(np.sum(eigvals))) if np.sum(eigvals) > 0 else 0.0
            ndof = n_hits - 4
            out = {
                "valid": True,
                "success": bool(success),
                "pars": pars,
                "vertex": np.asarray(pars[:3], dtype=float),
                "eps": float(pars[3]),
                "cov": cov,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "major_axis": major,
                "anisotropy": anis,
                "chi2": float(chi2),
                "ndof": ndof,
                "red_chi2": float(chi2) / max(ndof, 1),
                "n_hits": n_hits,
                "result": None,
                "solver": (
                    backend_label() if containment_refit is None else backend_label() + "_contained_slsqp"
                ),
                "n_eval": int(n_eval),
                "vertex_containment": {
                    "enforced": bool(ENFORCE_WCTE_VERTEX_CONTAINMENT),
                    "was_constrained": bool(containment_refit is not None),
                    "margin_mm": float(wcte_vertex_margin_mm(pars[:3])),
                    "unconstrained_vertex_mm": unconstrained_vertex,
                },
            }
            if cache_key is not None:
                self._point_fit_cache[cache_key] = out
            return out
        except Exception as exc:
            # Production-safety fallback: keep correctness if the fast solve hits a
            # numerical corner case.  The cache key remains distinct so a fallback
            # result is reused exactly for the same label mask.
            fallback = self._scipy_fit(
                hit_mask,
                guess=guess,
                max_nfev=max_nfev,
                position_bound_mm=position_bound_mm,
                time_bound_ns=time_bound_ns,
            )
            if cache_key is not None:
                self._point_fit_cache[cache_key] = fallback
            return fallback
