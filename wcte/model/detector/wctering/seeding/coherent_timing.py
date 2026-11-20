"""Low-rank Fermi-Eyges timing residual model for a fitted Cherenkov axis.

This is a detector-only hypothesis diagnostic, not a particle-track fitter.
The prior is integrated angular Brownian motion in each transverse direction.
Its amplitude and a robust electronic width are profiled on training modules.
Withheld-module residuals assess predictive adequacy of that coherent path.
"""

import math
import numpy as np
from wctering.seeding.folds import fold_ids_for_hits

C = 299.792458
N_PHASE = 1.344
N_GROUP = 1.388


def _unit(v):
    a = np.asarray(v, float)
    return a / max(float(np.linalg.norm(a)), 1e-12)


def _surface(positions, ring):
    v = np.asarray(ring["vertex"], float)
    d = _unit(ring["direction"])
    theta = math.radians(np.clip(float(ring["theta_c"]), 15, math.degrees(math.acos(1 / N_PHASE))))
    beta = 1 / (N_PHASE * math.cos(theta))
    dx = positions - v
    a = dx @ d
    transverse = dx - a[:, None] * d
    b = np.linalg.norm(transverse, axis=1)
    s = a - b / math.tan(theta)
    ray = dx - s[:, None] * d
    dist = np.linalg.norm(ray, axis=1)
    time = (s / beta + N_GROUP * dist) / C
    # An origin shift along the fitted line changes only the free clock.
    origin = float(np.quantile(s, 0.02))
    length = max(float(np.quantile(s, 0.98)) - origin, 100)
    u = np.clip((s - origin) / length, 0, 1.5)
    ref = np.eye(3)[np.argmin(np.abs(d))]
    e1 = _unit(np.cross(d, ref))
    e2 = np.cross(d, e1)
    rays = ray / np.maximum(dist[:, None], 1e-9)
    # Quadrature of integral_0^s (s-r) sqrt(T) dW(r). Coefficients have
    # unit-normal priors; end-angle RMS is the profiled amplitude below.
    knots = (np.arange(8) + 0.5) / 8
    integral = np.maximum(u[:, None] - knots, 0) * length / math.sqrt(8)
    basis = np.concatenate([integral * (rays @ e1)[:, None], integral * (rays @ e2)[:, None]], axis=1) * (
        -N_GROUP / C
    )
    return time, basis, s, length


def _fit_predict(train_x, train_y, test_x, angle_rms, noise):
    # Gaussian prior for coherent displacement and a diffuse common clock.
    a = np.column_stack([np.ones(len(train_y)) * 20, train_x * angle_rms])
    b = np.column_stack([np.ones(len(test_x)) * 20, test_x * angle_rms])
    weights = np.ones(len(train_y)) / (noise * noise)
    coef = np.zeros(a.shape[1])
    for _ in range(3):
        precision = np.eye(a.shape[1]) + a.T @ (weights[:, None] * a)
        coef = np.linalg.solve(precision, a.T @ (weights * train_y))
        residual = (train_y - a @ coef) / noise
        weights = np.minimum(1, 2 / np.maximum(np.abs(residual), 1e-12)) / (noise * noise)
    cov = np.linalg.inv(precision)
    mean = b @ coef
    variance = noise * noise + np.einsum("ij,jk,ik->i", b, cov, b)
    penalty = float(coef[1:] @ coef[1:])
    training_loss = float(np.sum(3 * np.log1p(((train_y - a @ coef) / noise) ** 2 / 5)) + penalty / 2)
    return mean, variance, training_loss, penalty


def coherent_timing_features(pmt_ids, positions, charges, times, diagnostics):
    """Predict held-out module times with geometries fitted to other modules.

    Student-t held-out scoring is a robust diagnostic; its score difference is
    not a chi-square statistic or an analytically calibrated p-value.
    """
    ids = np.asarray(pmt_ids)
    pos = np.asarray(positions, float)
    t = np.asarray(times, float)
    if len(ids) < 30:
        return {}
    folds = fold_ids_for_hits(ids)
    fold_scores = []
    straight_scores = []
    angles = []
    lengths = []
    for item in diagnostics.get("module_fold_refits", []):
        held = int(item["withheld_fold"])
        train = folds != held
        test = ~train
        h = item.get("hypotheses", {})
        ring = h.get("h1_partition_ring")
        if ring is None:
            fit = h.get("h1_fit", {})
            cones = h.get("partition_rings", [])
            if "information_weak_axis" in fit and cones:
                dominant = max(cones, key=lambda c: float(c.get("charge", 0)))
                direction = _unit(fit["information_weak_axis"])
                if direction @ _unit(dominant["direction"]) < 0:
                    direction = -direction
                ring = {
                    "vertex": fit["vertex_mm"],
                    "direction": direction,
                    "theta_c": dominant["theta_c"],
                }
        if ring is None or np.count_nonzero(train) < 20 or np.count_nonzero(test) < 5:
            continue
        base, basis, s, length = _surface(pos, ring)
        residual = t - base
        center = float(np.median(residual[train]))
        residual -= center
        noise = float(
            np.clip(
                1.4826 * np.median(np.abs(residual[train] - np.median(residual[train]))),
                0.7,
                3.0,
            )
        )
        candidates = []
        for amplitude in [0, 0.03, 0.10, 0.30]:
            mean, var, loss, penalty = _fit_predict(
                basis[train], residual[train], basis[test], amplitude, noise
            )
            score = float(np.mean(0.5 * np.log(var) + 3 * np.log1p((residual[test] - mean) ** 2 / (5 * var))))
            candidates.append((loss, score, amplitude, penalty))
        selected = min(candidates, key=lambda x: x[0])
        straight_scores.append(candidates[0][1])
        fold_scores.append(selected[1])
        angles.append(selected[2])
        lengths.append(length)
    if not fold_scores:
        return {}
    return {
        "coherent_timing.cv_nll": float(np.mean(fold_scores)),
        "coherent_timing.straight_cv_nll": float(np.mean(straight_scores)),
        "coherent_timing.cv_gain": float(np.mean(np.array(straight_scores) - fold_scores)),
        "coherent_timing.cv_score_spread": float(np.std(fold_scores)),
        "coherent_timing.profile_angle_rms_mean": float(np.mean(angles)),
        "coherent_timing.longitudinal_span_mm": float(np.mean(lengths)),
    }
