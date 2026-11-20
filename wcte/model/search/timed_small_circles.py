"""Seed short light cones from three independent, time-compatible PMTs."""

import numpy as np

from bare_residual_search import residual_scores
from charge_starts import weighted_quantile
from emission_window import coherent_interval
from finite_residual_fit import optical_shape
from residual_axes import timed_residual_vertices
from residual_timing_vertices import clock_modes
from sparse_paths import positions_for_search
from wctering.seeding.coherent_timing import C, N_GROUP


def circles(rays, indices):
    """Find the small circle through each triplet of rays from one vertex."""
    r = rays[indices]
    cross = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0])
    area = np.linalg.norm(cross, axis=1)
    axis = cross / np.maximum(area[:, None], 1e-20)
    dot = np.einsum("ij,ij->i", axis, r[:, 0])
    axis *= np.where(dot < 0.0, -1.0, 1.0)[:, None]
    angle = np.arccos(np.clip(abs(dot), 0.0, 1.0))
    return axis, angle, area


def sample_axes(rays, weights, modules, random):
    """Favor excess prompt charge while keeping separate module footprints."""
    if np.count_nonzero(weights > 0.05 * weights.max()) < 12 or weights.sum() < 9.0:
        return np.empty((0, 3)), np.empty(0)
    module = np.bincount(modules, weights=weights)
    weight = weights / np.sqrt(np.maximum(module[modules], 0.5))
    triples = random.choice(len(weights), size=(512, 3), p=weight / weight.sum())
    separate = np.all(np.diff(np.sort(modules[triples], axis=1), axis=1) != 0, axis=1)
    axis, angle, area = circles(rays, triples[separate])
    keep = (area > 0.035) & (angle >= np.deg2rad(16.0)) & (angle <= np.deg2rad(42.4))
    return axis[keep], angle[keep]


def starts(event, model, bank, geometry, shapes, amp, *, marked_residual=False):
    """Use timing for sampling and calibrated all-PMT charge for ranking."""
    mean = amp @ shapes
    _, residual = model.obs.score(mean, model.ix)

    def score_residual(basis):
        return residual_scores(basis, mean, residual, shapes)

    t = event.times_ns - np.median(event.times_ns)
    charge = np.minimum(event.charges_pe, 2.0)
    unexplained = charge * np.clip(-residual[model.ix] * mean[model.ix], 0.05, 2.0)
    marked = None
    if marked_residual and event.timing_selection["prompt_window_ns"] is not None:
        from marked_residual import MarkedResidual

        try:
            marked = MarkedResidual(event, model, geometry, shapes, amp)
        except (ValueError, ArithmeticError):
            pass
    modules = np.unique(event.pmt_ids // 100, return_inverse=True)[1]
    points = positions_for_search(event, bank, geometry)
    points.extend(
        p
        for p in timed_residual_vertices(event, mean[model.ix], limit=3)
        if all(np.linalg.norm(p - v) > 0.16 for v in points)
    )
    random = np.random.default_rng(271828)
    candidates = []
    for point in points:
        delta = model.pos - 1000.0 * point
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        rays = delta / distance[:, None]
        corrected = t - N_GROUP * distance[model.ix] / C
        exposure = np.maximum(-np.einsum("ij,ij->i", rays, model.normals), 0.02) / (distance**2 + 10000.0)
        for clock in clock_modes(corrected, unexplained):
            in_time = np.exp(-0.5 * ((corrected - clock) / 1.8) ** 2)
            weight = unexplained * in_time
            axis, angle = sample_axes(rays[model.ix], weight, modules, random)
            if not len(angle):
                continue
            polar = np.arccos(np.clip(axis @ rays.T, -1.0, 1.0))
            gaussian = np.exp(-0.5 * ((polar - angle[:, None]) / np.deg2rad(3.5)) ** 2)
            basis = gaussian * exposure
            basis /= np.maximum(basis.sum(1, keepdims=True), 1e-30)
            if marked is None:
                score, fisher, _ = score_residual(basis)
                statistic = np.maximum(score, 0.0) ** 2 / np.maximum(fisher, 1e-20)
            else:
                # Point-source timing is adequate for choosing local
                # axes. The next pass uses each finite track's flight.
                statistic = 2.0 * marked(basis, clock + N_GROUP * distance[model.ix] / C)[0]
            best = []
            for j in np.argsort(statistic)[::-1]:
                if statistic[j] < 14.0:
                    break
                if any(axis[j] @ axis[k] > 0.965 and abs(angle[j] - angle[k]) < 0.1 for k in best):
                    continue
                best.append(j)
                support = gaussian[j, model.ix] * (charge + weight)
                interval = coherent_interval(event, point, axis[j], angle[j], support)
                if interval is None:
                    continue
                path, time, _ = interval
                path[-1] = np.log(np.deg2rad(2.6))
                shape = optical_shape(model, path, 4)[0]
                overlap = np.exp(-0.5 * ((time - t) / 2.0) ** 2)
                direct = overlap * shape[model.ix] * charge / np.maximum(mean[model.ix], 0.4)
                if direct.sum() <= 1e-15:
                    continue
                own = direct >= 0.12 * direct.max()
                if own.sum() < 18 or len(np.unique(modules[own])) < 3:
                    continue
                if marked is None:
                    local, information, _ = score_residual(shape[None])
                    if local[0] <= 0.0 or information[0] <= 1e-20:
                        continue
                    photons = np.clip(local[0] / information[0], 1.0, 0.4 * amp.sum())
                    delay = weighted_quantile(t - time, direct, [0.5])[0]
                    ranking = float(local[0] * photons) * np.exp(-0.5 * (delay / 2.0) ** 2)
                else:
                    ranking, photons = (values[0] for values in marked(shape[None], time))
                    if ranking < 7.0 or photons < 10.0:
                        continue
                    photons = np.clip(photons, 1.0, 0.4 * amp.sum())
                candidates.append((ranking, path, photons))
                if len(best) >= 3:
                    break
    unique = []
    for rank, path, photons in sorted(candidates, key=lambda row: row[0], reverse=True):
        if any(np.linalg.norm(p[:3] - path[:3]) < 0.16 and p[3:6] @ path[3:6] > 0.96 for p in unique):
            continue
        unique.append(path)
        yield path, photons
        if len(unique) == 3:
            break
