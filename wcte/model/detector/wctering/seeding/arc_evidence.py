"""Measure acceptance-corrected azimuth coverage and radial shape.

Use the geometry of each fitted cone to compare its reconstructed light
with the detector's acceptance. The optimized optical bank also uses the
fixed-template amplitude and deviance helpers in this module.
"""

from functools import lru_cache
from dataclasses import replace
import numpy as np
from wctering.seeding.detector import load_wcte_physical_geometry
from wctering.seeding.io import active_wcte_ids


@lru_cache(maxsize=1)
def _active_geometry():
    full = load_wcte_physical_geometry()
    active = active_wcte_ids()
    keep = np.asarray([int(i) in active for i in full.pmt_ids])
    ids = full.pmt_ids[keep]
    return replace(
        full,
        pmt_ids=ids,
        positions_mm=full.positions_mm[keep],
        inward_normals=full.inward_normals[keep],
        id_to_index={int(i): j for j, i in enumerate(ids)},
    )


def _unit(x):
    x = np.asarray(x, float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def _quantile(x, w, qs):
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    s = np.cumsum(w) - w / 2
    return (
        np.interp(np.asarray(qs) * np.sum(w), s, x) if len(x) and np.sum(w) > 0 else np.full(len(qs), np.nan)
    )


def _summarize(values, prefix, out):
    for k in sorted(set().union(*(v.keys() for v in values))):
        a = np.asarray([v.get(k, np.nan) for v in values])
        a = a[np.isfinite(a)]
        if len(a):
            for name, fn in [("min", np.min), ("mean", np.mean), ("max", np.max)]:
                out[prefix + "." + k + "." + name] = float(fn(a))


def _view(geom, v, axis):
    dx = geom.positions_mm - v
    dist = np.maximum(np.linalg.norm(dx, axis=1), 1.0)
    ray = dx / dist[:, None]
    theta = np.degrees(np.arccos(np.clip(ray @ axis, -1, 1)))
    exposure = np.maximum(-np.einsum("ij,ij->i", geom.inward_normals, ray), 0.0) / (dist * dist + 100.0**2)
    return ray, theta, exposure, dist


def _amplitudes(shapes, q, mask, iterations=18):
    # Independent Poisson/quasi-Poisson observations on a subset of modules:
    # the denominator retains the expected counts on unhit training PMTs.
    s = shapes[:, mask]
    y = q[mask]
    exposure = np.maximum(s.sum(axis=1), 1e-20)
    amp = np.full(len(s), max(y.sum(), 1.0) / len(s)) / exposure
    for _ in range(iterations):
        density = np.maximum(amp @ s, 1e-20)
        amp *= np.maximum((s * (y / density)).sum(axis=1), 1e-20) / exposure
    return amp


def _deviance(q, lam):
    lam = np.maximum(lam, 1e-20)
    return 2 * float(np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-20) / lam), 0.0) - q + lam))


def _morphology(geom, ring, idx, weights, times, assignment):
    axis = _unit(ring["direction"])
    v = np.asarray(ring["vertex"], float)
    ray, theta, exposure, dist = _view(geom, v, axis)
    residual = theta - float(ring["theta_c"])
    j = idx[assignment]
    q = weights[assignment]
    if not len(j) or q.sum() <= 0:
        return {}
    total = float(q.sum())
    delta = residual[j]
    qu = _quantile(delta, q, [0.1, 0.25, 0.5, 0.75, 0.9])
    mad = float(_quantile(abs(delta - qu[2]), q, [0.5])[0])
    width = float(np.clip(1.4826 * mad, 4.0, 25.0))
    shape = exposure * np.exp(-0.5 * (residual / width) ** 2)
    shape /= max(shape.sum(), 1e-30)
    ref = np.eye(3)[np.argmin(abs(axis))]
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)
    phi = np.arctan2(ray @ e2, ray @ e1)
    hm = np.exp(1j * phi[:, None] * np.arange(1, 7)[None, :])
    observed = (q[:, None] * hm[j]).sum(axis=0) / total
    expected = (shape[:, None] * hm).sum(axis=0)
    diff = observed - expected
    out = {
        "radial_median_deg": qu[2],
        "radial_iqr_deg": qu[3] - qu[1],
        "radial_80_width_deg": qu[4] - qu[0],
        "radial_asymmetry": (qu[4] + qu[0] - 2 * qu[2]) / max(qu[4] - qu[0], 1.0),
        "radial_mad_deg": mad,
        "assigned_charge_fraction": total / max(weights.sum(), 1.0),
        "assigned_unique_pmts": float(len(set(j.tolist()))),
    }
    for band in [5, 10, 20]:
        out[f"core_{band}_fraction"] = float(q[abs(delta) <= band].sum() / total)
    for m in range(6):
        out[f"azimuth_power_m{m + 1}"] = float(abs(diff[m]) ** 2)
    # A smooth displacement changes the first harmonic across polar radius;
    # independent ridges can leave incoherent inner and outer arcs.
    parts = []
    for sign in [-1, 1]:
        take = sign * delta >= 0
        alltake = sign * residual >= 0
        qq = q[take].sum()
        ee = shape[alltake].sum()
        parts.append(
            (q[take, None] * hm[j[take], :3]).sum(axis=0) / max(qq, 1.0)
            - (shape[alltake, None] * hm[alltake, :3]).sum(axis=0) / max(ee, 1e-20)
        )
    for m in range(3):
        out[f"inner_outer_coherence_m{m + 1}"] = float((parts[0][m] * parts[1][m].conjugate()).real)
    tr = np.asarray(times)[assignment] - 1.388 * dist[j] / 299.792458
    tqu = _quantile(tr, q, [0.1, 0.5, 0.9])
    center = tr - tqu[1]
    out["point_time_80_width_ns"] = float(tqu[2] - tqu[0])
    out["point_time_tail_asymmetry"] = float((tqu[2] + tqu[0] - 2 * tqu[1]) / max(tqu[2] - tqu[0], 0.2))
    for m in range(3):
        out[f"time_azimuth_m{m + 1}"] = float(abs((q * np.clip(center, -5, 5) * hm[j, m]).sum() / total))
    dx = delta - np.average(delta, weights=q)
    tt = center - np.average(center, weights=q)
    out["angle_time_correlation"] = float(
        np.sum(q * dx * tt) / max(np.sqrt(np.sum(q * dx * dx) * np.sum(q * tt * tt)), 1e-10)
    )
    return out


def arc_evidence_features(
    pmt_ids,
    positions,
    charges,
    times,
    diagnostics,
    extra_h3=None,
):
    geom = _active_geometry()
    idx = np.asarray([geom.id_to_index[int(i)] for i in pmt_ids], int)
    w = np.clip(np.asarray(charges, float), 0, 2)
    q = np.zeros(len(geom.pmt_ids))
    np.add.at(q, idx, w)
    if not len(idx) or q.sum() <= 0:
        return {}
    h12 = diagnostics.get("full_event_hypotheses") or {}
    h3 = extra_h3 or (diagnostics.get("three_ring_hypothesis_test") or {}).get("full_event_hypotheses") or {}
    hypotheses = {}
    h1 = h12.get("h1_partition_ring")
    if h1:
        hypotheses[1] = ([h1], np.ones(len(idx), int))
    for k, h in [(2, h12), (3, h3)]:
        rings = h.get("partition_rings", [])
        labels = np.asarray(h.get("partition_labels", []), int)
        if len(rings) == k and labels.shape == (len(idx),) and set(labels) == set(range(1, k + 1)):
            hypotheses[k] = (rings, labels)
    out = {}
    for k, (rings, labels) in hypotheses.items():
        morph = [_morphology(geom, r, idx, w, times, labels == i + 1) for i, r in enumerate(rings)]
        _summarize(morph, f"arc.h{k}", out)
    if 2 in hypotheses and 3 in hypotheses:
        a = hypotheses[2][1] - 1
        b = hypotheses[3][1] - 1
        overlap = np.zeros((2, 3))
        np.add.at(overlap, (a, b), w)
        col = overlap / np.maximum(overlap.sum(axis=0, keepdims=True), 1e-10)
        row = overlap / np.maximum(overlap.sum(axis=1, keepdims=True), 1e-10)
        out["arc.h23.minimum_h3_parent_fraction"] = float(np.max(col, axis=0).min())
        out["arc.h23.mean_h3_parent_fraction"] = float(
            np.average(np.max(col, axis=0), weights=overlap.sum(axis=0))
        )
        out["arc.h23.maximum_h2_split_entropy"] = float(
            np.max(-np.sum(row * np.log(np.maximum(row, 1e-15)), axis=1))
        )
        out["arc.h23.minimum_h2_dominant_child_fraction"] = float(np.max(row, axis=1).min())
    return out
