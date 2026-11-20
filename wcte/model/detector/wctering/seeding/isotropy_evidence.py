"""Weighted SNO-style angular moments and their time coherence.

The spherical-harmonic addition theorem evaluates pairwise Legendre moments
without an N-hit squared matrix. Only rotationally invariant contractions are
exported. Self-pairs are removed exactly. Acceptance correction and charge
weighting are extensions tested for WCTE, not the original SNO beta14 model.
"""

import math
import numpy as np
from wctering.seeding.arc_evidence import _active_geometry, _view, _quantile


def harmonic_basis(ray, lmax=8):
    """Complex Y_lm for m>=0; negative m is recovered by conjugation."""
    x = np.clip(ray[:, 2], -1, 1)
    phi = np.arctan2(ray[:, 1], ray[:, 0])
    sin = np.sqrt(np.maximum(1 - x * x, 0))
    out = {}
    pmm = np.ones(len(ray))
    for m in range(lmax + 1):
        if m:
            pmm = -(2 * m - 1) * sin * pmm
        pprev = None
        p = pmm.copy()
        phase = np.exp(1j * m * phi)
        for l in range(m, lmax + 1):
            if l == m:
                current = p
            elif l == m + 1:
                current = x * (2 * m + 1) * p
            else:
                current = ((2 * l - 1) * x * p - (l + m - 1) * pprev) / (l - m)
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
            out[l, m] = norm * current * phase
            if l > m:
                pprev, p = p, current
    return out


def _moments_from_basis(basis, a, b, lmax):
    diagonal = float(a @ b)
    den = float(a.sum() * b.sum() - diagonal)
    if den <= 1e-12:
        return np.full(lmax, np.nan)
    values = []
    for l in range(1, lmax + 1):
        total = 0.0
        for m in range(l + 1):
            aa = a @ basis[l, m]
            bb = b @ basis[l, m]
            total += (1 if m == 0 else 2) * float((aa * bb.conjugate()).real)
        values.append((4 * math.pi / (2 * l + 1) * total - diagonal) / den)
    return np.asarray(values)


def isotropy_evidence_features(pmt_ids, positions, charges, times, diagnostics):
    geom = _active_geometry()
    idx = np.asarray([geom.id_to_index[int(i)] for i in pmt_ids], int)
    if len(idx) < 2:
        return {}
    w = np.clip(np.asarray(charges, float), 0, 2)
    q = np.zeros(len(geom.pmt_ids))
    np.add.at(q, idx, w)
    if q.sum() <= 0:
        return {}
    h = diagnostics.get("full_event_hypotheses") or {}
    point = h.get("h1_fit") or {}
    cone = h.get("h1_partition_ring") or {}
    views = {}
    if point.get("vertex_mm") is not None:
        views["point"] = np.asarray(point["vertex_mm"], float)
    if cone.get("vertex") is not None:
        views["cone"] = np.asarray(cone["vertex"], float)
    out = {}
    for name, v in views.items():
        ray, _, exposure, dist = _view(geom, v, np.array([0.0, 0.0, 1.0]))
        basis = harmonic_basis(ray, 8)
        residual = np.asarray(times, float) - 1.388 * dist[idx] / 299.792458
        median = float(_quantile(residual, w, [0.5])[0])
        early = np.zeros(len(q))
        late = np.zeros(len(q))
        timefield = np.zeros(len(q))
        np.add.at(early, idx, w * (residual <= median))
        np.add.at(late, idx, w * (residual > median))
        np.add.at(timefield, idx, w * np.clip(residual - median, -5, 5))
        typical = np.median(exposure[exposure > 0])
        corrected = q * typical / np.maximum(exposure, 0.1 * typical)
        fields = {
            "raw": (q, q),
            "exposure_corrected": (corrected, corrected),
            "early": (early, early),
            "late": (late, late),
            "early_late": (early, late),
        }
        for field, (a, b) in fields.items():
            values = _moments_from_basis(basis, a, b, 8)
            for l, value in enumerate(values, 1):
                if np.isfinite(value):
                    out[f"isotropy.{name}.{field}.beta{l}"] = float(value)
            if np.isfinite(values[[0, 3]]).all():
                out[f"isotropy.{name}.{field}.beta14"] = float(values[0] + 4 * values[3])
        # Correlated timing residual power; remove the common clock first.
        timefield -= q * timefield.sum() / q.sum()
        for l in range(1, 9):
            power = 0.0
            for m in range(l + 1):
                power += (1 if m == 0 else 2) * abs(timefield @ basis[l, m]) ** 2
            out[f"isotropy.{name}.time_power_l{l}"] = float(
                4 * math.pi / (2 * l + 1) * power / max(q.sum() ** 2, 1.0)
            )
    return out
