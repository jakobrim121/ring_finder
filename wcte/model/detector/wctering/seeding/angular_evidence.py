"""Acceptance-corrected angular Fourier evidence for additional Cherenkov arcs.

An axially symmetric single emitter fixes a polar charge profile but has no
preferred azimuth. Detector exposure breaks that symmetry even for H1. The
expected azimuth moments are therefore subtracted using all active PMTs,
including unhit PMTs, before residual angular power is measured.
"""

import numpy as np
from wctering.seeding.detector import load_wcte_physical_geometry


def _unit(x):
    a = np.asarray(x, float)
    return a / max(float(np.linalg.norm(a)), 1e-12)


def angular_evidence_features(pmt_ids, positions, charges, times, diagnostics):
    full = diagnostics.get("full_event_hypotheses") or {}
    fit = full.get("h1_fit", {})
    rings = full.get("partition_rings", [])
    if not rings or "information_weak_axis" not in fit:
        return {}
    dominant = max(rings, key=lambda r: float(r.get("charge", 0)))
    axis = _unit(fit["information_weak_axis"])
    if axis @ _unit(dominant["direction"]) < 0:
        axis = -axis
    v = np.asarray(fit["vertex_mm"], float)
    geom = load_wcte_physical_geometry()
    all_pos = geom.positions_mm
    dx = all_pos - v
    distance = np.linalg.norm(dx, axis=1)
    ray = dx / np.maximum(distance[:, None], 1e-9)
    cos_theta = np.clip(ray @ axis, -1, 1)
    theta = np.arccos(cos_theta)
    ref = np.eye(3)[np.argmin(abs(axis))]
    e1 = _unit(np.cross(axis, ref))
    e2 = np.cross(axis, e1)
    phi = np.arctan2(ray @ e2, ray @ e1)
    incidence = np.clip(-np.einsum("ij,ij->i", geom.inward_normals, ray), 0.02, 1)
    exposure = incidence / np.maximum(distance, 100) ** 2
    index = np.array([geom.id_to_index[int(i)] for i in pmt_ids], int)
    signal = np.zeros(len(all_pos))
    np.add.at(signal, index, np.clip(np.asarray(charges, float), 0, 2))
    total = float(np.sum(signal))
    if total <= 0:
        return {}
    # Fixed polar bins are relative to the reconstructed axis, never the beam.
    bins = np.minimum(np.searchsorted(np.radians([10, 20, 30, 40, 50, 60, 80, 110, 140]), theta), 9)
    harmonics = np.exp(1j * phi[:, None] * np.arange(1, 7)[None, :])
    powers = []
    moments = []
    qbins = []
    for b in range(10):
        mask = bins == b
        q = float(np.sum(signal[mask]))
        e = float(np.sum(exposure[mask]))
        qbins.append(q)
        if e <= 0:
            moment = np.zeros(6, dtype=complex)
            power = np.zeros(6)
        else:
            expected = np.sum(exposure[mask, None] * harmonics[mask], axis=0) / e
            raw = np.sum(signal[mask, None] * harmonics[mask], axis=0) - q * expected
            variance = np.maximum(q * (1 - abs(expected) ** 2), 1)
            moment = raw / max(q, 1)
            power = abs(raw) ** 2 / variance
        moments.append(moment)
        powers.append(power)
    powers = np.asarray(powers)
    moments = np.asarray(moments)
    qbins = np.asarray(qbins)
    out = {}
    weights = qbins / max(total, 1)
    for m in range(6):
        out[f"angular.power_m{m + 1}"] = float(np.sum(powers[:, m]))
        out[f"angular.relative_power_m{m + 1}"] = float(np.sum(weights * abs(moments[:, m]) ** 2))
        out[f"angular.max_band_power_m{m + 1}"] = float(np.max(powers[:, m]))
        adjacent = moments[:-1, m] * moments[1:, m].conj()
        out[f"angular.radial_coherence_m{m + 1}"] = float(
            np.sum(np.sqrt(weights[:-1] * weights[1:]) * adjacent.real)
        )
    for b in range(10):
        out[f"angular.polar_charge_fraction_b{b}"] = float(weights[b])
        out[f"angular.band_asymmetry_b{b}"] = float(np.sum(powers[b]))
    out["angular.high_to_low_power"] = float(np.sum(powers[:, 2:]) / max(float(np.sum(powers[:, :2])), 1))
    out["angular.polar_entropy"] = float(-np.sum(weights * np.log(np.maximum(weights, 1e-15))))
    return out
