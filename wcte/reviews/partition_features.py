"""Charge and geometry features for a bounded partition adjustment."""

import numpy as np
from scipy.special import logsumexp

from shared_scoring import match_charge_shares


def timing_features(event, predicted, saved, trial):
    from timed_optical import arrival

    n, k = predicted.shape
    features = np.zeros((n, k - 1, 7), dtype=np.float32)
    if trial is None:
        return features
    geometry = trial["geometry"]
    centers = arrival(event, geometry) + np.asarray(trial["clocks"])
    difference = event.times_ns[:, None] - np.median(event.times_ns) - centers
    weights = np.minimum(event.charges_pe, 2.0)
    match, optical, dice = match_charge_shares(predicted, saved[:, 1:].T, weights)
    distance = []
    for center, axis, opening, length in zip(
        geometry["vertices"],
        geometry["axes"],
        geometry["openings_deg"],
        geometry.get("lengths_mm", [0.0] * (k - 1)),
        strict=True,
    ):
        offset = event.pmt_positions_mm - center
        along = offset @ axis
        radial = np.linalg.norm(offset - along[:, None] * axis, axis=1)
        angle = np.deg2rad(np.clip(opening, 10.0, 42.3))
        span = length or 0.0
        emission = np.clip(along - radial / np.tan(angle), -0.5 * span, 0.5 * span)
        residual = np.abs(np.arctan2(radial, along - emission) - angle)
        distance.append(np.minimum(residual * 5.0, 3.0))
    widths = np.deg2rad(geometry["widths_deg"])
    for i, j, alignment in zip(match, optical, dice, strict=True):
        if alignment < 0.40:
            continue
        z = difference[:, j]
        spread = np.maximum(widths[j], 0.02)
        features[:, i, 0] = alignment
        features[:, i, 1] = np.minimum(np.maximum(z, 0.0), 6.0) / 3.0
        features[:, i, 2] = np.minimum(np.maximum(-z, 0.0), 6.0) / 3.0
        features[:, i, 3] = distance[j]
        features[:, i, 4] = min(spread * 5.0, 2.0)
        features[:, i, 5] = np.minimum(distance[j] / (5.0 * spread), 4.0) / 2.0
        features[:, i, 6] = saved[:, j + 1] - predicted[:, i + 1]
    return features


def features(event, shares, particle, saved=None, trial=None):
    n, m = shares.shape
    weights = np.minimum(event.charges_pe, 2.0)
    ring_charge = weights @ shares[:, 1:]
    effective = shares[:, 1:].sum(0)
    ring_share = ring_charge / max(weights.sum(), 1.0)
    multi = min(m - 2, 2)
    base = np.empty((n, m - 1, 13), dtype=np.float32)
    base[:, :, :3] = np.eye(3, dtype=np.float32)[multi]
    base[:, :, 3] = np.log(np.maximum(shares[:, 1:], 0.002))
    base[:, :, 4] = np.log(np.maximum(shares[:, :1], 0.002))
    base[:, :, 5] = np.log1p(np.minimum(event.charges_pe, 7.0))[:, None]
    base[:, :, 6] = np.log1p(effective) / 5.0
    base[:, :, 7] = ring_share
    base[:, :, 8] = np.log1p(n) / 5.0
    base[:, :, 9] = (ring_share < 0.2) * base[:, :, 3]
    base[:, :, 10] = (multi == 0) * base[:, :, 3]
    base[:, :, 11] = (multi == 0) * base[:, :, 4]
    base[:, :, 12] = float(particle == "pion")
    context = timing_features(event, shares, saved, trial)
    return np.concatenate((base, context), axis=-1)


def adjusted(logs, features, beta):
    raw = (features @ beta.astype(np.float32)).reshape(logs.shape[0], logs.shape[1] - 1)
    strength = np.tanh(raw / 1.7)
    design = logs.copy()
    design[:, 1:] += 1.7 * strength
    return np.exp(design - logsumexp(design, axis=1, keepdims=True)), 1.0 - strength**2


def limit_charge_loss(before, after, q, maximum_loss):
    old = before[:, 1:]
    mass = q @ old
    loss = q @ np.maximum(old - after[:, 1:], 0.0)
    proportion = np.max(np.divide(loss, mass, out=np.zeros_like(mass), where=mass > 1.0e-8), initial=0.0)
    fraction = min(1.0, maximum_loss / max(proportion, 1.0e-12))
    return before + fraction * (after - before)


def reshare(shares, design, weights, parameters, maximum_loss):
    before = shares.astype(np.float32)
    count = len(parameters)
    f = design[:, :, :count].reshape(-1, count)
    revised, _ = adjusted(np.log(np.maximum(before, 1.0e-12)), f, parameters)
    revised = revised.astype(float)
    if maximum_loss < 1.0:
        revised = limit_charge_loss(before, revised, weights, maximum_loss)
    return revised / revised.sum(1, keepdims=True)
