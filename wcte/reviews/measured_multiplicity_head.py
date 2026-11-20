"""Evaluate the small detector-only count estimator with array operations."""

from functools import lru_cache

import numpy as np
from scipy.special import erf, expit, softmax


@lru_cache(maxsize=6)
def load(path):
    with np.load(path, allow_pickle=False) as artifact:
        return dict(artifact)


def linear(x, state, path):
    return x @ state[path + ".weight"].T + state[path + ".bias"]


def norm(x, state, path):
    centered = x - x.mean(1, keepdims=True)
    inv = 1.0 / np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + 1.0e-5)
    return centered * inv * state[path + ".weight"] + state[path + ".bias"]


def silu(x):
    return x * expit(x)


def heads(x, state, prefix):
    first = prefix + ".self_attn."
    weights = state[first + "in_proj_weight"]
    d = weights.shape[1] // 4
    projected = x @ weights.T + state[first + "in_proj_bias"]
    q, k, v = (tensor.reshape(len(x), 4, d).transpose(1, 0, 2) for tensor in np.split(projected, 3, axis=1))
    mixture = softmax((q @ k.swapaxes(1, 2)) / np.sqrt(np.float32(d)), axis=2) @ v
    combined = mixture.transpose(1, 0, 2).reshape(len(x), 4 * d)
    return linear(combined, state, first + "out_proj")


def estimate(hits, proton, state):
    x = np.asarray(hits, dtype=np.float32)
    sensors = state["sensor"]
    structure = state["module"]
    groups = np.r_[0, np.flatnonzero(structure[:-1] != structure[1:]) + 1]
    sizes = state["tubes"][:, None]
    cond = np.full((len(x), 1), float(proton), dtype=np.float32)
    combined = np.column_stack([sensors, x, cond])
    start = linear(combined, state, "pmt.0")
    start = silu(norm(start, state, "pmt.1"))
    start = silu(linear(start, state, "pmt.3"))
    charge = x[:, 2:3]

    def collect(a):
        return np.add.reduceat(a, groups, axis=0)

    signal = collect(charge)
    pulses = collect(x[:, :1])
    moment = collect(np.column_stack((x[:, 1:2], charge * x[:, 4:5])))
    series = np.column_stack(
        (
            collect(start) / sizes,
            collect(start * charge) / np.maximum(signal, 0.5),
            signal / sizes,
            pulses / sizes,
            moment / sizes,
            state["detector"],
        )
    )
    local = silu(norm(linear(series, state, "module_input.0"), state, "module_input.1"))
    for layer in ("layers.0", "layers.1"):
        local += heads(norm(local, state, layer + ".norm1"), state, layer)
        u = linear(norm(local, state, layer + ".norm2"), state, layer + ".linear1")
        u = 0.5 * u * (1.0 + erf(u / np.sqrt(np.float32(2.0))))
        local += linear(u, state, layer + ".linear2")
    attention = softmax(linear(local, state, "global_score"), axis=0)
    aggregate = (local * attention).sum(0)
    summary = np.r_[aggregate, local.max(0), np.float32(proton), np.log1p(x[:, :3].sum(0)) / 7.0]
    return softmax(linear(silu(linear(summary, state, "count.0")), state, "count.3"))


def measured(event, particle, state, beam_clock):
    ids = state["ids"]
    lookup = np.searchsorted(ids, event.pmt_ids)
    np.testing.assert_array_equal(ids[lookup], event.pmt_ids)
    q = event.charges_pe
    t = event.times_ns
    features = np.zeros((len(ids), 5), dtype=np.float32)
    if len(lookup):
        residual = t - beam_clock[lookup]
        features[lookup] = np.column_stack(
            [
                np.ones(len(lookup)),
                np.log1p(q),
                np.minimum(q, 2.0) / 2.0,
                np.clip((t - np.median(t)) / 6.0, -3.0, 3.0),
                np.clip((residual - np.median(residual)) / 6.0, -4.0, 4.0),
            ]
        )
    return estimate(features, particle == "proton", state)
