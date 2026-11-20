"""Evaluate the unchanged exported decision trees and available count classes.

Only detector-derived features enter the trees. No training happens here; the
pre-existing unified_v226 weights are loaded by multiplicity.py."""

import numpy as np
from numba import njit


def packed_trees(model):
    """Prepare immutable numerical copies once for this frozen exported model.

    Child indices become offsets in one flat forest. Leaf values, thresholds,
    missing-value routes and tree order are copied without numerical changes.
    The original JSON weights remain available and are never rewritten.
    """
    if "_packed_trees" not in model:
        trees = model["trees"]
        roots = np.cumsum([0] + [len(tree["left"]) for tree in trees[:-1]])

        def joined(name, dtype):
            return np.concatenate([np.asarray(tree[name], dtype) for tree in trees])

        left = np.concatenate(
            [np.asarray(tree["left"], np.int64) + offset for tree, offset in zip(trees, roots)]
        )
        right = np.concatenate(
            [np.asarray(tree["right"], np.int64) + offset for tree, offset in zip(trees, roots)]
        )
        model["_packed_trees"] = (
            np.asarray(model["baseline"], float),
            roots,
            np.asarray([tree["class_index"] for tree in trees], np.int64),
            left,
            right,
            joined("feature", np.int64),
            joined("threshold", float),
            joined("missing_left", bool),
            joined("leaf", bool),
            joined("value", float),
        )
    return model["_packed_trees"]


@njit(cache=True)
def tree_scores(features, packed):
    """Traverse each tree and add its leaf in the original scalar order."""
    baseline, roots, classes, left, right, feature, threshold, missing, leaf, value = packed
    raw = np.empty((len(features), len(baseline)))
    for row in range(len(features)):
        raw[row] = baseline
        for tree in range(len(roots)):
            node = roots[tree]
            while not leaf[node]:
                observed = features[row, feature[node]]
                go_left = missing[node] if np.isnan(observed) else observed <= threshold[node]
                node = left[node] if go_left else right[node]
            raw[row, classes[tree]] += value[node]
    return raw


def predict_exported(features, model):
    x = np.asarray(features, float)
    if x.ndim == 1:
        x = x[None, :]
    if model.get("kind") in ("probability_ensemble", "h1_score_ensemble"):
        weights = np.asarray(model["weights"], float)
        if (
            len(weights) != len(model["models"])
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("Invalid exported ensemble weights")
        probabilities = [predict_exported(x, m) for m in model["models"]]
        if model["kind"] == "h1_score_ensemble":
            high = sum(w * p[:, 1] for w, p in zip(weights, probabilities))
            return np.column_stack([1 - high, high])
        if any(p.shape != probabilities[0].shape for p in probabilities):
            raise ValueError("Exported class ensembles must have aligned classes")
        return sum(w * p for w, p in zip(weights, probabilities))
    raw = tree_scores(x, packed_trees(model))
    if raw.shape[1] == 1:
        p = 1 / (1 + np.exp(-np.clip(raw[:, 0], -709, 709)))
        return np.column_stack([1 - p, p])
    raw -= np.max(raw, axis=1, keepdims=True)
    p = np.exp(raw)
    return p / np.sum(p, axis=1, keepdims=True)


def score_features(features, model):
    x = np.asarray([features.get(k, np.nan) for k in model["feature_names"]], float)
    counts = predict_exported(x, model["standard"])[0]
    high = (
        predict_exported(x, model["high_purity"])[0, 1] if model.get("high_purity") is not None else counts[1]
    )
    return counts, float(high)


def available_classes(payload):
    """Only return multiplicities for which the unchanged fitter has labels."""
    # Legacy early returns keep observed support in zero_ring_support_gate,
    # while completed fits also copy it to the top level. Missing top-level
    # metadata must never be interpreted as an empty detector event.
    support = payload.get("zero_ring_support_gate") or {}
    n = int(payload.get("usable_hits", support.get("usable_hits", 0)))
    pmts = int(payload.get("usable_unique_pmts", support.get("usable_unique_pmts", 0)))
    allowed = np.array([True, pmts >= 30, False, False], dtype=bool)
    h12 = payload.get("full_event_hypotheses") or {}
    h3 = (payload.get("three_ring_hypothesis_test") or {}).get("full_event_hypotheses") or {}
    for k, h in [(2, h12), (3, h3)]:
        labels = np.asarray(h.get("partition_labels", []), int)
        allowed[k] = bool(n > 0 and labels.shape == (n,) and set(labels) == set(range(1, k + 1)))
    return allowed


def decide(probabilities, high_score, payload, threshold):
    allowed = available_classes(payload)
    p = np.asarray(probabilities, float).copy()
    p[~allowed] = 0
    prediction = int(np.argmax(p)) if np.sum(p) > 0 else 0
    selected = bool(allowed[1] and high_score >= threshold)
    return prediction, selected, allowed
