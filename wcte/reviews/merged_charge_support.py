"""Test whether a lower-count fit preserves two split parts of the same ring."""

from itertools import combinations, permutations

import numpy as np


def charge_preservation(prediction, lower):
    """Return the weakest forward and reverse light retention for each merge."""
    previous = prediction["fractions"][:, 1:]
    new = lower[:, 1:]
    sizes = previous.shape[1], new.shape[1]
    if sizes not in ((2, 1), (3, 2)):
        raise ValueError("Expected deletion of one fitted ring by merging a pair")
    q = np.minimum(prediction["charges"], 2.0)
    original_mass = np.maximum(q @ previous, 1e-12)
    lower_mass = np.maximum(q @ new, 1e-12)
    result = []
    for pair in combinations(range(sizes[0]), 2):
        remaining = tuple(j for j in range(sizes[0]) if j not in pair)
        for order in permutations(range(sizes[1])):
            groups = (pair,) if not remaining else (pair, remaining)
            grouped = np.column_stack([previous[:, group].sum(1) for group in groups])
            overlap = q @ np.minimum(grouped, new[:, order])
            forward = float(np.min(overlap / lower_mass[list(order)]))
            reverse = float(np.min(overlap / np.maximum(q @ grouped, 1e-12)))
            member = min(
                float((q @ np.minimum(previous[:, j], new[:, k])) / original_mass[j])
                for group, k in zip(groups, order, strict=True)
                for j in group
            )
            result.append((forward, reverse, member))
    return result


def physical_merger(trial, count):
    if count == 2:
        sections = tuple(tuple(group) for group in trial.get("lineage_sections", ()))
        return (
            sections == ((0, 1),)
            and len(set(trial.get("particles", ()))) == 1
            and trial.get("turn_deg", np.inf) < 15.0
        )
    return len(trial.get("lineage_sections", ())) == 2


def lost_ring_signal(prediction, proposed, modeled=None):
    """Describe the previous light left unexplained by a reduction."""
    q = np.minimum(prediction["charges"], 2.0)
    previous = prediction["fractions"][:, 1:]
    covered = proposed[:, 1:].sum(1) if modeled is None else np.maximum(proposed[:, 1:].sum(1), modeled)
    covered = np.clip(covered, 0.0, 1.0)
    retained = (q @ np.minimum(previous, covered[:, None])) / np.maximum(q @ previous, 1.0e-10)
    uncovered = q[:, None] * previous * (1.0 - covered[:, None])
    modules = prediction["pmt_ids"] // 100
    remaining = [
        float(charge.sum() - np.bincount(modules, weights=charge).max(initial=0.0)) for charge in uncovered.T
    ]
    return retained.tolist(), remaining


def preserves_distributed_light(retained, remaining):
    return all(kept >= 0.75 or missed <= 18.0 for kept, missed in zip(retained, remaining, strict=True))
