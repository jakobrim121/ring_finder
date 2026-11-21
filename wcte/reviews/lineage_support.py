"""Measure prompt support of linked emitting sections before dropping weak cores."""

import numpy as np

from short_kink import matched
from wcte.counting import track_counting


def group_sections(paths, support):
    from source import trajectory

    active = {j for j, signal in enumerate(support) if signal >= min(10.0, track_counting.get().min_track_pmts)}
    cores = []
    threshold = track_counting.get().merge_cosine
    for j in sorted(active):
        if j - 1 in active and trajectory(paths[j - 1], np.ones(1))[1][0] @ paths[j][3:6] >= threshold:
            cores[-1] = (*cores[-1], j)
        else:
            cores.append((j,))
    return cores


def compare(competition, model, state):
    from chain_sources import paths

    sections = tuple((j,) for j in range(len(model.particles)))
    separate = matched(competition, model, state, sections, "lineage_sections", physical_coverage=True)
    if separate is None:
        return None
    support = [v["gate_pmt_support"] for v in separate["neighborhoods"]]
    physical = [p for p, _ in paths(state["parameters"][: model.ng])]
    groups = group_sections(physical, support)
    if not groups:
        return None
    label = "lineage_union_" + "_".join(model.particles)
    merged = (
        matched(competition, model, state, groups, label, physical_coverage=True)
        if tuple(groups) != sections
        else separate
    )
    if merged is None:
        return None
    minimum = track_counting.get().min_track_pmts
    retained = [i for i, p in enumerate(merged["neighborhoods"]) if p["gate_pmt_support"] >= minimum]
    if not retained:
        return None
    result = dict(merged, label=label, count=len(retained))
    result["neighborhoods"] = [merged["neighborhoods"][i] for i in retained]
    result["lineage_sections"] = [groups[i] for i in retained]
    result["individual_prompt_support"] = support
    result["fractions"] = {}
    for name, share in merged["fractions"].items():
        kept = share[:, np.asarray(retained) + 1]
        result["fractions"][name] = np.column_stack([1.0 - kept.sum(1), kept])
    return result
