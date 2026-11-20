"""Check that an optical promotion has the same measured light as a track fit."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from joint_prompt_fit import wavefront
from residual_joint_optics import prompt
from wavefront_support import physical_prompt


def candidate_modes(alternatives, count, prior_gain, ids, charges, background):
    """Reconstruct the best eligible physical modes only if the test needs them."""
    from bare_model import Model as Bare
    from coupled_model import Model as Coupled

    selected = []
    for score, fractions, record in sorted(alternatives, key=lambda row: row[0], reverse=True):
        if score < prior_gain - 12.0:
            break
        masses = record.get("particles")
        if record["count"] != count or not isinstance(masses, tuple):
            continue
        factory = Bare if all(p == "electron" for p in masses) else Coupled
        model = factory(ids, charges, background, masses, 4, 2 * record["order"])
        state = model(np.asarray(record["parameters"]), True)
        selected.append((model, state, fractions, score))
        if len(selected) == 2:
            break
    return selected


def optical_starts(model, state, fractions):
    """Project fitted physical cores back into independently refittable cones."""
    from chain_sources import paths
    from physical_residual_seed import optical_geometry

    core = state["posterior"][1:].T
    q = np.minimum(model.q, 2.0)
    candidates = []
    for j, ((track, _), mass) in enumerate(zip(paths(state["parameters"][: model.ng]), model.particles)):
        cone = optical_geometry(track, mass)
        if cone is not None and q @ core[:, j] >= 12.0:
            candidates.append((j, cone[0]))
    if len(candidates) < fractions.shape[1] - 1:
        return None
    indices, cones = zip(*candidates)
    # A counted ring can contain several collinear physical sections.
    # Start that cone on the section with the largest measured overlap.
    overlap = (fractions[:, 1:].T * q) @ core[:, indices]
    _, match = linear_sum_assignment(-overlap)
    if min(overlap[np.arange(len(match)), match]) < 7.0:
        return None
    return np.asarray(cones)[match]


def corroborated(event, trial, evidence, modes):
    """Match detector-only physical modes even when their total scores are close.

    A residual cone can look narrow while stealing charge from the same
    optical track. Require a second model to locate the new light and
    reject a candidate that loses an existing prompt footprint.
    """
    q = np.minimum(event.charges_pe, 2.0)
    old = trial["fractions"][:, 1:]
    extra = evidence["extra_component"]
    if evidence["independent_charge_fraction"] < 0.50:
        return None
    k = old.shape[1]
    target = q @ old
    times = np.column_stack([wavefront(event.pmt_positions_mm, p)[0] for p in trial["paths"]])
    optical, _ = prompt(event, times, trial["fractions"])
    for model, state, f, gain in modes:
        if f.shape[1] != k + 1:
            continue
        mass = np.sum(q[:, None] * f[:, 1:], axis=0)
        common = np.sum(q[:, None, None] * np.minimum(old[:, :, None], f[:, None, 1:]), axis=0)
        _, match = linear_sum_assignment(-common / np.maximum(target[:, None] + mass[None], 1e-20))
        cross = common[np.arange(k), match]
        if np.any(cross < 0.56 * target) or cross[extra] < 0.56 * mass[match[extra]]:
            continue
        timed = physical_prompt(event, f, model, state, grouped=True)
        if timed is None:
            continue
        share = np.minimum(optical[:, extra], timed[:, match[extra]])
        receivers = share >= 0.20 * q
        if (
            share.sum() < 0.45 * optical[:, extra].sum()
            or receivers.sum() < 18
            or len(np.unique(event.pmt_ids[receivers] // 100)) < 3
        ):
            continue
        return dict(
            physical_count=k,
            charge_gain=float(gain),
            physical_track_masses=model.particles,
            matched_charge=(cross / np.maximum(target, 1e-20)).tolist(),
            matched_extra_prompt_pmts=int(receivers.sum()),
        )
    return None
