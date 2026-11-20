"""Test a short incoming proton before a bright measured pion candidate."""

from itertools import product

import numpy as np

from finite_chains import factory, refine


def nominated(original, hits):
    probabilities = original.get("initial_counter_scores")
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(str(i), probabilities.get(i, 0.0)) for i in range(4)]
    return (
        hits >= 400
        and probabilities is not None
        and len(probabilities) == 4
        and sum(probabilities[2:]) >= 0.14
    )


def hypotheses(competition, detector_records, optics):
    from balanced_search import feasible
    from chain_soft_search import canonical
    from charge_starts import allowed_chord
    from material import material
    from optical_chain_search import physical_start
    from staged_physics.primary import Aperture
    from value_only_receiver import seed_log_likelihood

    tracks = [
        r
        for r in detector_records
        if r.get("count") == 1
        and r.get("supported")
        and tuple(r.get("particles") or ()) in {("pion",), ("proton",), ("muon",)}
        and len(r.get("parameters", ())) == 14
    ]
    if not tracks:
        return []
    measured = sorted(tracks, key=lambda r: -r.get("gain", -np.inf))[:2]
    model = competition.model
    prototype = factory(competition.event, model.bg, ("proton", "pion"), 4)
    aperture = Aperture()
    entry = np.asarray([0.0, 0.0, aperture.z])
    forward = np.asarray([0.0, 0.0, 1.0])
    candidates = []
    opt_dirs = [np.asarray(a) for _, g, _ in optics for a in g["axes"] if a[2] > 0.68]
    for saved in measured:
        x = np.asarray(saved["parameters"], float)
        water = material(saved["particles"][0])
        tangent = x[3:6]
        if tangent[2] < 0.63:
            continue
        near_axis = -float(x[:2] @ tangent[:2]) / max(float(tangent[:2] @ tangent[:2]), 1.0e-10)
        origin_z = x[2] + near_axis * tangent[2]
        prior_depth = np.clip(origin_z - aperture.z, 0.035, 0.65)
        radiating = x[9] * x[10]
        energy = float(water["energy_at"](1000.0 * x[9]))
        beta = np.sqrt(1.0 - (water["mass"] / (water["mass"] + energy)) ** 2)
        theta = np.degrees(np.arccos(np.clip(1.0 / (water["index"] * beta), 0.0, 1.0)))
        directions = [tangent]
        for axis in opt_dirs:
            if 0.96 < tangent @ axis < 0.998 and not any(axis @ a > 0.998 for a in directions):
                directions.append(axis)
        for depth, axis, opening in product(
            np.unique(np.r_[prior_depth, 0.075, 0.14, 0.22, 0.32, 0.48, 0.67]),
            directions[:3],
            (26.0, 32.0, 36.0),
        ):
            point = entry + depth * forward
            chord = allowed_chord(point, axis)
            if chord is None or chord[0] > 1.0e-5 or chord[1] < 0.15:
                continue
            length = min(radiating, 0.996 * chord[1])
            for factor in (0.60, 1.0):
                p = physical_start(
                    entry,
                    (forward, axis),
                    (depth, length * factor),
                    (opening, theta),
                    prototype.particles,
                    1.0,
                    x[-2:],
                )
                if p is None or (p := canonical(prototype, p)) is None:
                    continue
                if feasible(prototype, p) and prototype.hardware_margin(p) > 1.0e-5:
                    candidates.append((seed_log_likelihood(prototype, p), prototype.particles, p, depth))
    finalists = []
    for entry in sorted(candidates, key=lambda r: -r[0]):
        if all(
            abs(entry[3] - other[3]) > 0.065 or entry[2][12:15] @ other[2][12:15] < 0.995
            for other in finalists
        ):
            finalists.append(entry)
        if len(finalists) == 3:
            break
    return finalists


def compare(competition, optics, original):
    from coupled_model import Model

    prior = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        records = [original.get("selected") or {}, *original.get("candidates", ())]
        seeds = hypotheses(competition, records, optics)
        output = []
        for seed in seeds:
            fit = refine(competition, seed, physical_coverage=True, visibility_probability=True)
            if fit is not None:
                fit["label"] = "short_proton_before_pion_" + fit["label"]
                output.append(fit)
        return output, len(seeds)
    finally:
        Model.delta_scale = prior
