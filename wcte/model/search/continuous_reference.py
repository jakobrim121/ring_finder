"""Find the continuous-track explanation of a proposed additional ring."""

import numpy as np

from balanced_search import feasible
from bare_model import Model as Bare
from chain_sources import paths
from charge_starts import allowed_chord
from continuous_track import trial
from material import material
from source import trajectory


def seed(state, rival, fit, short, particle):
    """Span both proposed footprints with one track of conserved material."""
    a, b = [p for p, _ in paths(fit["parameters"][: rival.ng])]
    origin, stop = a[:3], trajectory(b, np.array([1.0]))[0][0] / 1000.0
    span = stop - origin
    length = np.linalg.norm(span)
    if length < 0.06:
        return None
    direction = span / length
    strong = (a, b)[1 - short]
    water = allowed_chord(origin, direction)
    if water is None or water[0] > 0.0 or water[1] < 0.995 * length:
        return None
    length = min(length, 0.995 * water[1])
    parent = material(particle)
    original = material(rival.particles[1 - short])
    middle = 1000.0 * strong[9] * (1.0 - 0.5 * strong[10])
    beta_equiv = float(original["energy_at"](middle)) * parent["mass"] / original["mass"]
    energy = np.clip(beta_equiv, parent["energy"][0] + 1.01, parent["energy"][-1])
    ran = max(1.001 * length, float(parent["range_at"](energy)) / 1000.0 + 0.5 * length)
    if ran >= parent["maximum_range_mm"] / 1000.0:
        return None
    return np.r_[origin, direction, np.zeros(3), ran, length / ran, strong[11], state["parameters"][-2:]]


def fit(event, model, state, rival, rival_state, short):
    """Try a physical one-track fit spanning both apparent light regions."""
    modes = []
    args = event.pmt_ids, event.charges_pe, model.bg
    for particle in dict.fromkeys((model.particles[0], rival.particles[1 - short], "proton", "pion")):
        if particle == "electron":
            continue
        bare = Bare(*args, (particle,), 4, 8)
        starts = [seed(state, rival, rival_state, short, particle)]
        if particle == model.particles[0]:
            starts.append(state["parameters"])
        for x in starts:
            if x is not None and feasible(bare, x) and bare.hardware_margin(x) > 1e-5:
                modes.append((bare(x, True)["ll"], bare, x))
    results = []
    for _, bare, x in sorted(modes, key=lambda row: row[0], reverse=True)[:2]:
        fit = trial(bare, bare(x, True), iterations=18, maximum_evaluations=32)
        if fit is not None:
            results.append(fit)
    return max(results, key=lambda row: row[1]["ll"]) if results else None
