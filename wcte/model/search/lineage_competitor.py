"""Fit two successive emitting regions that still count as one ring."""

import numpy as np

from balanced_search import feasible
from chain_sources import paths
from charge_starts import recoil_start
from coupled_model import Model as Coupled
from junction_coordinates import Junction
from material import material
from source import trajectory
from track_coordinates import fit as fit_chart
from wcte.counting import track_counting


class OneLineage(Coupled):
    """Share the unrestricted split's energy, geometry and detector model."""

    def all_constraints(self, x):
        values, gradient = super().all_constraints(x)
        incoming = x[:12]
        tangent, derivative = trajectory(incoming, np.ones(1))[1::2]
        outgoing = x[12:15]
        cosine = tangent[0] @ outgoing
        d_cosine = np.r_[outgoing @ derivative[0], tangent[0], np.zeros(len(x) - 15)]
        return np.r_[values, cosine - track_counting.get().merge_cosine], np.vstack([gradient, d_cosine])


def starting_paths(state, particles, masses):
    """Keep the measured lengths and opening angles when changing material."""
    x = state["parameters"].copy()
    incoming, outgoing = [p for p, _ in paths(x[:-2])]
    a = trajectory(incoming, np.ones(1))[1][0]
    b = outgoing[3:6]
    transverse = b - (b @ a) * a
    norm = np.linalg.norm(transverse)
    rules = track_counting.get()
    if norm > 1e-12 and a @ b < rules.merge_cosine:
        angle = np.deg2rad(max(0.0, rules.track_merge_angle_deg - 2.0))
        axis = np.cos(angle) * a + np.sin(angle) * transverse / norm
        x[12:15] = axis
        x[15:18] -= (x[15:18] @ axis) * axis
    for j, (previous, particle) in enumerate(zip(particles, masses)):
        if previous == particle:
            continue
        ran, fraction = x[9 + 9 * j : 11 + 9 * j]
        length = ran * fraction
        old, new = material(previous), material(particle)
        # At equal beta, kinetic energy is proportional to rest mass.
        middle = old["energy_at"](1000.0 * (ran - 0.5 * length))
        energy = np.clip(middle * new["mass"] / old["mass"], new["energy"][0] + 1.01, new["energy"][-1])
        maximum = new["maximum_range_mm"] / 1000.0 * (1.0 - 1e-10)
        ran = min(maximum, max(1.001 * length, 0.5 * length + new["range_at"](energy) / 1000.0))
        if ran <= length:
            return None
        x[9 + 9 * j : 11 + 9 * j] = ran, length / ran
    return x


def fit(event, single, model, state, budget, *, other_starts=(), timed=False, separate_knockon=False):
    """Profile the local turn under the same physical hypothesis and noise law."""
    anchors = [model.particles, (single.particles[0],) * 2]
    # Heavy charged tracks can look similar in angle but imply different
    # slowing rates. Try either fitted region under proton and pion transport.
    anchors.extend((p, p) for p in ("proton", "pion"))
    inputs = event.pmt_ids, event.charges_pe, model.bg
    candidates = {}
    for masses in dict.fromkeys(anchors):
        coarse = OneLineage(*inputs, masses, 4, 16)
        for parts, origin in ((model.particles, state["parameters"]), *other_starts):
            x = starting_paths({"parameters": origin}, parts, masses)
            if x is None:
                continue
            x = recoil_start(coarse, x)
            if x is None or not feasible(coarse, x) or coarse.hardware_margin(x) <= 1e-5:
                continue
            ll = coarse(x, True)["ll"]
            if masses not in candidates or ll > candidates[masses][0]:
                candidates[masses] = ll, masses, x, coarse
    best, maximum = None, -np.inf
    for _, masses, x, coarse in sorted(candidates.values(), key=lambda row: row[0], reverse=True)[:2]:
        fit_result, x = fit_chart(coarse, x, budget, budget, full_likelihood=True, chart=Junction(coarse, x))
        if not feasible(coarse, x) or coarse.hardware_margin(x) <= 1e-5:
            continue
        fine = OneLineage(*inputs, masses, 4, 32)
        ll = fine(x, True)["ll"]
        error = abs(ll - coarse(x, True)["ll"])
        if error >= 0.01:
            continue
        record = dict(ll=ll, materials=masses, evaluations=int(fit_result.nfev), quadrature_ll_error=error)
        if timed:
            from physical_pulse import profile

            pulse = profile(event, fine, fine(x, True), separate_knockon=separate_knockon)
            if pulse is None:
                continue
            record.update(pulse)
        score = ll + record.get("timing_ll", 0.0)
        if score > maximum:
            best, maximum = record, score
    return best
