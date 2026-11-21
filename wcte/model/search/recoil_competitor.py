"""Compare two counted legs with one track plus a short charged recoil."""

import numpy as np
from scipy.optimize import minimize

from analytic_knockon import render as knockon
from chain_sources import paths
from charge_starts import allowed_chord, weighted_quantile
from chord_profile import profile
from coherent_partition import prediction_time
from geometry import hardware_margin
from joined_segments import arc
from material import material
from source import raw, trajectory
from wctering.seeding.coherent_timing import C
from wcte.counting import track_counting


def parent_tracks(one, one_state, two, state, short):
    """Preserve the measured main track and try its physical continuation."""
    incumbent = one_state["parameters"][:12]
    yield one.particles[0], incumbent
    dominant = paths(state["parameters"][: two.ng])[1 - short][0]
    particle = two.particles[1 - short]
    if particle != one.particles[0]:
        yield particle, dominant
    chord = allowed_chord(dominant[:3], dominant[3:6])
    if chord is None:
        return
    length = dominant[9] * dominant[10]
    limit = material(particle)["maximum_range_mm"] / 1000.0
    for extra in (0.10, 0.24):
        signed = -min(extra, -chord[0] - 0.002) if short == 0 else min(extra, chord[1] - length - 0.002)
        if (short == 0 and signed >= -0.01) or (short == 1 and signed <= 0.01):
            continue
        proposed = arc(dominant, signed / length, 1.0) if short == 0 else dominant.copy()
        if proposed is None:
            continue
        if short == 1:
            proposed[9] = max(proposed[9], length + signed)
            proposed[10] = (length + signed) / proposed[9]
        if proposed[9] < limit and hardware_margin(proposed) > 1e-5:
            yield particle, proposed


def render(model, p, mass, order, *, core_gradient=False):
    """Use the calibrated light of the arc, including its uncounted tail."""
    length = 1000.0 * p[9] * p[10]
    parts, derivative = raw(model.pos, model.normals, p, order, False, mass)
    delta, d_delta = knockon(model.pos, model.normals, p, mass, order)
    dr = np.zeros(len(p))
    dr[9:11] = 1000.0 * p[[10, 9]]
    summed = parts.sum(0)
    core = length * parts[0]
    light = length * summed + delta
    gradient = length * derivative.sum(0) + summed[:, None] * dr + d_delta
    if core_gradient:
        return core, light, gradient, length * derivative[0] + parts[0, :, None] * dr
    return core, light, gradient


def search(event, one, one_state, two, state, fractions, short):
    """Profile a connected recoil over physical energy and emitting length.

    The dominant track is held at each detector-derived solution. The recoil
    must remain below the PMT floor and explain timed charge from
    the apparent split. Optical response remains a shared nuisance.
    """
    rival = paths(state["parameters"][: two.ng])[short][0]
    q = np.minimum(event.charges_pe, 2.0)
    weak = fractions[:, short + 1]
    t = event.times_ns - np.median(event.times_ns)
    response, diffuse = np.exp(state["parameters"][-2:])
    modes = []
    for mass, parent in parent_tracks(one, one_state, two, state, short):
        primary = render(one, parent, mass, 8)[1]
        # A recoil can leave the parent at any point. The fitted weak cone
        # supplies a nearby birth and initial tangent, independent of mass.
        nodes = np.linspace(0.0, 1.0, 9)
        positions, _, _, _ = trajectory(parent, nodes)
        nearest = np.argsort(np.linalg.norm(positions / 1000.0 - rival[:3], axis=1))[:2]
        for index in nearest:
            if np.linalg.norm(positions[index] / 1000.0 - rival[:3]) > 0.30:
                continue
            origin = positions[index] / 1000.0
            for daughter in dict.fromkeys((two.particles[short], "pion", "proton")):
                if daughter == "electron":
                    continue
                water = material(daughter)
                ref = material(two.particles[short])
                energy = float(ref["energy_at"](1000.0 * rival[9])) * water["mass"] / ref["mass"]
                energy = np.clip(energy, water["energy"][0] + 1.01, water["energy"][-1])
                ran = float(water["range_at"](energy)) / 1000.0
                limit = allowed_chord(origin, rival[3:6])
                if limit is None or limit[0] > 0.0 or limit[1] < 0.012:
                    continue
                length = min(0.995 * limit[1], 0.98 * ran, rival[9] * rival[10])
                initial = np.r_[origin, rival[3:6], np.zeros(3), ran, length / ran, rival[11]]
                if hardware_margin(initial) <= 1e-5:
                    continue
                _, light, _ = render(one, initial, daughter, 8)
                ll = one.obs.score(response * (primary + light) + diffuse * one.bg, one.ix)[0]
                modes.append((ll, mass, parent, nodes[index], primary, daughter, initial, limit[1]))
    best = None
    for _, mass, parent, fraction, primary, daughter, initial, limit in sorted(
        modes, key=lambda row: row[0], reverse=True
    )[:2]:
        result = shorten(
            event,
            one,
            mass,
            parent,
            fraction,
            primary,
            daughter,
            initial,
            limit,
            q,
            t,
            weak,
            response,
            diffuse,
        )
        if result is not None and (best is None or result["ll"] > best["ll"]):
            best = result
    return best


def shorten(
    event,
    model,
    mass,
    parent,
    birth_fraction,
    primary,
    daughter,
    initial,
    limit,
    charge,
    time,
    weak,
    response,
    diffuse,
):
    """Optimize a small range/length chart and recheck both light integrals."""
    water = material(daughter)
    base = model.bounds[-2:]
    maximum = water["maximum_range_mm"] / 1000.0
    minimum = float(water["range_at"](water["energy"][0] + 1.0)) / 1000.0
    start = np.r_[np.log(initial[9:11]), np.log([response, diffuse])]
    bounds = [(np.log(minimum), np.log(maximum)), (np.log(0.002), 0.0), *base]

    def evaluate(z):
        p = initial.copy()
        p[9:11] = np.exp(z[:2])
        core, light, derivative = render(model, p, daughter, 8)
        efficiency, background = np.exp(z[-2:])
        signal = efficiency * (primary + light)
        sky = background * model.bg
        mean = signal + sky
        ll, residual = model.obs.score(mean, model.ix)
        jac = np.column_stack([efficiency * derivative[:, 9:11] * p[9:11], signal, sky])
        return -ll, residual @ jac

    # Log range plus log fraction is log arclength. This exact linear
    # constraint keeps the recoil inside its contiguous water interval.
    result = minimize(
        evaluate,
        start,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=[
            dict(
                type="ineq",
                fun=lambda z: np.log(0.995 * limit) - z[:2].sum(),
                jac=lambda z: np.array([-1.0, -1.0, 0.0, 0.0]),
            )
        ],
        options=dict(maxiter=18, ftol=1e-5),
    )
    p = initial.copy()
    p[9:11] = np.exp(result.x[:2])
    if p[9] * p[10] > limit or hardware_margin(p) <= 1e-5:
        return None
    primary_low, primary_high = (render(model, parent, mass, order)[1] for order in (16, 32))
    _, light_low, _ = render(model, p, daughter, 16)
    core, light, _ = render(model, p, daughter, 32)
    efficiency, background = np.exp(result.x[-2:])
    total = primary_high + light
    if total.sum() <= 0.0:
        return None
    ll, amplitudes, error = profile(
        np.vstack([total / total.sum(), model.bg]),
        model.obs,
        model.ix,
        [efficiency * total.sum(), background],
        budget=20,
    )
    efficiency = amplitudes[0] / total.sum()
    if error > 1e-5 or not base[0][0] <= np.log(efficiency) <= base[0][1]:
        return None
    mean = efficiency * total + amplitudes[1] * model.bg
    low = efficiency * (primary_low + light_low) + amplitudes[1] * model.bg
    quadrature = abs(model.obs.score(low, model.ix)[0] - ll)
    source = np.clip(efficiency * core[model.ix] / mean[model.ix], 0.0, 1.0)
    support = float(np.sum(1.0 - model.obs.no_source(mean[model.ix], source)))
    if quadrature >= 0.01 or support >= track_counting.get().min_track_pmts:
        return None

    # A recoil emitted at the same interaction has a common event clock.
    # Its small angle-dependent path delay is kept in the forward timing.
    s = np.linspace(0.0, birth_fraction, 9)
    material_parent = material(mass)
    energy = material_parent["energy_at"](1000.0 * parent[9] * (1.0 - parent[10] * s))
    beta = np.sqrt(np.maximum(1.0 - (1.0 + energy / material_parent["mass"]) ** -2, 1e-8))
    offset = 1000.0 * parent[9] * parent[10] * np.trapz(1.0 / beta, s) / C
    parent_time, visible = prediction_time(event, np.r_[parent, 0.0, 0.0], (mass,), charge)
    recoil_time, _ = prediction_time(event, np.r_[p, 0.0, 0.0], (daughter,), charge)
    direct = charge * (1.0 - weak) * (visible[:, 0] > 1e-6)
    if not direct.any():
        return None
    clock = weighted_quantile(time - parent_time[:, 0], direct, [0.5])[0]
    fraction = efficiency * light[model.ix] / mean[model.ix]
    prompt = np.exp(-0.5 * ((time - recoil_time[:, 0] - clock - offset) / 2.0) ** 2)
    own = charge * weak**2
    overlap = float(own @ (fraction * prompt) / max(own.sum(), 1e-15))
    if overlap < 0.35:
        return None
    return dict(
        ll=float(ll),
        expected_pmts=support,
        prompt_overlap=overlap,
        recoil_particle=daughter,
        primary_particle=mass,
        recoil_range_m=float(p[9]),
        source_evaluations=int(result.nfev),
        quadrature_ll_error=quadrature,
    )
