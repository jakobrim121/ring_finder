"""Fit a faint charged branch while allowing its main track to continue."""

import numpy as np
from scipy.optimize import minimize

from chain_sources import paths
from charge_starts import weighted_quantile
from geometry import hardware_margin
from joined_segments import arc
from material import material
from recoil_competitor import parent_tracks, render
from source import trajectory
from timed_charged_light import photon_time
from wctering.seeding.coherent_timing import C
from wcte.counting import track_counting


def child_at(parent, daughter, frame, z):
    """Attach the daughter to the parent with an analytic birth derivative."""
    u, *direction, log_range, log_fraction = z
    point, tangent, _, _ = trajectory(parent, np.array([u]))
    result = np.r_[point[0] / 1000.0, daughter]
    jacobian = np.zeros((12, 5))
    jacobian[:3, 0] = parent[9] * parent[10] * tangent[0]
    # The direction chart and range chart are independent at the birth.
    tangent_chart = daughter[:3] + frame @ direction
    norm = np.linalg.norm(tangent_chart)
    axis = tangent_chart / norm
    result[3:6] = axis
    jacobian[3:6, 1:3] = (frame - np.outer(axis, axis @ frame)) / norm
    result[9:11] = np.exp([log_range, log_fraction])
    jacobian[9, 3] = result[9]
    jacobian[10, 4] = result[10]
    return result, jacobian


def fit(event, single, single_state, rival, rival_state, fractions, short, budget=28):
    """Compare a connected charged recoil below the ring support floor."""
    weak = paths(rival_state["parameters"][: rival.ng])[short][0]
    main = paths(rival_state["parameters"][: rival.ng])[1 - short][0]
    seeds = list(parent_tracks(single, single_state, rival, rival_state, short))
    seeds.append((rival.particles[1 - short], main))
    q = np.minimum(event.charges_pe, 2.0)
    expected_response = rival_state["response"]
    diffuse = np.exp(rival_state["parameters"][-1])
    options = []
    nodes = np.linspace(0.0, 1.0, 12)
    for mass, parent in seeds:
        if mass == "electron":
            continue
        position = trajectory(parent, nodes)[0] / 1000.0
        target = weak[:3] + (weak[9] * weak[10] * weak[3:6] if short == 0 else 0.0)
        index = int(np.argmin(np.linalg.norm(position - target, axis=1)))
        if np.linalg.norm(position[index] - target) > 0.30:
            continue
        source = render(single, parent, mass, 4)[1]
        for recoil in dict.fromkeys((rival.particles[short], "pion", "proton")):
            if recoil == "electron":
                continue
            water = material(recoil)
            edge = water["margin_mm"] / 1000.0
            ref = material(rival.particles[short])
            energy = float(ref["energy_at"](1000.0 * weak[9])) * water["mass"] / ref["mass"]
            energy = np.clip(energy, water["energy"][0] + 1.01, water["energy"][-1])
            ran = float(water["range_at"](energy)) / 1000.0
            for r in (ran, edge + 0.42 * max(0.0, ran - edge)):
                child = weak.copy()
                child[:3] = position[index]
                child[6:9] = 0.0
                child[9] = r
                child[10] = min(0.999, weak[9] * weak[10] / r)
                if hardware_margin(child) <= 1e-5:
                    continue
                _, extra, _ = render(single, child, recoil, 4)
                ll = single.obs.score(expected_response * (source + extra) + diffuse * single.bg, single.ix)[
                    0
                ]
                options.append((ll, mass, parent, recoil, child, nodes[index]))
    ranked = sorted(options, key=lambda row: row[0], reverse=True)
    best = None
    seen = set()
    for _, mass, parent, recoil, child, u in ranked:
        pair = mass, recoil
        if pair in seen:
            continue
        seen.add(pair)
        result = refine(
            event,
            single,
            parent,
            mass,
            child,
            recoil,
            u,
            expected_response,
            diffuse,
            q * fractions[:, short + 1],
            budget,
        )
        if result is not None and (best is None or result["ll"] > best["ll"]):
            best = result
        if len(seen) == 2:
            break
    return best


def refine(event, model, parent, parent_mass, child, child_mass, u, response, diffuse, weight, budget):
    """Keep the same calibrated intensity and fit the branch's finite extent."""
    water = material(child_mass)
    origin = child[3:6]
    t = np.cross(origin, np.eye(3)[np.argmin(abs(origin))])
    t /= np.linalg.norm(t)
    frame = np.column_stack([t, np.cross(origin, t)])
    start = np.r_[u, 0.0, 0.0, np.log(child[9:11]), np.log([response, diffuse])]
    lower = float(water["range_at"](water["energy"][0] + 1.0)) / 1000.0
    upper = water["maximum_range_mm"] / 1000.0
    bounds = [
        (max(0.0, u - 0.25), min(1.0, u + 0.25)),
        (-0.6, 0.6),
        (-0.6, 0.6),
        (np.log(lower), np.log(upper)),
        (np.log(0.004), 0.0),
        *model.bounds[-2:],
    ]
    parent_light = render(model, parent, parent_mass, 8)[1]

    def evaluate(z, order=8, primary=parent_light):
        p, mapping = child_at(parent, child[3:], frame, z[:5])
        core, total, derivative = render(model, p, child_mass, order)
        eff, bg = np.exp(z[-2:])
        signal = eff * (primary + total)
        background = bg * model.bg
        mean = signal + background
        ll, residual = model.obs.score(mean, model.ix)
        jacobian = np.column_stack([eff * derivative @ mapping, signal, background])
        return ll, residual @ jacobian, p, mean[model.ix], eff * core[model.ix], eff * total[model.ix]

    def objective(z):
        ll, gradient, *_ = evaluate(z)
        return -ll / model.total, gradient / model.total

    def limit(z):
        # The short-branch envelope only bounds the search. Conditional
        # PMT support determines whether it would count as another ring.
        ran, portion = np.exp(z[3:5])
        return water["margin_mm"] / 1000.0 + 0.10 - ran * portion

    def limit_gradient(z):
        slope = -np.exp(z[3:5].sum())
        return np.r_[np.zeros(3), slope, slope, 0.0, 0.0]

    fit = minimize(
        objective,
        start,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=[dict(type="ineq", fun=limit, jac=limit_gradient)],
        options=dict(maxiter=budget, ftol=5e-7),
    )
    z = fit.x
    if limit(z) < -1e-6:
        return None
    low, high = [render(model, parent, parent_mass, n)[1] for n in (16, 32)]
    coarse, *_ = evaluate(z, 16, low)
    ll, _, child, mean, core, extra = evaluate(z, 32, high)
    if hardware_margin(child) <= 1e-5 or abs(coarse - ll) >= 0.01:
        return None
    support = float(np.sum(1.0 - model.obs.no_source(mean, core / mean)))
    if support >= track_counting.get().min_track_pmts:
        return None
    parent_time = photon_time(event, model.normals[model.ix], [parent], parent_mass)
    child_time = photon_time(event, model.normals[model.ix], [child], child_mass)
    if z[0] >= 0.008 / (parent[9] * parent[10]):
        prefix = arc(parent, 0.0, z[0])
        if prefix is not None:
            m = material(parent_mass)
            midpoint = 1000.0 * prefix[9] * (1.0 - 0.5 * prefix[10])
            beta = np.sqrt(1.0 - (1.0 + float(m["energy_at"](midpoint)) / m["mass"]) ** -2)
            child_time += 1000.0 * prefix[9] * prefix[10] / (beta * C)
    time = event.times_ns - np.median(event.times_ns)
    parent_share = np.exp(z[-2]) * high[model.ix] / mean
    clock_weight = np.minimum(event.charges_pe, 2.0) * parent_share
    if clock_weight.sum() < 24.0:
        return None
    clock = weighted_quantile(time - parent_time, clock_weight, [0.5])[0]
    coherence = np.exp(-0.5 * ((time - clock - child_time) / 2.0) ** 2)
    overlap = float(weight @ (coherence * extra / mean) / max(weight.sum(), 1e-15))
    if overlap < 0.3:
        return None
    return dict(
        ll=float(ll),
        support=support,
        prompt_overlap=overlap,
        child_mass=child_mass,
        parent_mass=parent_mass,
        branch_fraction=float(z[0]),
        child_path=child.tolist(),
        evaluations=int(fit.nfev),
        quadrature_ll_error=abs(coarse - ll),
    )
