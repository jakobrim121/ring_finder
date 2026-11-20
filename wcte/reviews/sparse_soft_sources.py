"""Fit multiple physically bounded soft emitters alongside counted paths."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from cascade_visible_light import material_table, render
from light_competition import Template, optical_template
from module_vertices import birth_points


@dataclass
class Shower:
    birth: np.ndarray
    axis: np.ndarray
    energy: float
    rates: np.ndarray
    arrival: np.ndarray

    @property
    def yield_pe(self):
        return float(self.rates.sum())


@lru_cache(maxsize=2)
def angular_profile(energy):
    energies, cosine, luminosity, shifts = material_table()
    hi = int(np.clip(np.searchsorted(energies, energy), 1, len(energies) - 1))
    alpha = (energy - energies[hi - 1]) / (energies[hi] - energies[hi - 1])
    angular = (1.0 - alpha) * luminosity[:, hi - 1, 1].mean(0) + alpha * luminosity[:, hi, 1].mean(0)
    distance = (1.0 - alpha) * shifts[0, :, hi - 1, 1].mean() + alpha * shifts[0, :, hi, 1].mean()
    return cosine, angular, max(float(distance), 0.0)


def trial_vertices(competition, optics):
    from wctering.seeding.geometry import wcte_contains_vertex

    vertices = []

    def add(point):
        if wcte_contains_vertex(point) and all(np.linalg.norm(point - p) >= 105.0 for p in vertices):
            vertices.append(point)

    for _, geometry, _ in optics:
        for point in np.asarray(geometry["vertices"], float):
            add(point)
    for _, geometry, _ in optics:
        points = np.asarray(geometry["vertices"], float)
        axes = np.asarray(geometry["axes"], float)
        lengths = geometry.get("lengths_mm", [None] * len(points))
        for point, axis, length in zip(points, axes, lengths, strict=True):
            if length and length > 180.0:
                for distance in (-0.4, 0.4):
                    add(point + distance * length * axis)
    for _, _, t in optics:
        base = t.amplitudes @ t.shapes
        for p, *_ in birth_points(competition.event, competition.model, base, directional=True)[:2]:
            add(p)
    return vertices


def nominate_pool(competition, optics, seeds):
    """Share one detector-derived library across all the tested counts."""
    from material import material
    from residual_search import axis_grid
    from timed_charged_light import projected_aperture

    model = competition.model
    grid = np.vstack([axis_grid(144), *(np.asarray(g["axes"], float) for _, g, _ in optics)])
    grid /= np.linalg.norm(grid, axis=1)[:, None]
    surface = grid @ model.normals.T
    electron = material("electron")
    edge = electron["energy_at"](electron["margin_mm"])
    energies = (0.62 * edge, 0.975 * edge)
    statistics = []
    for _, t in seeds:
        expected = t.amplitudes @ t.shapes
        _, slope = model.obs.score(expected, model.ix)
        weights = 1.0 / np.maximum(expected, 0.08)
        old = t.shapes
        nuisance = (old * weights) @ old.T
        inv = np.linalg.pinv(nuisance, rcond=1e-9)
        statistics.append((slope - old.T @ (inv @ (old @ slope)) * weights, weights, old, inv))

    proposals = [[] for _ in statistics]
    for vertex in trial_vertices(competition, optics)[:18]:
        delta = model.pos - vertex
        cosine_numerator = -np.einsum("ij,ij->i", delta, model.normals)
        distance2 = np.sum(delta * delta, axis=1)
        along = grid @ delta.T
        for energy in energies:
            angles, density, offset = angular_profile(energy)
            distance = np.sqrt(np.maximum(distance2 - 2.0 * offset * along + offset**2, 1.0))
            angle = np.clip((along - offset) / distance, -1.0, 1.0)
            incident = (cosine_numerator + offset * surface) / distance
            raw = (
                model.response
                * projected_aperture(incident)
                / (distance**2 + 10000.0)
                * np.interp(angle, angles, density)
            )
            flux = raw.sum(1)
            keep = flux >= 4.0
            if not np.any(keep):
                continue
            unit = raw[keep] / flux[keep, None]
            for j, (slope, weight, shapes, inverse) in enumerate(statistics):
                score = -(unit @ slope)
                cross = (unit * weight) @ shapes.T
                curvature = np.maximum((unit * unit) @ weight - ((cross @ inverse) * cross).sum(1), 1.0e-5)
                amplitude = np.clip(score / curvature, 0.0, 1.30 * flux[keep])
                gain = amplitude * (score - 0.5 * amplitude * curvature)
                order = np.argsort(gain)[-3:][::-1]
                proposals[j].extend(
                    (float(gain[i]), vertex, grid[keep][i], energy) for i in order if gain[i] > 2.0
                )
    output = []
    for pool in proposals:
        recent = []
        for _, vertex, axis, energy in sorted(pool, key=lambda r: -r[0]):
            if any(np.linalg.norm(vertex - v) < 125.0 and axis @ a > 0.94 for v, a in recent):
                continue
            recent.append((vertex, axis))
            if not any(
                np.linalg.norm(vertex - p.birth) < 75.0 and axis @ p.axis > 0.975 and energy == p.energy
                for p in output
            ):
                light = render(model, vertex, axis, energy)
                if light is not None and light[0][1].sum() >= 4.0:
                    output.append(Shower(vertex, axis, energy, light[0][1], light[1][1]))
            if len(recent) >= 5:
                break
    return output


def profile(model, shapes, starts, upper):
    def objective(x):
        ll, slope = model.obs.score(model.total * (x @ shapes), model.ix)
        return -ll / model.total, shapes @ slope

    bounds = np.column_stack([np.full(len(shapes), 1.0e-8), upper / model.total])
    init = np.clip(starts / model.total, *bounds.T)
    solution = minimize(
        objective,
        init,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=31, ftol=4e-10, gtol=1e-6),
    )
    if not np.isfinite(solution.fun):
        raise ArithmeticError("Non-finite profiled charge")
    return float(-solution.fun * model.total), solution.x * model.total


def fit_pool(model, template, sources):
    base = template.shapes
    budget = np.full(len(base), 1.0e6)
    value, amps = profile(model, base, template.amplitudes, budget)
    wanted = []
    if not sources:
        return wanted, amps
    yield_ = np.asarray([s.yield_pe for s in sources])
    fields = np.stack([s.rates for s in sources]) / yield_[:, None]
    for _ in range(3):
        mean = amps @ base
        _, slope = model.obs.score(mean, model.ix)
        fisher = (fields**2) @ (1.0 / np.maximum(mean, 0.08))
        norm = np.clip((-fields @ slope) / np.maximum(fisher, 1.0e-9), 1.0e-4, 1.30 * yield_)
        proposals = [
            -np.inf
            if any(
                np.linalg.norm(p.birth - sources[j].birth) < 130.0 and p.axis @ sources[j].axis > 0.94
                for j in wanted
            )
            else model.obs.score(mean + rate * shape, model.ix)[0]
            for p, rate, shape in zip(sources, norm, fields, strict=True)
        ]
        finalists = []
        for i in np.argsort(proposals)[-3:][::-1]:
            if proposals[i] <= value + 1.0:
                continue
            shape = np.vstack([base, fields[i]])
            ceiling = np.r_[budget, 1.30 * yield_[i]]
            ll, a = profile(model, shape, np.r_[amps, norm[i]], ceiling)
            finalists.append((ll, a, i, shape, ceiling))
        if not finalists:
            break
        ll, a, i, shape, ceiling = max(finalists, key=lambda r: r[0])
        if ll < value + 16.0 or a[-1] < 4.0:
            break
        wanted.append(int(i))
        value, amps, base, budget = ll, a, shape, ceiling
    return wanted, amps


def refit_geometry(competition, geometry, t, background, *, retain_seed_spans=False):
    from analog_optical import Analog, contain_finite, endpoint

    try:
        model = Analog(competition.event, geometry, background, "finite", 8)
        if retain_seed_spans:
            tracks = model.x[: model.k * model.size].reshape(model.k, model.size)
            for i, length in enumerate(geometry.get("lengths_mm", ())):
                if length is not None:
                    tracks[i, :3] = np.asarray(geometry["vertices"][i]) * 0.001
                    tracks[i, 6] = length * 0.001
                    contain_finite(tracks[i])
    except (ArithmeticError, AssertionError, ValueError):
        return None
    x, _ = endpoint(model, 34)
    if x is None:
        return None
    coarse = model.evaluate(x)[0]
    model.order = 24
    model.last = None
    exact = model.evaluate(x)[0]
    if abs(exact - coarse) > 0.055:
        model.x = x
        x, _ = endpoint(model, 18)
        if x is None:
            return None
        exact = model.evaluate(x)[0]
        model.order = 48
        model.last = None
        if abs(model.evaluate(x)[0] - exact) > 0.055:
            return None
    fitted = model.geometry(x)
    offset = model.size * model.k
    fitted["weights"] = softmax(np.r_[0.0, x[offset:-1]])
    return optical_template(competition.model, fitted, t.label + "_refined")


def photon_trial(competition, geometry, template, sources, indices, amps, *, refine):
    count = len(template.core_groups)
    active = [sources[i] for i in indices]
    background = amps[0] * template.shapes[0]
    for p, amount in zip(active, amps[count + 1 :], strict=True):
        background += amount / p.yield_pe * p.rates
    core = (
        refit_geometry(competition, geometry, template, background / background.sum()) if refine else template
    )
    if core is None:
        return None
    amounts = np.r_[core.amplitudes if refine else amps[: count + 1], amps[count + 1 :]]
    budget = {count + 1 + i: 1.30 * p.yield_pe for i, p in enumerate(active)}
    fields = np.vstack([core.shapes, *(p.rates / p.yield_pe for p in active)])
    clocks = np.column_stack([core.center, *(p.arrival for p in active)])
    return Template(
        f"sparse_soft_{core.label}_{len(active)}",
        fields,
        clocks,
        amounts,
        core.core_groups,
        np.arange(len(amounts) - 1),
        core.dimensions + 6 * len(active),
        budget,
    )


def measured(competition, template, sources, indices, *, refine):
    if template is None:
        return None
    result = competition.fit(template, iterations=28, physical_coverage=True)
    if result is not None:
        result.update(
            soft_sources=[
                dict(energy=p.energy, birth_mm=p.birth, axis=p.axis) for p in (sources[i] for i in indices)
            ],
            sparse_geometry_refined=refine,
        )
    return result


def subset(geometry, template, remove):
    k = len(template.core_groups)
    keep = np.flatnonzero(np.arange(k) != remove)
    result = {
        name: [data[i] for i in keep]
        for name, data in geometry.items()
        if name in ("vertices", "axes", "openings_deg", "widths_deg", "lengths_mm")
    }
    result["weights"] = np.r_[template.amplitudes[0], template.amplitudes[keep + 1]]
    return result


def with_counted_alternative(competition, lower, upper, geometry, fitted):
    """Retest the potentially visible ring with the same modeled soft light."""
    from shared_scoring import match_charge_shares

    if len(upper.core_groups) != len(lower.core_groups) + 1:
        return []
    higher = competition.fit(upper, iterations=20)
    if higher is None:
        return []
    shares = fitted["fractions"].get("prompt", fitted["fractions"]["timed"])
    previous = higher["fractions"].get("prompt", higher["fractions"]["timed"])
    observed, new, similarity = match_charge_shares(shares, previous[:, 1:].T, competition.model.q)
    positions = list(range(len(upper.core_groups)))
    lost = [j for j in positions if j not in set(new[similarity >= 0.42])]
    if not lost or len(observed) != len(lower.core_groups):
        return []
    outcomes = []
    lengths = geometry.get("lengths_mm", [0.0] * len(upper.core_groups))
    for j in lost:
        (ring,) = upper.core_groups[j]
        shape = np.vstack([lower.shapes, upper.shapes[ring]])
        center = np.column_stack([lower.center, upper.center[:, ring - 1]])
        estimate = np.r_[fitted["ring_amplitudes"], max(5.0, higher["ring_amplitudes"][ring])]
        timings = np.r_[lower.time_groups, lower.time_groups.max() + 1]
        labels = (*lower.core_groups, (len(lower.amplitudes),))
        added_parameters = 8 + bool(lengths[j] and lengths[j] > 0.1)
        trial = Template(
            "sparse_readded_core",
            shape,
            center,
            estimate,
            labels,
            timings,
            lower.dimensions + added_parameters,
            lower.limits,
        )
        rival = competition.fit(trial, iterations=28, physical_coverage=True)
        if rival is not None:
            outcomes.append(rival)
    return outcomes


def compare(competition, usable, count, *, matched_rivals=False):
    model = competition.model
    optics = [r for r in usable if max(1, count - 1) <= len(r[2].core_groups) <= count + 1]
    seeds = [(g, t) for _, g, t in optics]
    reference = next(((g, t) for g, t in seeds if len(t.core_groups) == count), None)
    if reference is not None and count > 1:
        g, t = reference
        for j in range(count):
            trimmed = subset(g, t, j)
            new = optical_template(model, trimmed, f"trimmed_{count}_{j}")
            if new is not None:
                seeds.append((trimmed, new))
    if not seeds:
        return [], 0
    sources = nominate_pool(competition, optics, seeds)
    fitted = {}
    for geometry, template in seeds:
        indices, amps = fit_pool(model, template, sources)
        basis = photon_trial(competition, geometry, template, sources, indices, amps, refine=False)
        result = measured(competition, basis, sources, indices, refine=False)
        if result is None:
            continue
        rank = (
            2.0 * (result["charge"] + 0.35 * (result["joint"] - result["charge"]))
            - np.log(len(model.active)) * result["dimension"]
        )
        count = result["count"]
        if count not in fitted or rank > fitted[count][0]:
            fitted[count] = (rank, result, geometry, template, indices, amps, basis)
    output = []
    for _, scored, geometry, template, indices, amps, basis in fitted.values():
        output.append(scored)
        refined = photon_trial(competition, geometry, template, sources, indices, amps, refine=True)
        guess = measured(competition, refined, sources, indices, refine=True)
        if guess is not None:
            output.append(guess)
        if (
            matched_rivals
            and reference is not None
            and indices
            and scored["count"] + 1 == len(reference[1].core_groups)
        ):
            high_geometry, upper = reference
            output.extend(with_counted_alternative(competition, basis, upper, high_geometry, scored))
            if guess is not None:
                output.extend(with_counted_alternative(competition, refined, upper, high_geometry, guess))
        if indices:
            _, flat = profile(
                model, template.shapes, template.amplitudes, np.full(len(template.shapes), 1.0e6)
            )
            diffuse = photon_trial(competition, geometry, template, sources, [], flat, refine=True)
            control = measured(competition, diffuse, sources, [], refine=True)
            if control is not None:
                control["label"] = control["label"].replace("sparse_soft_", "refined_no_soft_")
                output.append(control)
    return output, len(sources)
