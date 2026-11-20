"""Score charge-only starts without forming a receiver Jacobian."""

import numpy as np

import source
import analytic_knockon
from chain_sources import paths
from fast_csda import integrate as integrate_values
from material import material
from vector_knockon import integrate as integrate_knockon


def receiver_values(positions, normals, x, order, particle, *, integrate=integrate_values):
    """Use the same transport panels and integration order as source.raw."""
    water = material(particle)
    x = source.bounded_path(x, water["maximum_range_mm"])
    ran = 1000.0 * x[9]
    travelled = x[10]
    if travelled > 0.0:
        cut = (1.0 - water["margin_mm"] / ran) / travelled
    else:
        cut = float(ran >= water["margin_mm"])
    cut = float(np.clip(cut, 0.0, 1.0))
    base, weight = source.rule(order)
    bounds = source.material_panels(x, cut, water)
    parts = []
    for low, high in ((0.0, cut), (cut, 1.0)):
        if low == high:
            parts.append(np.zeros(len(positions)))
            continue
        edges = bounds[(bounds >= low) & (bounds <= high)]
        nodes = np.concatenate([a + (b - a) * base for a, b in zip(edges[:-1], edges[1:])])
        weights = np.concatenate([(b - a) * weight for a, b in zip(edges[:-1], edges[1:])])
        path = source.trajectory(x, nodes)
        light = source.states(
            x,
            nodes,
            water["distance"],
            water["energy_at"].c,
            water["scatter_at"].c,
            *source.angular_rule(),
            False,
            water["mass"],
            water["index"],
            *source.moment_table(),
            water["margin_mm"],
        )
        value = integrate(positions, normals, x, nodes, weights, path, light)
        if value.sum() < 1e-250:
            value = source.integrate_reference(positions, normals, x, nodes, weights, path, light)[0]
        parts.append(value)
    return np.asarray(parts)


def secondaries(positions, normals, path, particle, order):
    """Render the original knock-on charge without its unused derivatives."""
    if particle == "electron":
        return np.zeros(len(positions))
    water = material(particle)
    path = analytic_knockon.bounded_path(path, water["maximum_range_mm"])
    nodes, weights = analytic_knockon.roots_legendre(order)
    u = (nodes + 1.0) / 2.0
    points, axes, jp, jt = source.trajectory(path, u)
    remaining = 1000.0 * path[9] * (1.0 - path[10] * u)
    energy = water["energy_at"](remaining)
    gamma = 1.0 + energy / water["mass"]
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    opening = np.arccos(np.clip(1.0 / (water["index"] * beta), -1.0, 1.0))
    length = 1000.0 * path[9] * path[10]
    saved = analytic_knockon.table()
    return integrate_knockon(
        positions,
        normals,
        points,
        axes,
        jp,
        jt,
        opening,
        weights / 2.0,
        length,
        saved["opening_rad"],
        saved["polar_rad"],
        analytic_knockon.coefficients(),
    )


def charge_state(model, x, *, knockon=False, integrate=integrate_values):
    """Match Bare.__call__'s order of PE summation for seed tie-breaking."""
    pieces = np.empty((len(model.particles), 2, len(model.pos)))
    for index, ((path, _), particle) in enumerate(zip(paths(x[: model.ng]), model.particles)):
        value = receiver_values(model.pos, model.normals, path, model.order, particle, integrate=integrate)
        pieces[index] = (1000.0 * path[9] * path[10]) * value
        if knockon and model.delta_scale:
            pieces[index, 1] += model.delta_scale * secondaries(
                model.pos, model.normals, path, particle, model.order
            )
    response, background = np.exp(x[-2:])
    means = response * pieces.sum((0, 1)) + background * model.bg
    log_likelihood = model.obs.score(means, model.ix)[0]
    return log_likelihood, means, response * pieces[:, 0]


def log_likelihood(model, x, *, knockon=False, integrate=integrate_values):
    return charge_state(model, x, knockon=knockon, integrate=integrate)[0]
