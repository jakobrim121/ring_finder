"""Compare charged geometries using the earliest PE and calibrated charge."""

from functools import lru_cache
from contextvars import ContextVar
from contextlib import contextmanager

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar

from chain_sources import paths
from charge_starts import weighted_quantile
from gated_photon_time import evaluate, log_density
from material import material
from source import angular_rule, material_panels, moment_table, rule, states, trajectory
from timed_charged_light import projected_aperture
from wctering.seeding.coherent_timing import C, N_GROUP


_CLOCK_VALUE = ContextVar("physical_clock_value", default=False)


@contextmanager
def clock_values(enabled):
    """Use the matching value-only time likelihood during scalar clock fits."""
    token = _CLOCK_VALUE.set(enabled)
    try:
        yield
    finally:
        _CLOCK_VALUE.reset(token)


@lru_cache(maxsize=None)
def flight_integral(particle):
    """Antiderivative of inverse speed in the material's range coordinate."""
    water = material(particle)
    distance = water["distance"]
    energy = water["energy_at"](distance)
    gamma = 1.0 + energy / water["mass"]
    beta = np.sqrt(1.0 - gamma**-2)
    slope = -water["energy_at"].derivative()(distance) / (water["mass"] * (gamma * beta) ** 3)
    return CubicHermiteSpline(distance, 1.0 / beta, slope).antiderivative()


def wavefront(positions, normals, path, particle, order=3, *, spread=False):
    """Integrate distinct arrival moments above and below the energy margin."""
    water = material(particle)
    remaining = 1000.0 * path[9]
    length = remaining * path[10]
    cut = np.clip((remaining - water["margin_mm"]) / length, 0.0, 1.0)
    nodes, weights = rule(order)
    edges = material_panels(path, cut, water)
    spans = np.diff(edges)
    u = (edges[:-1, None] + spans[:, None] * nodes).ravel()
    weights = (spans[:, None] * weights).ravel()
    points, axes, _, _ = trajectory(path, u)
    emission = states(
        path,
        u,
        water["distance"],
        water["energy_at"].c,
        water["scatter_at"].c,
        *angular_rule(),
        False,
        water["mass"],
        water["index"],
        *moment_table(),
        water["margin_mm"],
    )
    endpoints = np.r_[0.0, u, 1.0]
    r = np.clip(remaining - length * endpoints, 0.0, water["maximum_range_mm"])
    if spread:
        primitive = flight_integral(particle)(r)
        flight = (primitive[0] - primitive[1:]) / C
    else:
        energy = water["energy_at"](r)
        inverse_beta = 1.0 / np.sqrt(np.maximum(1.0 - (1.0 + energy / water["mass"]) ** -2, 1e-8))
        flight = length / C * np.cumsum(np.diff(endpoints) * 0.5 * (inverse_beta[1:] + inverse_beta[:-1]))
    delta = positions[:, None, :] - points
    distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-8)
    ray = delta / distance[:, :, None]
    acceptance = projected_aperture(-np.einsum("ij,ikj->ik", normals, ray))
    polar = np.arccos(np.clip(np.einsum("ikj,kj->ik", ray, axes), -1.0, 1.0))
    angular = np.exp(-0.5 * (polar - emission[:, 0]) ** 2 / emission[:, 6])
    density = (length * weights * emission[:, 1]) * angular
    density *= acceptance / (distance**2 + 10000.0)
    time = flight[:-1] + N_GROUP * distance / C
    fallback = N_GROUP * np.linalg.norm(positions - 1000.0 * path[:3], axis=1) / C
    centers, strengths, variances = [], [], []
    for keep in (u <= cut, u > cut):
        local = density[:, keep]
        norm = local.sum(1)
        mean = np.divide((local * time[:, keep]).sum(1), norm, out=fallback.copy(), where=norm > 1e-30)
        centers.append(mean)
        strengths.append(norm)
        if spread:
            # The PMT sees a weighted range of emission positions. Convolve
            # its longitudinal variance with the measured time resolution.
            delta = time[:, keep] - mean[:, None]
            variances.append(
                np.divide((local * delta**2).sum(1), norm, out=np.zeros_like(norm), where=norm > 1e-30)
            )
    result = np.asarray(centers), np.asarray(strengths), float(flight[-1])
    return (*result, np.asarray(variances)) if spread else result


def components(
    event, model, state, *, separate_knockon=False, return_elapsed=False, spread=False, all_receivers=False
):
    """Use charge-fit core yields and separate modeled tails in the time law."""
    model(state["parameters"], True)
    _, light, cores, _ = model.primary_cache
    response, background = np.exp(state["parameters"][-2:])
    ix = slice(None) if all_receivers else model.ix
    positions = model.pos if all_receivers else event.pmt_positions_mm
    normals = model.normals[ix]
    means, strengths, variances = [], [], []
    offset = 0.0
    tracks = [p for p, _ in paths(state["parameters"][: model.ng])]
    for path, particle in zip(tracks, model.particles):
        time, rate, elapsed, *extra = wavefront(
            positions, normals, path, particle, order=8 if spread else 3, spread=spread
        )
        means.extend(time + offset)
        strengths.append(rate[1])
        if spread:
            variances.extend(extra[0])
        offset += elapsed
    core = cores[:, ix]
    remainder = np.maximum(light[ix] - core.sum(0), 0.0)
    if separate_knockon:
        # The charged track's own slowing tail has a known emission interval.
        # Use the hit-only timing integral for its approximate PE rate. The
        # charge fit still supplies the exact total tail and knock-on rate.
        # With no independent knock-on flight prediction, time that excess
        # as diffuse light instead of inheriting the slowing tail's clock.
        tails = np.asarray(strengths)
        tails *= np.minimum(1.0, remainder / np.maximum(tails.sum(0), 1e-300))
    else:
        # Keep the original moment-closure approximation for comparisons
        # against the frozen first-pulse pilot.
        tails = np.asarray(strengths)
        weights = np.divide(tails, tails.sum(0), out=np.zeros_like(tails), where=tails.sum(0) > 1e-30)
        tails = remainder * weights
    diffuse = background * model.bg[ix] + response * np.maximum(remainder - tails.sum(0), 0.0)
    rate = response * np.stack([core, tails], axis=1).reshape(-1, len(diffuse))
    result = np.column_stack([diffuse, rate.T]), np.column_stack(means)
    if spread:
        result = (*result, np.column_stack(variances))
    return (*result, offset) if return_elapsed else result


def profile(
    event, model, state, *, separate_knockon=False, precomputed=None, spread=False, timing_only=False
):
    """Score the timing with one relative clock for all serial sections."""
    gate = event.timing_selection["prompt_window_ns"]
    if gate is None:
        return None
    center = np.median(event.times_ns)
    time = event.times_ns - center
    low, high = np.asarray(gate) - center
    if np.any((time < low - 1e-9) | (time > high + 1e-9)):
        return None
    if precomputed is None:
        precomputed = components(event, model, state, separate_knockon=separate_knockon, spread=spread)
    rates, means, *extra = precomputed
    variance = extra[0] if extra else None
    weight = rates[:, 1:] / rates.sum(1, keepdims=True)
    weight *= np.minimum(event.charges_pe, 2.0)[:, None]
    if not weight.any():
        return None
    clock = weighted_quantile((time[:, None] - means).ravel(), weight.ravel(), [0.5])[0]
    q = np.maximum(event.charges_pe, 0.25)
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / q) / 2.355**2)
    timing_only = timing_only or _CLOCK_VALUE.get()

    def objective(offset):
        args = time, low, high, means + offset, rates, sigma, model.obs.offsets, model.obs.coefficients
        density = (
            log_density(*args, variance=variance) if timing_only else evaluate(*args, variance=variance)[0]
        )
        return -density.sum()

    result = minimize_scalar(
        objective, bounds=(clock - 4.0, clock + 4.0), method="bounded", options={"xatol": 0.02}
    )
    return dict(clock_ns=float(result.x), timing_ll=float(-result.fun))
