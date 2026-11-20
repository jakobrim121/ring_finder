"""Approximate the direct light of a gamma conversion and its radiative tail."""

from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import roots_legendre

from directional_shower import angular_density
from electron_material import table
from material import material
from timed_charged_light import projected_aperture
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


PAIR_FLIGHT_MM = 9.0 / 7.0 * 360.8


@lru_cache(maxsize=2)
def nodes(order):
    x, w = roots_legendre(order)
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=1)
def radiative_fraction():
    """Use the radiative share of electron stopping power from NIST ESTAR."""
    basic, _ = table()
    return PchipInterpolator(np.log(basic[:, 0]), basic[:, 2] / basic[:, 3])


@lru_cache(maxsize=192)
def emissions(energy, split=0.5, cascade=True, compton=False, by_arm=False):
    """Return position, flight time, angular variance, cone cosine and yield.

    The first pair loses energy through the measured total stopping power.
    Approximate its radiated energy by one additional, energy-conserving
    quadrature generation. The photon energy weights integrate the complete-
    screening bremsstrahlung energy spectrum. This deliberately averages
    over the fluctuations of deeper cascades and the small pair opening.
    """
    u, w = nodes(6)
    water = material("electron")
    mass = water["mass"]
    threshold = water["energy"][0]
    maximum = water["energy"][-1]
    soft = float(water["energy_at"](water["margin_mm"]))
    radial = radiative_fraction()

    def track(kinetic, inherited_scatter=0.0):
        length = water["range_at"](np.clip(kinetic, threshold, maximum))
        s = np.asarray(length)[..., None] * u
        k = water["energy_at"](length[..., None] - s)
        beta = np.sqrt(1.0 - (1.0 + k / mass) ** -2)
        cos = np.minimum(1.0 / (N_PHASE * beta), 1.0)
        scatter = np.maximum(
            water["scatter_at"](length[..., None] - s) - water["scatter_at"](length)[..., None], 0.0
        )
        # Water scattering shortens the mean forward displacement. A
        # single-node approximation to its integrated angular diffusion.
        position = s * np.exp(-0.5 * scatter - np.asarray(inherited_scatter)[..., None])
        initial_beta = np.sqrt(1.0 - (1.0 + np.asarray(kinetic) / mass) ** -2)
        flight = 0.5 * s * (1.0 / np.maximum(initial_beta[..., None], 1e-6) + 1.0 / beta) / C
        strength = length[..., None] * w * (1.0 - cos**2)
        return position, flight, scatter + np.asarray(inherited_scatter)[..., None], cos, strength, k >= soft

    if compton:
        # A forward, high-transfer recoil is a conservative visible-ring
        # competitor to a split pair. Conserve the remaining energy in
        # the outgoing photon and include its neutral flight below.
        recoil = split * 2.0 * energy**2 / (mass + 2.0 * energy)
        pair = np.array([recoil])
    else:
        pair = max(0.0, energy - 2.0 * mass) * np.array([split, 1.0 - split])
    # Keep above- and below-margin emission separate throughout.
    blocks = [np.stack(track(pair), axis=-1).reshape(-1, 6)]
    if cascade:
        fraction, weights = nodes(2)
        weights = weights * (4.0 / 3.0 - 4.0 / 3.0 * fraction + fraction**2)
        # Bin radiation by remaining electron energy. The power times
        # each bin width is the emitted energy available to its children.
        energy_nodes, energy_weights = nodes(3)
        width = np.maximum(pair - threshold, 0.0)
        incident = threshold + width[:, None] * energy_nodes
        distance = water["range_at"](np.clip(pair, threshold, maximum))[:, None] - water["range_at"](
            np.minimum(incident, maximum)
        )
        parent_spread = np.maximum(
            water["scatter_at"](water["range_at"](np.minimum(incident, maximum)))
            - water["scatter_at"](water["range_at"](np.clip(pair, threshold, maximum)))[:, None],
            0.0,
        )
        e_gamma = incident[..., None] * fraction
        radiated = width[:, None, None] * energy_weights[None, :, None] * weights
        radiated = radiated * radial(np.log(incident))[..., None]
        pair_rate = np.divide(radiated, e_gamma, out=np.zeros_like(radiated), where=e_gamma > 2.0 * mass)
        child_energy = np.maximum(0.5 * e_gamma - mass, threshold)
        # A representative exponential-flight quadrature preserves both
        # short and long transport. The water boundary culls escaped light.
        survival, probabilities = nodes(3)
        photon_length = -PAIR_FLIGHT_MM * np.log1p(-survival)
        advance, delay, spread, cosine, yield_weight, visible = track(child_energy, parent_spread[..., None])
        for length, probability in zip(photon_length, probabilities):
            position = advance + distance[..., None, None] * np.exp(-0.5 * parent_spread[..., None, None])
            position = position + length * np.exp(-parent_spread[..., None, None])
            flight = delay + (distance[..., None, None] + length) / C
            strength = yield_weight * (2.0 * probability * pair_rate[..., None])
            blocks.append(
                np.stack((position, flight, spread, cosine, strength, visible), axis=-1).reshape(-1, 6)
            )
        if compton and energy - recoil > 2.0 * mass:
            # This trial keeps the low-energy scattered photon collinear.
            # Its pair supplies only the remainder of the original energy.
            child = max(0.5 * (energy - recoil) - mass, threshold)
            advance, delay, spread, cosine, strength, visible = track(np.array([child]))
            for length, probability in zip(photon_length, probabilities):
                blocks.append(
                    np.stack(
                        (
                            advance + length,
                            delay + length / C,
                            spread,
                            cosine,
                            2.0 * probability * strength,
                            visible,
                        ),
                        axis=-1,
                    ).reshape(-1, 6)
                )
    result = np.vstack(blocks)
    if by_arm:
        # A true ring is supported by an individual charged trajectory.
        # Preserve the first conversion's lepton identities. Their
        # radiative descendants enter the unresolved response here.
        owner = np.full(len(result), -1.0)
        owner[: len(blocks[0])] = np.repeat(np.arange(len(pair)), len(u))
        result = np.column_stack([result, owner])
    return result[result[:, 4] > 0.0]


def light_and_time(model, photons, visible, distance, flight):
    """Collect light by energy margin and the hit-conditioned first moment."""
    light = np.stack((photons @ (1.0 - visible), photons @ visible))
    time = flight + N_GROUP * distance[model.ix] / C
    hit = photons[model.ix]
    norm = hit.sum(1)
    mean = np.divide((hit * time).sum(1), norm, out=time[:, 0].copy(), where=norm > 0.0)
    return light, mean


@lru_cache(maxsize=32)
def angular_table(energy, split, cascade, compton, by_arm=False):
    """Tabulate the smooth angular response of each integration node."""
    points = emissions(energy, split=split, cascade=cascade, compton=compton, by_arm=by_arm)
    spread, cone, strength = points[:, 2:5].T
    angle = np.linspace(0.0, np.pi, 4097)
    density = angular_density(np.cos(angle[:, None]), cone, 1.0 / (spread + np.deg2rad(2.0) ** 2))
    return angle, density * strength


def render_fast(
    model,
    point_mm,
    axis,
    energy,
    water_length_mm,
    *,
    cascade=True,
    split=0.5,
    compton=False,
    by_arm=False,
    separate_times=False,
):
    """Evaluate the same nodes using the axial geometry and table interpolation."""
    energy = float(energy)
    points = emissions(energy, split=split, cascade=cascade, compton=compton, by_arm=by_arm)
    keep = points[:, 0] < water_length_mm
    s, time, _, _, _, visible = points[keep, :6].T
    if not len(s):
        time_shape = (len(model.ix), 3) if by_arm and separate_times else len(model.ix)
        return np.zeros((3 if by_arm else 2, len(model.pos))), np.zeros(time_shape)
    delta = model.pos - point_mm
    along = delta @ axis
    transverse = delta - along[:, None] * axis
    radius = np.linalg.norm(transverse, axis=1)
    z = along[:, None] - s
    distance = np.maximum(np.hypot(radius[:, None], z), 1e-8)
    incidence = (-(delta * model.normals).sum(1)[:, None] + (model.normals @ axis)[:, None] * s) / distance
    angle, light = angular_table(energy, split, cascade, compton, by_arm)
    table = light[:, keep]
    step = angle[1]
    x = np.arctan2(radius[:, None], z) / step
    low = np.minimum(x.astype(int), len(angle) - 2)
    part = np.clip(x - low, 0.0, 1.0)
    index = np.arange(len(s))
    photons = (1.0 - part) * table[low, index] + part * table[low + 1, index]
    photons *= projected_aperture(incidence) / (distance**2 + 10000.0)
    if not by_arm:
        return light_and_time(model, photons, visible, distance, time)
    owner = points[keep, 6]
    cores = np.column_stack([(owner == j) * visible for j in range(2)])
    # Keep the approximation's conserved total, including the entire
    # radiative tail, regardless of the number of direct cores.
    light, mean_time = light_and_time(model, photons, visible, distance, time)
    charged = (photons @ cores).T
    spectra = np.vstack([charged, np.maximum(light.sum(0) - charged.sum(0), 0.0)])
    if not separate_times:
        return spectra, mean_time
    # The first-conversion e+/e- and their unresolved descendants have
    # different optical flight times. Averaging the routes would give
    # their earliest PE the mean clock of the entire shower.
    groups = np.column_stack([cores, ~cores.any(1)]).astype(float)
    flight = time + N_GROUP * distance[model.ix] / C
    pulse = (photons[model.ix] * flight) @ groups
    normalization = spectra[:, model.ix].T
    timed = np.broadcast_to(mean_time[:, None], normalization.shape).copy()
    np.divide(pulse, normalization, out=timed, where=normalization > 0.0)
    return spectra, timed
