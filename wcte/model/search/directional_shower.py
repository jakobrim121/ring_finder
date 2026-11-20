"""Separate prompt cores from the directional light of short stopping tracks."""

from functools import lru_cache

import numpy as np
from scipy.special import i0e, roots_legendre

from material import material
from timed_charged_light import projected_aperture
from wctering.likelihood.geometry import NORMALS, UPPER
from wctering.likelihood.water_boundary import clearance
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


@lru_cache(maxsize=8)
def emission(fraction, order):
    """Average Cherenkov light over the slowing and diffusion of a soft lepton."""
    water = material("electron")
    edge = water["energy_at"](water["margin_mm"])
    length = float(water["range_at"](fraction * edge))
    x, weights = roots_legendre(order)
    s = 0.5 * length * (1.0 + x)
    remaining = length - s
    energy = water["energy_at"](remaining)
    beta = np.sqrt(1.0 - (1.0 + energy / water["mass"]) ** -2)
    cosine = np.clip(1.0 / (N_PHASE * beta), 0.0, 1.0)
    variance = water["scatter_at"](remaining) - water["scatter_at"](length)
    # The von Mises–Fisher tangent law has projected angular variance
    # 1/kappa in the small-angle limit and tends to isotropic as kappa→0.
    concentration = 1.0 / (np.maximum(variance, 0.0) + np.deg2rad(1.8) ** 2)
    # Under rotational diffusion the mean axial velocity is attenuated by
    # exp(-variance). A trapezoid estimates the mean forward displacement.
    z = np.r_[0.0, s]
    attenuation = np.r_[1.0, np.exp(-np.maximum(variance, 0.0))]
    along = np.cumsum(0.5 * np.diff(z) * (attenuation[:-1] + attenuation[1:]))
    inverse_beta = np.r_[1.0 / np.sqrt(1.0 - (1.0 + fraction * edge / water["mass"]) ** -2), 1.0 / beta]
    flight = np.cumsum(0.5 * np.diff(z) * (inverse_beta[1:] + inverse_beta[:-1])) / C
    weight = 0.5 * length * weights * (1.0 - cosine**2)
    return along, flight, cosine, concentration, weight


def angular_density(cosine, opening_cosine, concentration):
    """Integrate a Cherenkov cone around a rotationally diffusing axis."""
    # Averaging exp(kappa * axis·direction) over the Cherenkov azimuth
    # gives I0(kappa * sin(alpha) * sin(theta)). Use the scaled Bessel
    # function and combine its exponent with the vMF normalizer.
    sine_product = np.sqrt(np.maximum(1.0 - cosine**2, 0.0) * np.maximum(1.0 - opening_cosine**2, 0.0))
    exponent = concentration * np.minimum(cosine * opening_cosine + sine_product - 1.0, 0.0)
    denominator = -np.expm1(-2.0 * concentration)
    return concentration / (2.0 * np.pi * denominator) * np.exp(exponent) * i0e(concentration * sine_product)


def shower(model, birth, axis, fraction, order):
    """Return the compact shower's PMT response and relative arrival moment."""
    s, flight, cosine, concentration, weight = emission(fraction, order)
    delta = model.pos[:, None] - (1000.0 * birth + s[:, None] * axis)
    distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-8)
    rays = delta / distance[:, :, None]
    mu = -np.einsum("ijk,ik->ij", rays, model.normals)
    aperture = projected_aperture(mu)
    polar = np.clip(rays @ axis, -1.0, 1.0)
    density = angular_density(polar, cosine, concentration)
    light = weight * density * np.maximum(aperture / (distance**2 + 10000.0), 1e-30)
    response = light.sum(1)
    arrival = flight + N_GROUP * distance[model.ix] / C
    fallback = N_GROUP * np.linalg.norm(model.pos[model.ix] - 1000.0 * birth, axis=1) / C
    time = np.divide(
        np.sum(light[model.ix] * arrival, axis=1),
        response[model.ix],
        out=fallback,
        where=response[model.ix] > 1e-30,
    )
    return response, time


def geometry_modes(path, length):
    """A compact secondary can be displaced along the measured light axis."""
    axis = path[3:6]
    for displacement in (-0.14, -0.04, 0.07):
        birth = path[:3] + displacement * axis
        end = birth + length * axis
        if min(UPPER - NORMALS @ birth) <= 1e-5 or min(UPPER - NORMALS @ end) <= 1e-5:
            continue
        if min(clearance(0.5 * (birth + end), axis, length)[0]) > 1e-5:
            yield birth, axis
