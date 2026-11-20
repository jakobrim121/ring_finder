"""Render a finite event hypothesis using a small transported cascade table."""

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def material_table():
    path = Path(__file__).with_name("electron_cascade_table_v1.npz")
    with np.load(path, allow_pickle=False) as data:
        return tuple(data[k] for k in ("energies", "angles", "emission", "shifts"))


def render(model, birth, axis, energy):
    from material import material
    from timed_charged_light import projected_aperture
    from wctering.likelihood.geometry import NORMALS, UPPER
    from wctering.seeding.coherent_timing import C, N_GROUP

    energies, cosines, temp, offsets = material_table()
    j = np.clip(np.searchsorted(energies, energy), 1, len(energies) - 1)
    weight = np.clip((energy - energies[j - 1]) / (energies[j] - energies[j - 1]), 0.0, 1.0)
    # Measured Cherenkov light has no charge-sign discrimination. Use the
    # equal e-/e+ mean of the analytic water transport tables.
    density = (temp[:, j - 1] * (1.0 - weight) + temp[:, j] * weight).mean(0)
    spatial = (offsets[:, :, j - 1] * (1.0 - weight) + offsets[:, :, j] * weight).mean(0)
    table = material("electron")
    if energy < table["energy_at"](table["margin_mm"]):
        density[0] = 0.0
    centers = birth + np.maximum(spatial[0], 0.0)[:, None] * axis
    # Infinite-water higher generations are represented by their moments.
    # Restrict this branch to showers whose centroids remain within the tank.
    if np.any(centers @ NORMALS.T >= 1000.0 * UPPER):
        return None
    distance = model.pos[None] - centers[:, None]
    span = np.maximum(np.linalg.norm(distance, axis=-1), 1.0)
    rays = distance / span[:, :, None]
    polar = rays @ axis
    mu = -np.einsum("pij,ij->pi", rays, model.normals)
    aperture = projected_aperture(mu) / (span**2 + 10000.0)
    radiance = np.asarray([np.interp(x, cosines, y) for x, y in zip(polar, density, strict=True)])
    rates = model.response * aperture * radiance
    times = (N_GROUP * span + spatial[1, :, None]) / C
    return rates, times
