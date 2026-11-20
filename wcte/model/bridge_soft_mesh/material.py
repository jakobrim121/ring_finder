"""Common water energy-loss coordinates, with no simulated-event inputs.

Electrons reuse the pinned ESTAR table. Heavy particles use the PDG Bethe
expression with finite-mass maximum transfer and the liquid-water Sternheimer
density correction. These are electromagnetic mean ranges; hadronic energy
losses and decays are separate interactions, not hidden in a fitted range.
"""

from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.special import roots_legendre
from electron_material import material as electron_material
from spectrum import maximum_index

MASS = 0.51099895
INDEX = 1.344
MASSES = dict(muon=105.6583755, pion=139.57039, proton=938.27208816)


def bounded_path(x, maximum_range_mm):
    """Correct floating-point drift at a closed material boundary.

    Cross-CPU table sums and metre/mm conversions can put the endpoint a few
    ulps outside its table. Real excursions still raise; no extrapolation is
    enabled. The returned copy keeps caller parameters and archived fits intact.
    """
    x = np.asarray(x, float)
    r, f = 1000 * x[9], x[10]
    # Summing 4097 table intervals on ARM and x86 differed by nine epsilons
    # in the proton endpoint (2.4e-11 mm). Allow sixteen there, but retain
    # the original eight-epsilon check on the dimensionless path fraction.
    tolerance = 8 * np.finfo(float).eps
    if not (0 < r <= maximum_range_mm * (1 + 2 * tolerance) and -tolerance <= f <= 1 + tolerance):
        raise ValueError("Unphysical range")
    if r > maximum_range_mm or not 0 <= f <= 1:
        x = x.copy()
        if r > maximum_range_mm:
            x[9] = np.nextafter(maximum_range_mm / 1000, 0.0)
        x[10] = np.clip(f, 0.0, 1.0)
    return x


def stopping(energy, mass):
    gamma = 1 + np.asarray(energy) / mass
    beta2 = 1 - 1 / gamma**2
    bg = np.sqrt(beta2) * gamma
    x = np.log10(bg)
    delta = np.where(
        x < 0.24, 0.0, 2 * np.log(10) * x - 3.5017 + 0.0912 * np.maximum(2.8004 - x, 0.0) ** 3.4773
    )
    transfer = 2 * MASS * beta2 * gamma**2 / (1 + 2 * gamma * MASS / mass + (MASS / mass) ** 2)
    # K in MeV cm²/mol, Z/A in mol/g, rho=1 g/cm³; /10 gives MeV/mm.
    return (
        0.307075
        * 0.55509
        / (10 * beta2)
        * (0.5 * np.log(2 * MASS * beta2 * gamma**2 * transfer / (78e-6) ** 2) - beta2 - 0.5 * delta)
    )


def scattering_power(energy, mass):
    momentum2 = energy * (energy + 2 * mass)
    beta2 = momentum2 / (energy + mass) ** 2
    # Projected-plane local Gaussian power, not per-step Highland logarithms.
    return 13.6**2 / (beta2 * momentum2 * 360.8)


@lru_cache(None)
def material(particle="electron"):
    if particle == "electron":
        return dict(electron_material(), mass=MASS, index=INDEX)
    mass = MASSES[particle]
    reference_threshold = mass * (1 / np.sqrt(1 - 1 / INDEX**2) - 1)
    threshold = mass * (1 / np.sqrt(1 - 1 / maximum_index() ** 2) - 1)
    grid = np.unique(np.r_[np.geomspace(threshold, 3000.0, 4097), reference_threshold + 30])
    node, weight = roots_legendre(12)
    mid = 0.5 * (grid[:-1] + grid[1:])
    half = 0.5 * np.diff(grid)
    energy = mid[:, None] + half[:, None] * node
    stop = stopping(energy, mass)
    increments = half * np.sum(weight / stop, axis=1)
    scatter_increments = half * np.sum(weight * scattering_power(energy, mass) / stop, axis=1)
    distance = np.r_[0.0, np.cumsum(increments)]
    scatter = np.r_[np.cumsum(scatter_increments[::-1])[::-1], 0.0]
    energy_at = CubicHermiteSpline(distance, grid, stopping(grid, mass), extrapolate=False)
    scatter_at = CubicHermiteSpline(distance, scatter, -scattering_power(grid, mass), extrapolate=False)
    range_at = PchipInterpolator(grid, distance, extrapolate=False)
    return dict(
        energy=grid,
        distance=distance,
        scatter=scatter,
        energy_at=energy_at,
        scatter_at=scatter_at,
        range_at=range_at,
        margin_mm=float(range_at(reference_threshold + 30)),
        maximum_range_mm=float(distance[-1]),
        mass=mass,
        index=INDEX,
    )
