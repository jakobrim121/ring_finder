"""Positive quadrature of the documented water dispersion and PMT QE shape.

This is a detector-response hypothesis, not an event-derived calibration.
The WCTE R14374 table in the available production source matches these knots.
Constant QE/collection factors cancel because total intensity is profiled.
Absorption, glass refraction and finite aperture are not included here.
"""

from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import CubicHermiteSpline


@lru_cache(None)
def spectrum(order=8):
    data = json.loads((Path(__file__).with_name("wcsim_water_optics.json")).read_text())
    water = data["water"]
    pmt = data["pmt_r12199_02"]
    energy = np.asarray(water["energy_eV"])
    index = np.asarray(water["phase_refractive_index"])
    wavelengths = np.asarray(pmt["wavelength_nm"])
    qe = np.asarray(pmt["qe_shape_before_constant_correction"])
    hc = data["constants"]["hc_standard_eV_nm"]
    conversion = data["constants"]["hc_wcsim_stacking_QE_eV_nm"] / hc
    zero = wavelengths[0] - qe[0] * (wavelengths[1] - wavelengths[0]) / (qe[1] - qe[0])
    low = max(hc / energy.max(), zero / conversion, 200.0 / conversion)
    high = min(hc / energy.min(), 660.0 / conversion)
    knots = np.unique(np.r_[low, high, hc / energy, wavelengths / conversion])
    knots = knots[(knots >= low) & (knots <= high)]
    x, w = roots_legendre(order)
    half = np.diff(knots) / 2
    mid = (knots[1:] + knots[:-1]) / 2
    wave = (mid[:, None] + half[:, None] * x).ravel()
    weights = (half[:, None] * w).ravel()
    corrected = wave * conversion
    efficiency = np.interp(corrected, wavelengths, qe)
    efficiency = np.where(
        corrected < wavelengths[0],
        qe[0] + (corrected - wavelengths[0]) * (qe[1] - qe[0]) / (wavelengths[1] - wavelengths[0]),
        efficiency,
    )
    weights *= np.maximum(efficiency, 0) / wave**2
    weights /= weights.sum()
    indices = np.interp(hc / wave, energy, index)
    return np.ascontiguousarray(indices), np.ascontiguousarray(weights)


def maximum_index():
    data = json.loads((Path(__file__).with_name("wcsim_water_optics.json")).read_text())
    # The integration's smallest supported wavelength determines threshold.
    # Its endpoint QE is zero; using the supremum avoids a grid-dependent cut.
    p = data["pmt_r12199_02"]
    wave = np.asarray(p["wavelength_nm"])
    qe = np.asarray(p["qe_shape_before_constant_correction"])
    zero = wave[0] - qe[0] * (wave[1] - wave[0]) / (qe[1] - qe[0])
    energy = data["constants"]["hc_wcsim_stacking_QE_eV_nm"] / zero
    return float(np.interp(energy, data["water"]["energy_eV"], data["water"]["phase_refractive_index"]))


def continuous_moments(beta2, order=24):
    """Integrate three unnormalized angular moments and their derivatives.

    Frank--Tamm spectral weight is uniform in photon energy before QE. Split
    at every material/QE knot and at the *continuous* Cherenkov threshold.
    The squared local coordinate removes the square-root endpoint from the
    angular moments. Unlike fixed wavelength clipping, this produces no
    artificial jumps when a wavelength node turns on.
    """
    data = json.loads(Path(__file__).with_name("wcsim_water_optics.json").read_text())
    water = data["water"]
    pmt = data["pmt_r12199_02"]
    energies = np.asarray(water["energy_eV"])
    indices = np.asarray(water["phase_refractive_index"])
    wave = np.asarray(pmt["wavelength_nm"])
    qe = np.asarray(pmt["qe_shape_before_constant_correction"])
    hc = data["constants"]["hc_wcsim_stacking_QE_eV_nm"]
    zero = wave[0] - qe[0] * (wave[1] - wave[0]) / (qe[1] - qe[0])
    low = max(energies[0], hc / 660.0)
    high = min(energies[-1], hc / zero)
    knots = np.unique(np.r_[low, high, energies, hc / wave])
    knots = knots[(knots >= low) & (knots <= high)]
    node, weight = roots_legendre(order)
    node = (node + 1) / 2
    weight *= node
    beta2 = np.atleast_1d(beta2)
    cut = np.interp(1 / np.sqrt(beta2), indices, energies)
    left = np.maximum(knots[:-1][None, :], cut[:, None])
    right = knots[1:][None, :]
    span = np.maximum(right - left, 0.0)
    energy = left[:, :, None] + span[:, :, None] * node**2
    index = np.interp(energy, energies, indices)
    wavelength = hc / energy
    efficiency = np.interp(wavelength, wave, qe)
    efficiency = np.where(
        wavelength < wave[0],
        qe[0] + (wavelength - wave[0]) * (qe[1] - qe[0]) / (wave[1] - wave[0]),
        efficiency,
    )
    quadrature = span[:, :, None] * weight * np.maximum(efficiency, 0.0)
    value = np.maximum(1 - 1 / (index * index * beta2[:, None, None]), 0.0)
    angle = np.arcsin(np.sqrt(value))
    derivative = 1 / (index * index * beta2[:, None, None] ** 2)
    # value*d(angle)/d(beta²) is regular at threshold; write it that way.
    angular_term = 0.5 * derivative * np.sqrt(value / np.maximum(1 - value, 1e-30))
    moments = np.stack([value, value * angle, value * angle * angle], axis=-1)
    slopes = np.stack(
        [
            derivative,
            derivative * angle + angular_term,
            derivative * angle * angle + 2 * angle * angular_term,
        ],
        axis=-1,
    )
    return (quadrature[:, :, :, None] * moments).sum((1, 2)), (quadrature[:, :, :, None] * slopes).sum((1, 2))


@lru_cache(None)
def moment_table():
    """Data-independent Hermite table; derivatives belong to the same spline."""
    threshold = 1 / maximum_index() ** 2
    fraction = np.unique(np.r_[0.0, np.geomspace(1e-10, 0.025, 1537), np.linspace(0, 1, 8193)])
    grid = threshold + (1 - threshold) * fraction
    values = []
    slopes = []
    for start in range(0, len(grid), 128):
        value, slope = continuous_moments(grid[start : start + 128])
        values.append(value)
        slopes.append(slope)
    values = np.concatenate(values)
    slopes = np.concatenate(slopes)
    # The common scale cancels in normalized shapes, but keep FT dimensionless.
    normalization = values[-1, 0] / sum(w * (1 - 1 / n**2) for n, w in zip(*spectrum(32)))
    return (
        grid,
        *[
            np.ascontiguousarray(
                CubicHermiteSpline(
                    grid, values[:, j] / normalization, slopes[:, j] / normalization, extrapolate=False
                ).c
            )
            for j in range(3)
        ],
    )
