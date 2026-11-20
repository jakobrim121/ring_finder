"""Place electron bends using material physics, without fitted breakpoints.

For a short segment of length h, a Brownian direction with scattering power
T has mean angular approximation error proportional to T*h. Weighting by
the emitted light w gives a total discretization cost integral(w*T/n ds),
where n is the number of segments per unit length. At fixed segment count,
its variational minimum is n proportional to sqrt(w*T).

The monitor uses the same water stopping, scattering and Frank--Tamm spectrum
as the source. It concentrates resolution where directions change rapidly
and light is still emitted. No event, truth or beam information enters this
table. This is an approximation mesh, not a stochastic transport trajectory.
"""

from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator, PPoly
from scipy.special import roots_legendre
from material import material
from spectrum import moment_table


@lru_cache(None)
def tables(order=12):
    m = material("electron")
    spectral_grid, coefficients, _, _ = moment_table()
    light = PPoly(coefficients, spectral_grid, extrapolate=False)
    scattering = m["scatter_at"].derivative()

    def monitor(remaining):
        energy = m["energy_at"](remaining)
        beta2 = 1 - (m["mass"] / (energy + m["mass"])) ** 2
        # At zero visible range the wavelength support vanishes. Rounding
        # beta² at that endpoint must not request spectral extrapolation.
        beta2 = np.clip(beta2, spectral_grid[0], spectral_grid[-1])
        return np.sqrt(np.maximum(light(beta2), 0) * np.maximum(-scattering(remaining), 0))

    grid = m["distance"]
    nodes, weights = roots_legendre(order)
    half = np.diff(grid) / 2
    samples = (grid[1:] + grid[:-1])[:, None] / 2 + half[:, None] * nodes
    increments = half * (monitor(samples) @ weights)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    assert np.all(np.diff(cumulative) > 0)
    forward = CubicHermiteSpline(grid, cumulative, monitor(grid), extrapolate=False)
    inverse = PchipInterpolator(cumulative, grid, extrapolate=False)
    return forward, inverse


def boundaries(log_range_m, pieces, order=12):
    """Remaining ranges in mm, with derivatives with respect to log range."""
    ran = 1000 * np.exp(log_range_m)
    forward, inverse = tables(order)
    fractions = 1 - np.arange(1, pieces) / pieces
    coordinates = fractions * forward(ran)
    interior = inverse(coordinates)
    derivative = inverse.derivative()(coordinates) * fractions * forward.derivative()(ran) * ran
    # Exact endpoints avoid interpolation and derivative singularities there.
    return np.r_[ran, interior, 0.0], np.r_[ran, derivative, 0.0]
