"""Independent water electron material approximation from official ESTAR data."""

from functools import lru_cache
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.special import roots_legendre
from spectrum import maximum_index

HERE = Path(__file__).resolve().parent
MASS = 0.51099895
INDEX = 1.344
X0_MM = 360.8
SCATTER_MEV = 15.0
REFERENCE_THRESHOLD = MASS * (1 / np.sqrt(1 - 1 / INDEX**2) - 1)
THRESHOLD = MASS * (1 / np.sqrt(1 - 1 / maximum_index() ** 2) - 1)


@lru_cache(maxsize=2)
def table():
    manifest = json.loads((HERE / "estar_manifest.json").read_text())
    for r in manifest["records"]:
        assert hashlib.sha256((HERE / r["path"]).read_bytes()).hexdigest() == r["sha256"]
    basic = np.loadtxt(HERE / "estar_water_basic.txt")
    long = np.loadtxt(HERE / "estar_water_long.txt")
    assert basic.shape == (81, 7) and long.shape == (97, 5)
    assert np.max(abs(basic[:, 1] + basic[:, 2] - basic[:, 3]) / basic[:, 3]) < 0.001
    return basic, long


def scattering_power(energy):
    momentum2 = energy * (energy + 2 * MASS)
    beta2 = momentum2 / (energy + MASS) ** 2
    return SCATTER_MEV**2 / (beta2 * momentum2 * X0_MM)


@lru_cache(maxsize=4)
def material(order=12, points=4097):
    _, data = table()
    spline = PchipInterpolator(np.log(data[:, 0]), np.log(data[:, 3] / 10), extrapolate=False)
    grid = np.unique(
        np.r_[
            np.geomspace(THRESHOLD, 500.0, points),
            REFERENCE_THRESHOLD + 30,
            data[(data[:, 0] > THRESHOLD) & (data[:, 0] < 500), 0],
        ]
    )
    nodes, weights = roots_legendre(order)
    mid = (grid[1:] + grid[:-1]) / 2
    half = np.diff(grid) / 2
    e = mid[:, None] + half[:, None] * nodes
    stop = np.exp(spline(np.log(e)))
    increments = half * np.sum(weights / stop, axis=1)
    scatter_increments = half * np.sum(weights * scattering_power(e) / stop, axis=1)
    distance = np.r_[0, np.cumsum(increments)]
    # Integrate scattering from high energy downwards. A forward primitive is
    # dominated by the near-threshold variance; subtracting its almost equal
    # values at high energies loses the small physical angular increment.
    scatter = np.r_[np.cumsum(scatter_increments[::-1])[::-1], 0.0]
    stop_at_grid = np.exp(spline(np.log(grid)))
    energy_at = CubicHermiteSpline(distance, grid, stop_at_grid, extrapolate=False)
    scatter_at = CubicHermiteSpline(distance, scatter, -scattering_power(grid), extrapolate=False)
    range_at = PchipInterpolator(grid, distance, extrapolate=False)
    return dict(
        energy=grid,
        distance=distance,
        scatter=scatter,
        energy_at=energy_at,
        scatter_at=scatter_at,
        range_at=range_at,
        margin_mm=float(range_at(REFERENCE_THRESHOLD + 30)),
        maximum_range_mm=float(distance[-1]),
    )
