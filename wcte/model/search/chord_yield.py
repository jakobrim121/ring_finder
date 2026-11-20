"""Compare the fitted chord's direct light with the absolute PE calibration."""

import numpy as np

from angular import normalization
from readout import parameters
from source import angular_rule
from wctering.receiver.angular_panels import finite_integrate
from wctering.receiver.finite_source import quadrature


def fraction(positions, normals, chord):
    """Ratio for a uniform, nearly relativistic track in its fitted geometry.

    The angular integral uses unit-normalized cone intensity, the same
    receiver exposure and the same Frank–Tamm scale as the charged models.
    This is an optical completeness check for a *throughgoing* trajectory.
    """
    origin, axis = np.asarray(chord["anchor_m"]), np.asarray(chord["axis"])
    theta, width = np.deg2rad([chord["opening_deg"], chord["width_deg"]])
    angular_norm = normalization(theta, width**2, *angular_rule())[0]
    expected = 0.0
    for lo, hi in chord["emitting_intervals"]:
        length = hi - lo
        x = np.r_[origin + 0.5 * (lo + hi) * axis, axis, length, theta, np.log(width)]
        expected += 1000 * length * finite_integrate(positions, normals, x, *quadrature(8))[0].sum()
    expected *= parameters()["light_coefficient_photons"] * np.sin(theta) ** 2 / angular_norm
    return float(chord["amplitudes"][1] / max(expected, 1e-30)), float(expected)
