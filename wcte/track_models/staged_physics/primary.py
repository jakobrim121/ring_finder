"""Complete primary track from the surveyed entrance to its visible endpoint.

The current detector water domain is the surveyed instrument prism with mPMT
dome exclusions. It is an approximation to the complete Geant4 material map,
especially for high-energy tracks entering downstream hardware.
"""

from dataclasses import dataclass

import numpy as np

from wctering.likelihood.crossing import intervals


@dataclass(frozen=True)
class Aperture:
    """Surveyed beam window, in detector coordinates and metres."""

    z: float = (-1537.963 + 188.0 + 1.2) / 1000.0
    radius: float = 0.077


class PrimaryChart:
    """Two entry, two direction, range, width and two response coordinates.

    The path endpoint is derived, not independently shortened to hide light.
    Its range is the distance to the optical Cherenkov threshold in water.
    """

    def __init__(self, model, start, aperture=Aperture()):
        self.aperture = aperture
        self.axis = np.asarray(start[3:6], float)
        self.axis /= np.linalg.norm(self.axis)
        pole = np.eye(3)[np.argmin(abs(self.axis))]
        tangent = np.cross(self.axis, pole)
        tangent /= np.linalg.norm(tangent)
        self.frame = np.column_stack([tangent, np.cross(self.axis, tangent)])
        self.initial = np.r_[start[:2], 0.0, 0.0, np.log(start[9]), start[11:14]]
        self.bounds = np.array(
            [(-aperture.radius, aperture.radius)] * 2
            + [(-2.0, 2.0)] * 2
            + [tuple(np.log(model.range_bounds)), model.width_bounds, *model.response_bounds]
        )
        self.cached = None

    def full(self, y):
        if self.cached is not None and np.array_equal(y, self.cached[0]):
            return self.cached[1:]
        x = np.zeros(14)
        jac = np.zeros((14, len(y)))
        x[:3] = [y[0], y[1], self.aperture.z]
        jac[:2, :2] = np.eye(2)
        vector = self.axis + self.frame @ y[2:4]
        norm = np.linalg.norm(vector)
        axis = vector / norm
        da = (self.frame - np.outer(axis, axis @ self.frame)) / norm
        x[3:6] = axis
        jac[3:6, 2:4] = da
        ran = np.exp(y[4])
        x[9] = ran
        jac[9, 4] = ran
        # The first water interval containing the entrance supplies the
        # endpoint and its implicit plane/sphere boundary derivatives.
        water = intervals(x[:3], axis)
        containing = [(lo, hi) for lo, hi in water if lo["t"] <= 1e-7 < hi["t"]]
        if not containing:
            raise ValueError("Entrance is outside the modeled water domain")
        endpoint = containing[0][1]
        distance = endpoint["t"] - 1e-6  # one micrometre inside the boundary
        if ran <= distance:
            x[10] = 1.0
        else:
            x[10] = distance / ran
            jac[10] = (endpoint["dc"] @ jac[:3] + endpoint["da"] @ jac[3:6]) / ran
            jac[10, 4] -= x[10]
        x[11:14] = y[5:8]
        jac[11:14, 5:8] = np.eye(3)
        radius = self.aperture.radius
        margin = np.array([(radius**2 - y[:2] @ y[:2]) / radius])
        gradient = np.zeros((1, len(y)))
        gradient[0, :2] = -2 * y[:2] / radius
        self.cached = (y.copy(), x, jac, margin, gradient)
        return self.cached[1:]
