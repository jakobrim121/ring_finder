"""Exact line-segment exclusion of the detector's truncated spherical domes.

For each dome, minimize max(norm(r)-R, cut-normal.r) along the segment.
Nonnegative minima mean that the entire emitting segment is outside the dome.
The minimum of this convex one-dimensional function is at an endpoint, the
radial minimum, or an intersection of its two analytic branches. Gradients use
the envelope theorem; a branch crossing uses the stationary convex combination.
All distances and gradients use metres. No detector measurements enter geometry.
"""

import numpy as np
from wctering.likelihood.constraints import last_input
from wctering.likelihood.occupancy import OccupancyModel
from wctering.likelihood.charge import direction

from wctering.likelihood.water_boundary import CENTRES, NORMALS, RADIUS, CUT, clearance


class WaterDomain:
    """Add the same complete water domain to either optical likelihood."""

    def __init__(self, *args, **kwargs):
        self._water_cache = None
        super().__init__(*args, **kwargs)
        self.water_upper = np.concatenate((self.water_upper, np.zeros(self.k * len(CENTRES))))

    def dome_values_jacobian(self, x):
        if self._water_cache is not None and np.array_equal(x, self._water_cache[0]):
            return self._water_cache[1:]
        values = []
        jac = np.zeros((self.k * len(CENTRES), len(x)))
        for j in range(self.k):
            offset = self.size * j
            p = x[offset : offset + self.size]
            a, da = direction(p[3:6])
            length = p[6] if self.kind == "finite" else 0.0
            g, c, u, l = clearance(p[:3], a, length)
            values.extend(-g)
            rows = slice(j * len(CENTRES), (j + 1) * len(CENTRES))
            jac[rows, offset : offset + 3] = -c
            jac[rows, offset + 3 : offset + 6] = -u @ da
            if self.kind == "finite":
                jac[rows, offset + 6] = -l
        values = np.asarray(values)
        self._water_cache = (np.asarray(x).copy(), values, jac)
        return values, jac

    @last_input
    def water_values(self, x):
        return np.concatenate((super().water_values(x), self.dome_values_jacobian(x)[0]))

    @last_input
    def water_jacobian(self, x):
        return np.vstack((super().water_jacobian(x), self.dome_values_jacobian(x)[1]))


class WaterOccupancyModel(WaterDomain, OccupancyModel):
    """Hit/unhit likelihood with prism and dome exclusions."""
