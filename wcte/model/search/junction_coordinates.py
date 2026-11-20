"""Fit a connected chain around the best measured interaction point."""

import numpy as np

from chain_sources import paths
from track_coordinates import Chart, fit


class Junction(Chart):
    """Move an incoming leg without implicitly translating downstream light."""

    def __init__(self, model, start):
        super().__init__(model, start)
        # Use the join before the longest emitting leg. A long leg can
        # constrain its position while a short incoming leg changes angle.
        span = start[9 : model.ng : 9] * start[10 : model.ng : 9]
        self.section = max(1, int(np.argmax(span)))
        self.initial[:3] = paths(start[: model.ng])[self.section][0][:3]
        self.junction_cache = None

    def full(self, y):
        if self.junction_cache is not None and np.array_equal(y, self.junction_cache[0]):
            return self.junction_cache[1]
        x, mapping, margins, derivatives = super().full(y)
        # The relative displacement depends only on the preceding arcs.
        # Removing it from the origin leaves the optimized junction fixed.
        local, chain_map = paths(x[: self.model.ng])[self.section]
        spatial = chain_map[:3] @ mapping[: self.model.ng]
        start = 2.0 * x[:3] - local[:3]
        start_map = 2.0 * mapping[:3] - spatial
        # Keep the original first-vertex bounds in addition to the usual
        # path and range constraints. The junction itself is inside water.
        limits = self.bounds[:3]
        coordinates = np.r_[start, x[3:]]
        jacobian = np.vstack([start_map, mapping[3:]])
        margin = np.r_[margins, start - limits[:, 0], limits[:, 1] - start]
        jac = np.vstack([derivatives, start_map, -start_map])
        self.junction_cache = y.copy(), (coordinates, jacobian, margin, jac)
        return self.junction_cache[1]


def minimize(model, start, iterations, maximum_evaluations):
    chart = Junction(model, start) if len(model.particles) > 1 else None
    return fit(model, start, iterations, maximum_evaluations, full_likelihood=True, chart=chart)
