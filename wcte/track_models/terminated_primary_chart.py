"""A finite primary ending at an unresolved inelastic interaction.

The existing source, material energy loss and smooth-scattering prior are
unchanged. One additional coordinate places an endpoint before the ordinary
stopping/water boundary. Unobserved outgoing products are not reconstructed;
this is an adequacy model for the incoming light, not a complete reaction law.
"""

import numpy as np

from staged_physics.scattered_primary import ScatteredPrimaryChart


class TerminatedPrimaryChart:
    def __init__(self, model, start):
        self.base = ScatteredPrimaryChart(model, start)
        available = self.base.full(self.base.initial)[0][10]
        retained = np.clip(start[10] / available, 1e-6, 1.0)
        self.initial = np.r_[self.base.initial, retained]
        # A positive numerical endpoint avoids the zero-length singularity of
        # the existing angular diffusion density; it is not a ring threshold.
        self.bounds = np.vstack([self.base.bounds, [1e-6, 1.0]])

    def full(self, coordinates):
        fraction = coordinates[-1]
        original, derivative, margins, constraint = self.base.full(coordinates[:-1])
        path = original.copy()
        jacobian = np.column_stack([derivative, np.zeros(len(original))])
        # Bend and length scale together, preserving the original local
        # curvature and energy-loss law on the retained path prefix.
        scaled = [6, 7, 8, 10]
        path[scaled] *= fraction
        jacobian[scaled, :-1] *= fraction
        jacobian[scaled, -1] = original[scaled]
        return path, jacobian, margins, np.column_stack([constraint, np.zeros(len(margins))])
