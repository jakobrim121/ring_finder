"""A coherent circular primary with the existing material scattering law.

This is the first MCS approximation: two fitted mean-bend coordinates, a
normalized moment-matched endpoint-direction density, and the existing
unresolved slowing-tail diffusion. It is not a complete stochastic path
integral or a model of rare sharp Coulomb deflections. Those need a separate
trajectory hypothesis, rather than an independent daughter particle.
"""

import numpy as np

from source import trajectory

from .circular_domain import first_exit
from .primary import Aperture, PrimaryChart


class ScatteredPrimaryChart(PrimaryChart):
    """Fit a curved mean path whose endpoint follows the same material domain."""

    def __init__(self, model, start, aperture=Aperture()):
        super().__init__(model, start, aperture)
        complete_bend = np.asarray(start[6:9]) / start[10]
        coordinates = self.frame.T @ complete_bend
        self.initial = np.r_[self.initial[:4], coordinates, self.initial[4:]]
        self.bounds = np.vstack([self.bounds[:4], [(-1.0, 1.0)] * 2, self.bounds[4:]])
        radius = np.linalg.norm(self.initial[:2])
        if aperture.radius < radius <= aperture.radius + 1e-6:
            self.initial[:2] *= (aperture.radius - 1e-8) / radius

    def full(self, y):
        if self.cached is not None and np.array_equal(y, self.cached[0]):
            return self.cached[1:]
        x = np.zeros(14)
        jac = np.zeros((14, len(y)))
        x[:3] = y[0], y[1], self.aperture.z
        jac[:2, :2] = np.eye(2)
        vector = self.axis + self.frame @ y[2:4]
        norm = np.linalg.norm(vector)
        axis = vector / norm
        da = (self.frame - np.outer(axis, axis @ self.frame)) / norm
        x[3:6] = axis
        jac[3:6, 2:4] = da
        raw_bend = self.frame @ y[4:6]
        along = axis @ raw_bend
        bend = raw_bend - along * axis
        db = np.zeros((3, len(y)))
        db[:, 2:4] = -along * da - np.outer(axis, raw_bend @ da)
        db[:, 4:6] = self.frame - np.outer(axis, axis @ self.frame)
        ran = np.exp(y[6])
        x[9] = ran
        jac[9, 6] = ran
        x[11:14] = y[7:10]
        jac[11:14, 7:10] = np.eye(3)
        boundary = first_exit(x[:3], axis, bend, ran)
        fraction = 1.0 if boundary is None else boundary["length"] / ran
        x[6:9] = fraction * bend
        jac[6:9] = fraction * db
        x[10] = fraction
        if boundary is not None:
            _, _, position_jac, _ = trajectory(x[:12], np.ones(1))
            endpoint = position_jac[0]
            df_position = endpoint[:, 6:9] @ bend + endpoint[:, 10]
            normal = boundary["normal"]
            denominator = float(normal @ df_position)
            if abs(denominator) < 1e-10:
                raise ArithmeticError("Grazing circular water boundary")
            df = -(normal @ endpoint @ jac[:12]) / denominator
            # Match the straight model's one-micrometre interior offset.
            fraction -= 1e-6 / ran
            df[6] += 1e-6 / ran
            x[6:9] = fraction * bend
            jac[6:9] = fraction * db + np.outer(bend, df)
            x[10] = fraction
            jac[10] = df
        radius = self.aperture.radius
        aperture = (radius**2 - y[:2] @ y[:2]) / radius
        da_aperture = np.zeros(len(y))
        da_aperture[:2] = -2 * y[:2] / radius
        # A smooth one-core arc cannot hide a >15-degree turn within 1 cm.
        # Total curvature over a long core is not itself a ring split.
        local_turn2 = 0.01**2 * (bend @ bend) / ran**2
        local_margin = np.deg2rad(15.0) ** 2 - local_turn2
        local_gradient = -2 * 0.01**2 * (bend @ db) / ran**2
        local_gradient[6] += 2 * local_turn2
        margins = np.array([aperture, local_margin])
        constraint_jac = np.vstack([da_aperture, local_gradient])
        self.cached = y.copy(), x, jac, margins, constraint_jac
        return self.cached[1:]
