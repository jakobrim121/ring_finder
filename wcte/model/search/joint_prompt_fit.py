"""Fit the emitting geometry to the observed charge and prompt wavefront."""

import numpy as np
from scipy.special import softmax

from analog_optical import Analog, contain_finite
from charge_starts import weighted_quantile
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def wavefront(positions, p):
    """Straight-track arrival and derivative in the existing optical chart."""
    axis = p[3:6] / np.linalg.norm(p[3:6])
    tangent = (np.eye(3) - np.outer(axis, axis)) / np.linalg.norm(p[3:6])
    delta = positions - 1000 * p[:3]
    z = delta @ axis
    transverse = delta - z[:, None] * axis
    radius = np.maximum(np.linalg.norm(transverse, axis=1), 1e-7)
    cosine, sine = np.cos(p[7]), np.sin(p[7])
    # The upper/lower endpoints also define the first possible arrival on
    # a line whose ideal emission point lies outside the emitting segment.
    tangent_angle = max(sine, 0.05)
    cotangent = cosine / tangent_angle
    emission = z - radius * cotangent
    end = 500.0 * p[6]
    s = np.clip(emission, -end, end)
    emit = np.empty((len(z), 9))
    dz = np.c_[np.tile(-1000 * axis, (len(z), 1)), delta @ tangent]
    dr = np.c_[-1000 * transverse / radius[:, None], -z[:, None] * dz[:, 3:6] / radius[:, None]]
    emit[:, :6] = dz - cotangent * dr
    emit[:, 6] = 0.0
    emit[:, 7] = radius / (sine * sine) if sine > 0.05 else radius * sine / 0.05
    emit[:, 8] = 0.0
    clipped = emission != s
    emit[clipped] = 0.0
    emit[clipped, 6] = 500 * np.sign(emission[clipped])
    ray = delta - s[:, None] * axis
    distance = np.maximum(np.linalg.norm(ray, axis=1), 1e-7)
    ray /= distance[:, None]
    d_distance = -(ray @ axis)[:, None] * emit
    d_distance[:, :3] -= 1000 * ray
    d_distance[:, 3:6] -= s[:, None] * (ray @ tangent)
    derivative = N_GROUP * d_distance + N_PHASE * cosine * emit
    derivative[:, 7] -= N_PHASE * sine * s
    return (N_PHASE * cosine * s + N_GROUP * distance) / C, derivative / C


class Joint(Analog):
    """Initialize finite paths and a relative clock for the gated joint fit."""

    def __init__(self, event, geometry, bg):
        super().__init__(event, geometry, bg, "finite", 8)
        for j, length in enumerate(geometry.get("lengths_mm", ())):
            if length is not None and length > 0.0:
                p = self.x[9 * j : 9 * j + 9]
                p[:3], p[6] = np.asarray(geometry["vertices"][j]) / 1000, length / 1000
                contain_finite(p)
        self.positions = event.pmt_positions_mm
        self.time = event.times_ns - np.median(event.times_ns)
        self.clock = self.start_clock(self.x)

    def start_clock(self, x):
        shape = np.asarray([self.component(p)[0][self.ix] for p in x[: 9 * self.k].reshape(-1, 9)])
        times = np.column_stack([wavefront(self.positions, p)[0] for p in x[: 9 * self.k].reshape(-1, 9)])
        weight = shape.T * softmax(np.r_[0.0, x[9 * self.k : -1]])[1:] * np.minimum(self.obs.q, 2.0)[:, None]
        return weighted_quantile((self.time[:, None] - times).ravel(), weight.ravel(), [0.5])[0]
