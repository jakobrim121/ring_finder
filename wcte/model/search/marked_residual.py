"""Rank proposed wavefronts using charge and first-pulse timing together."""

import math

from numba import njit
import numpy as np
from scipy.special import ndtr

from joint_prompt_fit import wavefront
from residual_joint_optics import path_from_geometry, prompt


def pulse(times, arrival, sigma, low, high):
    """Density at the first time and survival to the end of the gate."""
    delta = times - arrival
    norm = np.sqrt(2.0 * np.pi)
    wide = np.hypot(sigma, 3.0)
    broad_delay = 1.5
    sharp = (delta / sigma) ** 2
    broad = ((delta - broad_delay) / wide) ** 2
    density = (0.84 * np.exp(-0.5 * sharp) / sigma + 0.16 * np.exp(-0.5 * broad) / wide) / norm

    def acceptance(begin):
        sharp = ndtr((high - arrival) / sigma) - ndtr((begin - arrival) / sigma)
        broad = ndtr((high - arrival - broad_delay) / wide) - ndtr((begin - arrival - broad_delay) / wide)
        return 0.84 * sharp + 0.16 * broad

    scale = np.maximum(acceptance(low), 1e-12)
    return density / scale, np.maximum(acceptance(times), 0.0) / scale


@njit(cache=True)
def gains(shapes, first, later, rest, loss, maximum):
    """Profile one positive yield with a concave surrogate for the fit gain."""
    n, hits = shapes.shape
    score = np.zeros(n)
    amount = np.zeros(n)
    for j in range(n):
        a = shapes[j] * first
        b = shapes[j] * later
        penalty = loss[j]
        if np.sum(a + rest * b) <= penalty:
            continue
        low, high = 0.0, maximum / penalty
        # Increasing the yield decreases the slope monotonically. The
        # sum of the inferred hit multiplicities bounds the optimum.
        for _ in range(12):
            test = 0.5 * (low + high)
            derivative = -penalty
            for i in range(hits):
                derivative += a[i] / (1.0 + test * a[i])
                derivative += rest[i] * b[i] / (1.0 + test * b[i])
            if derivative > 0.0:
                low = test
            else:
                high = test
        trial = 0.5 * (low + high)
        value = -trial * penalty
        for i in range(hits):
            value += math.log1p(trial * a[i]) + rest[i] * math.log1p(trial * b[i])
        score[j] = max(value, 0.0)
        amount[j] = trial
    return score, amount


class MarkedResidual:
    """Keep the incumbent light in the denominator of each trial wavefront."""

    def __init__(self, event, model, geometry, shapes, amplitudes):
        self.ix = model.ix
        self.time = event.times_ns - np.median(event.times_ns)
        self.gate = np.asarray(event.timing_selection["prompt_window_ns"]) - np.median(event.times_ns)
        self.sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
        hit_rates = (amplitudes[:, None] * shapes[:, self.ix]).T
        hit_mean = hit_rates.sum(1)
        fractions = hit_rates / hit_mean[:, None]
        paths = [path_from_geometry(geometry, j) for j in range(len(amplitudes) - 1)]
        arrivals = np.column_stack([wavefront(event.pmt_positions_mm, p)[0] for p in paths])
        _, clocks = prompt(event, arrivals, fractions)
        direct, after = pulse(self.time[:, None], arrivals + clocks, self.sigma[:, None], *self.gate)
        width = np.diff(self.gate)[0]
        sky = hit_rates[:, 0] / width
        self.first = sky + (hit_rates[:, 1:] * direct).sum(1)
        self.later = sky * (self.gate[1] - self.time) + (hit_rates[:, 1:] * after).sum(1)
        _, slope = model.obs.score(amplitudes @ shapes, model.ix)
        self.rest = np.maximum(hit_mean * (1.0 - slope[model.ix]) - 1.0, 0.0)
        self.maximum = float(len(hit_mean) + self.rest.sum())
        self.missed = model.obs.r

    def __call__(self, shape, arrival):
        """Use one shared trial clock per batch, then refit with exact optics."""
        density, survive = pulse(self.time, arrival, self.sigma, *self.gate)
        hit_shape = np.ascontiguousarray(shape[:, self.ix])
        # A small expected photon cost is recovered for unhit PMTs through
        # the measured missed-single-PE probability. The whole normalized
        # shape still pays for putting light where there is no recorded hit.
        penalty = 1.0 - self.missed * np.maximum(0.0, 1.0 - hit_shape.sum(1))
        return gains(
            hit_shape,
            density / np.maximum(self.first, 1e-10),
            survive / np.maximum(self.later, 1e-10),
            self.rest,
            penalty,
            self.maximum,
        )
