"""Evaluate the first-pulse likelihood with measured charge integrated out."""

import math
import numpy as np
from numba import njit


def create(event, model):
    """Share the calibrated digit threshold, gate, and time response."""
    from selected_pulse import Selection

    class Occupancy(Selection):
        def score(self, rates, times):
            return likelihood(
                self.hit, self.time, *self.gate, times, rates, self.width, self.obs.r, self.obs.p2
            )

    return Occupancy(event, model)


@njit(cache=True)
def detection_series(z, missed_one, excess):
    ratio = math.exp(-(1.0 - missed_one) * z)
    zero = math.exp(-z)
    first = max(1.0 - missed_one * ratio - excess * z * zero, 1.0e-300)
    second = 1.0 - missed_one**2 * ratio - excess * zero
    return z + math.log(first), second / first


def build_likelihood(censored=False):
    from gated_photon_time import series, wavefront

    @njit(cache=False)
    def score(
        hit,
        times,
        low,
        high,
        centers,
        rates,
        resolution,
        miss_one,
        miss_two,
        charge=None,
        cutoff=0.0,
        offsets=None,
        coefficients=None,
        tail_coefficients=None,
    ):
        n, k = rates.shape
        delta = 0.5 * (miss_two - miss_one**2)
        dz = 2.0 * delta
        uniform = 1.0 / (high - low)
        scores = np.empty(n)
        d_rate = np.empty_like(rates)
        d_time = np.empty_like(centers)
        a, b = np.empty(k), np.empty(k)
        left, right, tprime = np.empty(k), np.empty(k), np.empty(k)
        for i in range(n):
            index = hit[i]
            a[0], b[0] = (uniform, (high - times[index]) * uniform) if index >= 0 else (1.0, 0.0)
            rate = rates[i, 0]
            first, later, c, d = rate * a[0], rate * b[0], rate, 0.0
            total = rate
            for j in range(1, k):
                center = centers[i, j - 1]
                amplitude = rates[i, j]
                total += amplitude
                if index >= 0:
                    a[j], b[j], tprime[j] = wavefront(times[index] - center, resolution[i])
                    first += amplitude * a[j]
                    later += amplitude * b[j]
                else:
                    left[j], a[j], _ = wavefront(low - center, resolution[i])
                    right[j], b[j], _ = wavefront(high - center, resolution[i])
                    c += amplitude * a[j]
                    d += amplitude * b[j]
            if index >= 0:
                if censored and charge[index] < cutoff:
                    value, slope = series(later, coefficients, offsets[index], offsets[index + 1], 1)
                elif censored and later < 100.0:
                    value, slope = series(later, tail_coefficients, 0, len(tail_coefficients), 1)
                else:
                    value, slope = detection_series(later, miss_one, dz)
                inv_first = 1.0 / max(first, 1e-300)
                scores[i] = -total + math.log(max(first, 1e-300)) + value
                for j in range(k):
                    d_rate[i, j] = -1.0 + inv_first * a[j] + slope * b[j]
                for j in range(1, k):
                    d_time[i, j - 1] = rates[i, j] * (inv_first * tprime[j] + slope * a[j])
            else:
                zero = math.exp(-total)
                exp_c = math.exp(c - total)
                exp_d = math.exp(d - total)
                miss_c = math.exp(miss_one * c - total)
                miss_d = math.exp(miss_one * d - total)
                lost = (
                    -math.expm1(min(c - total, 0.0))
                    + miss_c
                    - exp_d * math.expm1((miss_one - 1.0) * d)
                    + delta * (c * c - d * d) * zero
                )
                inverse = 1.0 / max(lost, 1e-300)
                slope_c = exp_c - miss_one * miss_c - dz * c * zero
                slope_d = exp_d - miss_one * miss_d - dz * d * zero
                scores[i] = math.log(max(lost, 1e-300))
                kept = (
                    (-math.expm1((miss_one - 1.0) * c) * exp_c)
                    - (-math.expm1((miss_one - 1.0) * d) * exp_d)
                    - delta * (c * c - d * d) * zero
                )
                for j in range(k):
                    d_rate[i, j] = (kept - slope_c * a[j] + slope_d * b[j]) * inverse
                for j in range(1, k):
                    d_time[i, j - 1] = rates[i, j] * (slope_d * right[j] - slope_c * left[j]) * inverse
        return scores, d_rate, d_time

    return score


def likelihood(*args):
    global _compiled
    if _compiled is None:
        _compiled = build_likelihood()
    return _compiled(*args)


_compiled = None
