"""Cache the selected-digit score derivative for a new source of zero rate."""

import numpy as np
from numba import njit

from gated_photon_time import series, wavefront


@njit(cache=True)
def factors(hit, times, gate, rates, centers, widths, offsets, coefficients, miss_one, miss_two):
    answer = np.zeros((len(rates), 5))
    low, high = gate
    correction = 0.5 * (miss_two - miss_one**2)
    for i in range(len(rates)):
        j = hit[i]
        diffuse = rates[i, 0]
        total = rates[i].sum()
        sigma = widths[i]
        if j >= 0:
            t = times[j]
            early = diffuse / (high - low)
            late = early * (high - t)
            for rate, clock in zip(rates[i, 1:], centers[i]):
                density, survival, _ = wavefront(t - clock, sigma)
                early += rate * density
                late += rate * survival
            _, derivative = series(late, coefficients, offsets[j], offsets[j + 1], 1)
            answer[i, 0] = -1.0
            answer[i, 1] = 1.0 / max(early, 1.0e-300)
            answer[i, 2] = derivative
        else:
            left, right = diffuse, 0.0
            for rate, clock in zip(rates[i, 1:], centers[i]):
                left += rate * wavefront(low - clock, sigma)[1]
                right += rate * wavefront(high - clock, sigma)[1]
            zero = np.exp(-total)
            exp_left, exp_right = np.exp(left - total), np.exp(right - total)
            miss_left, miss_right = np.exp(miss_one * left - total), np.exp(miss_one * right - total)
            missing = -np.expm1(min(left - total, 0.0)) + miss_left
            missing -= exp_right * np.expm1((miss_one - 1.0) * right)
            adjust = correction * (left * left - right * right) * zero
            missing += adjust
            scale = 1.0 / max(missing, 1.0e-300)
            detected = -np.expm1((miss_one - 1.0) * left) * exp_left
            detected += np.expm1((miss_one - 1.0) * right) * exp_right - adjust
            answer[i, 0] = detected * scale
            answer[i, 3] = -(exp_left - miss_one * miss_left - 2.0 * correction * left * zero) * scale
            answer[i, 4] = (exp_right - miss_one * miss_right - 2.0 * correction * right * zero) * scale
    return answer


@njit(cache=True, inline="always")
def score(center, observed, width, gate, factor, seen):
    slope = factor[0]
    if seen:
        density, survival, _ = wavefront(observed - center, width)
        slope += density * factor[1] + survival * factor[2]
    else:
        low = wavefront(gate[0] - center, width)[1]
        high = wavefront(gate[1] - center, width)[1]
        slope += low * factor[3] + high * factor[4]
    return slope


def cache(selection, rates, centers):
    obs = selection.obs
    return factors(
        selection.hit,
        selection.time,
        selection.gate,
        rates,
        centers,
        selection.width,
        obs.offsets,
        obs.coefficients,
        obs.r,
        obs.p2,
    )
