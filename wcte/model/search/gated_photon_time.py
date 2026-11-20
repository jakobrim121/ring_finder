"""Charge-conditioned first-photoelectron timing for a mixture of sources."""

import math

from numba import njit
import numpy as np


@njit(cache=True, inline="always")
def coefficient(values, low, i, derivative, log_rate):
    n = i - low + 1
    term = values[i] + (n - derivative) * log_rate
    return term + math.log(n) if derivative else term


@njit(cache=True, inline="always")
def series(rate, coefficients, low, high, derivative, with_moment=True):
    """Log charge series and its logarithmic rate derivative.

    The coefficients are the same calibrated p(charge | n)/n! used by
    the charge likelihood. Dropping exp(-rate) is exact in the conditional
    time likelihood, where it cancels between numerator and denominator.
    """
    log_rate = math.log(max(rate, 1e-300))
    # Terms from n=3 onward are log concave. The calibrated first two
    # coefficients have exceptions, so include both in the peak search.
    a, b = low + 2, high - 1
    while a < b:
        mid = (a + b) // 2
        if coefficient(coefficients, low, mid + 1, derivative, log_rate) > coefficient(
            coefficients, low, mid, derivative, log_rate
        ):
            a = mid + 1
        else:
            b = mid
    peak = max(
        coefficient(coefficients, low, low, derivative, log_rate),
        coefficient(coefficients, low, low + 1, derivative, log_rate),
        coefficient(coefficients, low, a, derivative, log_rate),
    )
    total, moment = 0.0, 0.0
    for i in range(low, high):
        n = i - low + 1
        term = coefficient(coefficients, low, i, derivative, log_rate)
        w = math.exp(term - peak)
        total += w
        if with_moment:
            moment += (n - derivative) * w
        if i >= a and w <= 1e-17 * total:
            break
    return peak + math.log(total), moment / total / max(rate, 1e-300)


@njit(cache=True, inline="always")
def gaussian(delta, sigma):
    """Density, survival and derivative of the density by source time."""
    z = delta / sigma
    density = math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
    survival = 0.5 * math.erfc(z / math.sqrt(2.0))
    return density, survival, density * z / sigma


@njit(cache=True, inline="always")
def wavefront(delta, sigma):
    narrow = gaussian(delta, sigma)
    broad = gaussian(delta - 1.5, math.sqrt(sigma * sigma + 9.0))
    # The wider, delayed density absorbs optical and electronics tails.
    # This is a moment approximation for a finite emission path.
    return (
        0.84 * narrow[0] + 0.16 * broad[0],
        0.84 * narrow[1] + 0.16 * broad[1],
        0.84 * narrow[2] + 0.16 * broad[2],
    )


@njit(cache=True, inline="always")
def source_parameters(center, sigma, delayed, halo_mask, j):
    """Use the same single-PE parameters in scoring and derivatives."""
    if j == delayed:
        return center, math.sqrt(sigma**2 + 3.5**2), True
    if halo_mask & (1 << j):
        return center + 1.5, math.sqrt(sigma**2 + 9.0), True
    return center, sigma, False


@njit(cache=True, inline="always")
def source_density(delta, sigma, simple):
    return gaussian(delta, sigma) if simple else wavefront(delta, sigma)


@njit(cache=True)
def evaluate(
    times, low, high, means, rates, sigma, offsets, coefficients, delayed=0, halo_mask=0, variance=None
):
    """Log density for first times, and derivatives by rates and source times.

    The time gate is measured from the event. All predictions are in the
    same relative frame. Each recorded pulse integrates charge from an
    unknown number of photons, with only the earliest time observed.
    """
    nr, k = rates.shape
    timing, d_rate, d_mean = np.zeros(nr), np.zeros((nr, k)), np.zeros((nr, k - 1))
    density = np.empty(k)
    remain = np.empty(k)
    derivative = np.empty(k - 1)
    before = np.empty(k)
    after = np.empty(k)
    before_density = np.empty(k - 1)
    after_density = np.empty(k - 1)
    for i in range(nr):
        density[0] = 1.0 / (high - low)
        remain[0] = (high - times[i]) * density[0]
        before[0], after[0] = 1.0, 0.0
        a = rates[i, 0] * density[0]
        b = rates[i, 0] * remain[0]
        c, d = rates[i, 0], 0.0
        for j in range(1, k):
            mu = means[i, j - 1]
            resolution = sigma[i]
            if variance is not None:
                resolution = math.sqrt(resolution**2 + variance[i, j - 1])
            mu, width, simple = source_parameters(mu, resolution, delayed, halo_mask, j)
            density[j], remain[j], derivative[j - 1] = source_density(times[i] - mu, width, simple)
            before_density[j - 1], before[j], _ = source_density(low - mu, width, simple)
            after_density[j - 1], after[j], _ = source_density(high - mu, width, simple)
            w = rates[i, j]
            a += w * density[j]
            b += w * remain[j]
            c += w * before[j]
            d += w * after[j]
        start, stop = offsets[i], offsets[i + 1]
        log_jac, d_log_jac = series(b, coefficients, start, stop, 1)
        log_left, d_left = series(c, coefficients, start, stop, 0)
        log_right, d_right = series(d, coefficients, start, stop, 0)
        ratio = math.exp(min(log_right - log_left, 0.0))
        normalizer = -math.expm1(min(log_right - log_left, -1e-15))
        left, right = d_left / normalizer, d_right * ratio / normalizer
        inv_a = 1.0 / max(a, 1e-300)
        timing[i] = math.log(max(a, 1e-300)) + log_jac - log_left - math.log(normalizer)
        for j in range(k):
            d_rate[i, j] = inv_a * density[j] + d_log_jac * remain[j] - left * before[j] + right * after[j]
        for j in range(1, k):
            d_mean[i, j - 1] = rates[i, j] * (
                inv_a * derivative[j - 1]
                + d_log_jac * density[j]
                - left * before_density[j - 1]
                + right * after_density[j - 1]
            )
    return timing, d_rate, d_mean


@njit(cache=True)
def log_density(
    times, low, high, means, rates, sigma, offsets, coefficients, delayed=0, halo_mask=0, variance=None
):
    """Evaluate the first-PE density when profiling needs no gradient."""
    result = np.empty(len(times))
    width = high - low
    background_density = 1.0 / width
    for i, t in enumerate(times):
        sky = rates[i, 0]
        at_hit = sky * background_density
        after_hit = sky * ((high - t) * background_density)
        after_low, after_high = sky, 0.0
        for j in range(1, rates.shape[1]):
            center = means[i, j - 1]
            resolution = sigma[i]
            if variance is not None:
                resolution = math.sqrt(resolution**2 + variance[i, j - 1])
            center, resolution, simple = source_parameters(center, resolution, delayed, halo_mask, j)
            density, tail, _ = source_density(t - center, resolution, simple)
            left = source_density(low - center, resolution, simple)[1]
            right = source_density(high - center, resolution, simple)[1]
            weight = rates[i, j]
            at_hit += weight * density
            after_hit += weight * tail
            after_low += weight * left
            after_high += weight * right
        start, stop = offsets[i], offsets[i + 1]
        hit = series(after_hit, coefficients, start, stop, 1, False)[0]
        before = series(after_low, coefficients, start, stop, 0, False)[0]
        after = series(after_high, coefficients, start, stop, 0, False)[0]
        normalizer = -math.expm1(min(after - before, -1e-15))
        result[i] = math.log(max(at_hit, 1e-300)) + hit - before - math.log(normalizer)
    return result
