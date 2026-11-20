"""Preserve the original floating-point summation while skipping inert terms.

The n>=3 posterior terms are log-concave. Binary search finds exactly the
same maximum as a full scan. Terms are then added in the original increasing
n order. Once the next terms cannot change any accumulator even by one ulp,
and both mass and first-moment terms are decreasing, all later additions
would also be inert. No probability tolerance is used in this shortcut.
The last four terms are still computed for the original table-tail guard.
"""

import math
import numpy as np
from numba import literally, njit


@njit(cache=True)
def positive_exact(rates, offsets, coefficients, source_fraction):
    return sum_positive(rates, offsets, coefficients, source_fraction, False)


@njit(cache=True)
def positive_without_source(rates, offsets, coefficients, source_fraction):
    """Share the probability mass with the absence sum when it is identical."""
    if np.any(source_fraction != 0.0):
        return positive_exact(rates, offsets, coefficients, source_fraction)
    return sum_positive(rates, offsets, coefficients, source_fraction, True)


@njit(cache=True)
def sum_positive(rates, offsets, coefficients, source_fraction, charge_only):
    charge_only = literally(charge_only)
    density = np.empty(len(rates))
    average = np.empty(len(rates))
    absent = np.empty(len(rates))
    boundary = np.empty(len(rates))
    for i in range(len(rates)):
        rate = max(rates[i], 1e-300)
        log_rate = math.log(rate)
        begin, end = offsets[i], offsets[i + 1]
        low, high = begin + 2, end - 1
        while low < high:
            mid = (low + high) // 2
            if coefficients[mid + 1] - coefficients[mid] + log_rate > 0.0:
                low = mid + 1
            else:
                high = mid
        mode = low
        maximum = coefficients[mode] + (mode - begin + 1) * log_rate
        for j in range(begin, begin + 2):
            maximum = max(maximum, coefficients[j] + (j - begin + 1) * log_rate)
        log_absence = 0.0
        if not charge_only:
            log_absence = math.log1p(-source_fraction[i]) if source_fraction[i] < 1.0 else -1e300
        total, mean, missing = 0.0, 0.0, 0.0
        weight = math.exp(coefficients[begin] + log_rate - maximum)
        for j in range(begin, end):
            n = j - begin + 1
            total += weight
            mean += n * weight
            if not charge_only:
                missing += weight * math.exp(n * log_absence)
            if j + 1 == end:
                break
            next_weight = math.exp(coefficients[j + 1] + (n + 1) * log_rate - maximum)
            if j >= mode:
                # Checking the moment ratio is stronger than checking that
                # weights decrease. Later ratios only decrease by concavity.
                decreasing = next_weight * (n + 1) < weight * n or next_weight == 0.0
                if decreasing and total + next_weight == total and mean + (n + 1) * next_weight == mean:
                    if charge_only:
                        break
                    next_missing = next_weight * math.exp((n + 1) * log_absence)
                    if missing + next_missing == missing:
                        break
            weight = next_weight
        tail = 0.0
        for j in range(max(begin, end - 4), end):
            tail += math.exp(coefficients[j] + (j - begin + 1) * log_rate - maximum)
        density[i] = -rate + maximum + math.log(total)
        average[i] = mean / total
        absent[i] = 1.0 if charge_only else missing / total
        boundary[i] = tail / total
    return density, average, absent, boundary
