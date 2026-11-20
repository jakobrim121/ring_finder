"""Propagate PMT readout uncertainty to the probability of 30 visible sensors."""

import numpy as np
from numba import njit


@njit(cache=True)
def tail(sensors, threshold):
    """Keep the exact Poisson-binomial distribution below the count cut."""
    p = np.zeros(threshold)
    p[0] = 1.0
    at_least = 0.0
    for support in sensors:
        at_least += support * p[-1]
        for j in range(threshold - 1, 0, -1):
            p[j] = (1.0 - support) * p[j] + support * p[j - 1]
        p[0] *= 1.0 - support
    return min(max(at_least, 0.0), 1.0)


def visibility(ids, likelihood, threshold=30):
    tubes, inverse = np.unique(ids, return_inverse=True)
    absent = np.ones(len(tubes))
    np.multiply.at(absent, inverse, 1.0 - np.clip(likelihood, 0.0, 1.0))
    pmt_support = 1.0 - absent
    return float(tail(pmt_support, threshold)), float(pmt_support.sum())
