"""Use a censored positive-charge observation for very bright PMT pulses."""

from functools import lru_cache

import numpy as np
from scipy.special import gammainc, gammaincc, gammaln

from readout import parameters, missed_probability


@lru_cache(maxsize=8)
def coefficients(cut, length):
    config = parameters()
    shape = config["gamma_shape_per_photon"]
    scale = config["mean_charge_per_photon"] / shape
    n = np.arange(1, length + 1, dtype=float)
    # Expand the probability of charge below the cut or a missed pulse.
    # This series goes to zero quickly for large photon counts.
    missed = missed_probability(n, config)
    lower = missed + (1.0 - missed) * gammainc(n * shape, cut / scale)
    upper = (1.0 - missed) * gammaincc(n * shape, cut / scale)
    # The first-PE charge follows the measured tabulation.
    x = np.asarray(config["single_photon_charge_grid"])
    y = np.asarray(config["single_photon_positive_density"])
    tail = np.r_[cut, x[x > cut]]
    survival = np.trapz(np.interp(tail, x, y, right=0.0), tail) if cut < x[-1] else 0.0
    upper[0] = (1.0 - missed[0]) * survival
    lower[0] = 1.0 - upper[0]
    with np.errstate(divide="ignore"):
        factorial = gammaln(n + 1)
        lower, upper = np.log(lower) - factorial, np.log(upper) - factorial
    return lower, upper
