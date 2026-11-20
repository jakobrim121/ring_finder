"""Evaluate calibrated charge evidence for each active PMT."""

import numpy as np


def log_probabilities(obs, means, ix):
    from readout import positive

    logp, residual = obs.no_hit(means)
    hit = np.maximum(means[ix], 1e-300)
    density, mean, _, tail = positive(hit, obs.offsets, obs.coefficients, obs.zero_source)
    if tail.max() > 1e-11:
        raise ArithmeticError("Photon-count summation did not converge")
    logp[ix], residual[ix] = density, 1.0 - mean / hit
    return logp, residual
