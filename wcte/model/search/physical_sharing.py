"""Condition the core and tail charge fractions on the measured first PE."""

from math import exp, log

from numba import njit
import numpy as np

from gated_photon_time import series, wavefront


@njit(cache=True)
def ownership(times, gate, rates, means, sigma, offsets, coefficients):
    """Average the PE fraction over the calibrated posterior photon number."""
    output = np.empty_like(rates)
    first, later = np.empty(rates.shape[1]), np.empty(rates.shape[1])
    for i, t in enumerate(times):
        first[0] = rates[i, 0] / (gate[1] - gate[0])
        later[0] = first[0] * (gate[1] - t)
        for j in range(1, rates.shape[1]):
            density, tail, _ = wavefront(t - means[i, j - 1], sigma[i])
            first[j] = rates[i, j] * density
            later[j] = rates[i, j] * tail
        a, b = first.sum(), later.sum()
        # p(N=n | q,t_first) ∝ n c_n B^(n-1). Therefore E[1/N]
        # is G(B)/(B G'(B)). The first PE and each subsequent PE
        # have different component probabilities.
        if a <= 1e-300 or b <= 1e-300:
            output[i] = rates[i] / rates[i].sum()
            continue
        first_index, last = offsets[i], offsets[i + 1]
        denominator = series(b, coefficients, first_index, last, 1)[0]
        numerator = series(b, coefficients, first_index, last, 0)[0]
        inverse = exp(min(0.0, numerator - denominator - log(b)))
        output[i] = inverse * first / a + (1.0 - inverse) * later / b
    return output
