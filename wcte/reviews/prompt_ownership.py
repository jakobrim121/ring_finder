"""Condition shared charge and ring visibility on the prompt photon gate."""

from math import exp, expm1, log

from numba import njit
import numpy as np


def build():
    from gated_photon_time import series, wavefront

    @njit(cache=True)
    def prompt(time, gate, rates, centers, widths, offsets, coefficients, core_masks, photon_gate=None):
        """Expected fraction and presence of each source among prompt photons.

        The digit charge observes a mixture of all photons. Given its first
        arrival, the later photons are exchangeable draws from the surviving
        intensity. Thin those draws to the prompt window before computing
        the ratio; the first photon remains a draw from the first-time density.
        """
        shares = np.empty_like(rates)
        present = np.empty((len(rates), len(core_masks)))
        first = np.empty(rates.shape[1])
        in_gate = np.empty(rates.shape[1])
        low, high = gate[0], gate[1]
        if photon_gate is not None:
            low, high = photon_gate[0], photon_gate[1]
        for i, t in enumerate(time):
            first_in_gate = low <= t < high
            first[0] = rates[i, 0] / (gate[1] - gate[0])
            later = first[0] * max(0.0, gate[1] - t)
            in_gate[0] = first[0] * max(0.0, min(gate[1], high) - max(t, low))
            for j in range(1, rates.shape[1]):
                center = centers[i, j - 1]
                density, survival, _ = wavefront(t - center, widths[i])
                first[j] = rates[i, j] * density
                later += rates[i, j] * survival
                if t >= high:
                    in_gate[j] = 0.0
                else:
                    _, outside, _ = wavefront(high - center, widths[i])
                    at_gate = survival
                    if t < low:
                        _, at_gate, _ = wavefront(low - center, widths[i])
                    in_gate[j] = rates[i, j] * max(0.0, at_gate - outside)
            earliest = first.sum()
            prompt_later = in_gate.sum()
            if earliest <= 1e-280 or later <= 1e-280:
                if not first_in_gate and prompt_later <= 1e-280:
                    shares[i] = np.zeros(len(first))
                    shares[i, 0] = 1.0
                    present[i] = np.zeros(len(core_masks))
                    continue
                shares[i] = rates[i] / rates[i].sum()
                for k, mask in enumerate(core_masks):
                    p = 0.0
                    for j in range(rates.shape[1]):
                        if mask & (1 << j):
                            p += shares[i, j]
                    present[i, k] = p
                continue
            start, stop = offsets[i], offsets[i + 1]
            derivative = series(later, coefficients, start, stop, 1)[0]
            outside = max(later - prompt_later, 0.0)
            if first_in_gate:
                first_weight = 1.0
                if prompt_later > 1e-8 * later:
                    value = series(later, coefficients, start, stop, 0)[0]
                    remainder = series(outside, coefficients, start, stop, 0)[0]
                    log_difference = log(max(-expm1(min(remainder - value, 0.0)), 1e-300))
                    first_weight = exp(min(0.0, value + log_difference - derivative - log(prompt_later)))
                shares[i] = first_weight * (first / earliest)
                if prompt_later > 1e-280:
                    shares[i] += (1.0 - first_weight) * (in_gate / prompt_later)
            else:
                shares[i] = np.zeros(len(first))
                no_prompt = exp(min(0.0, series(outside, coefficients, start, stop, 1)[0] - derivative))
                if prompt_later > 1e-280:
                    shares[i] = (1.0 - no_prompt) * (in_gate / prompt_later)
                shares[i, 0] += no_prompt
            for k, mask in enumerate(core_masks):
                core_first, core_later = 0.0, 0.0
                for j in range(rates.shape[1]):
                    if mask & (1 << j):
                        core_first += first[j]
                        core_later += in_gate[j]
                rest = max(later - core_later, 0.0)
                no_first = max(1.0 - core_first / earliest, 1e-300) if first_in_gate else 1.0
                minus = series(rest, coefficients, start, stop, 1)[0]
                present[i, k] = -expm1(min(0.0, log(no_first) + minus - derivative))
        return shares, present

    return prompt


_kernel = None


def ownership(*args):
    global _kernel
    if _kernel is None:
        _kernel = build()
    return _kernel(*args)
