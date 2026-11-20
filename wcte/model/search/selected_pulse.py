"""Score whether a PMT produced a digit whose first time entered the gate."""

import math

from numba import njit
import numpy as np

from gated_photon_time import series, wavefront


@njit(cache=True)
def likelihood(hit, times, low, high, centers, rates, resolution, offsets, coefficients, miss_one, miss_two):
    """Return the measured log density and derivatives for every active PMT.

    The fitted intensities describe integrated charge. The analysis gate
    keeps or rejects the digit by its first time. It does not remove later
    photoelectrons from that digit. As in the existing timing law, we
    approximate the integration window as longer than the light pulse.
    """
    n, k = rates.shape
    delta = 0.5 * (miss_two - miss_one**2)
    uniform = 1.0 / (high - low)
    scores = np.empty(n)
    d_rate = np.empty_like(rates)
    d_time = np.empty_like(centers)
    a, b = np.empty(k), np.empty(k)
    left_density, right_density = np.empty(k), np.empty(k)
    t_derivative = np.empty(k)
    for i in range(n):
        index = hit[i]
        # Background is uniform inside the measured analysis window.
        a[0], b[0] = (uniform, (high - times[index]) * uniform) if index >= 0 else (1.0, 0.0)
        left_density[0], right_density[0] = 0.0, 0.0
        rate = rates[i, 0]
        first, later, c, d = rate * a[0], rate * b[0], rate, 0.0
        total = rate
        for j in range(1, k):
            center = centers[i, j - 1]
            rate = rates[i, j]
            total += rate
            if index >= 0:
                a[j], b[j], t_derivative[j] = wavefront(times[index] - center, resolution[i])
                first += rate * a[j]
                later += rate * b[j]
            else:
                left_density[j], a[j], _ = wavefront(low - center, resolution[i])
                right_density[j], b[j], _ = wavefront(high - center, resolution[i])
                c += rate * a[j]
                d += rate * b[j]
        if index >= 0:
            # G(z) = sum_n p(q | n, recorded) p(recorded | n) z^n/n!.
            # One PE arrives at t and each of the other n-1 arrives later.
            log_series, slope = series(later, coefficients, offsets[index], offsets[index + 1], 1)
            inv_first = 1.0 / max(first, 1e-300)
            scores[i] = -total + math.log(max(first, 1e-300)) + log_series
            for j in range(k):
                d_rate[i, j] = -1.0 + inv_first * a[j] + slope * b[j]
            for j in range(1, k):
                d_time[i, j - 1] = rates[i, j] * (inv_first * t_derivative[j] + slope * a[j])
        else:
            # Integrating out charge replaces G(z) by the exact detection
            # series exp(z) - exp(r z) - (p2 - r^2) z^2/2. Its values at
            # the endpoints give the probability of a selected digit.
            zero = math.exp(-total)
            exp_c = math.exp(c - total)
            exp_d = math.exp(d - total)
            miss_c = math.exp(miss_one * c - total)
            miss_d = math.exp(miss_one * d - total)
            missing = (
                -math.expm1(min(c - total, 0.0))
                + miss_c
                - exp_d * math.expm1((miss_one - 1.0) * d)
                + delta * (c * c - d * d) * zero
            )
            inv_missing = 1.0 / max(missing, 1e-300)
            slope_c = exp_c - miss_one * miss_c - 2.0 * delta * c * zero
            slope_d = exp_d - miss_one * miss_d - 2.0 * delta * d * zero
            scores[i] = math.log(max(missing, 1e-300))
            # Rewriting the gradient this way preserves small derivatives
            # when most of the light lies before or after the window.
            detected = (
                (-math.expm1((miss_one - 1.0) * c) * exp_c)
                - (-math.expm1((miss_one - 1.0) * d) * exp_d)
                - delta * (c * c - d * d) * zero
            )
            for j in range(k):
                d_rate[i, j] = (detected - slope_c * a[j] + slope_d * b[j]) * inv_missing
            for j in range(1, k):
                d_time[i, j - 1] = (
                    rates[i, j] * (slope_d * right_density[j] - slope_c * left_density[j]) * inv_missing
                )
    return scores, d_rate, d_time


def resolution(charge):
    return np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(charge, 0.25)) / 2.355**2)


class Selection:
    """Reuse detector data across hypotheses with a common observed gate."""

    def __init__(self, event, model):
        center = np.median(event.times_ns)
        self.time = event.times_ns - center
        self.gate = np.asarray(event.timing_selection["prompt_window_ns"]) - center
        self.positions = model.pos
        self.hit = np.full(len(self.positions), -1, dtype=int)
        self.hit[model.ix] = np.arange(len(model.ix))
        # An unrecorded pulse has no measured PE to set its resolution.
        # Use the calibrated single-PE width until it is observed.
        self.width = np.full(len(self.positions), resolution(1.0))
        self.width[model.ix] = resolution(event.charges_pe)
        self.obs = model.obs
        self.model = model

    def score(self, rates, times):
        """Evaluate both recorded digits and their missing counterparts."""
        return likelihood(
            self.hit,
            self.time,
            *self.gate,
            times,
            rates,
            self.width,
            self.obs.offsets,
            self.obs.coefficients,
            self.obs.r,
            self.obs.p2,
        )

    def profile(self, shapes, amplitude, times):
        """Estimate integrated yields and a free relative clock per source."""
        from scipy.optimize import minimize

        from charge_starts import weighted_quantile

        model = self.model
        n = len(shapes)
        weights = shapes[1:, model.ix] * amplitude[1:, None]
        weights *= np.minimum(model.q, 2.0)
        if np.any(weights.sum(1) <= 0.0):
            return None
        initial = [weighted_quantile(self.time - t, w, [0.5])[0] for t, w in zip(times[model.ix].T, weights)]
        start = np.r_[np.log(np.maximum(amplitude, 1e-8)), initial]
        bounds = [(-18.0, np.log(1e6))] * n + [(t - 5.0, t + 5.0) for t in initial]

        def objective(x):
            rates = shapes.T * np.exp(x[:n])
            score, d_rate, d_time = self.score(rates, times + x[n:])
            grad = np.r_[(rates * d_rate).sum(0), d_time.sum(0)]
            return -float(score.sum()) / model.total, -grad / model.total

        fit = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=32, ftol=1e-9, gtol=1e-6),
        )
        if not np.isfinite(fit.fun) or fit.fun > objective(start)[0]:
            return None
        rates = shapes.T * np.exp(fit.x[:n])
        fitted_time = times + fit.x[n:]
        return self.score(rates, fitted_time)[0], rates, fitted_time
