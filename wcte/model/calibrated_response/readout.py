"""Calibrated compound-Poisson charge likelihood and photon responsibilities.

Geometry predicts a mean number of detected photoelectrons, lambda. The
electronics may miss a small burst, and its reported charge is continuous.
We sum over the unknown integer photon count instead of treating charge as
that count. Calibration parameters are external detector configuration.
"""

from functools import lru_cache
from pathlib import Path
import json
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln


@lru_cache(None)
def parameters():
    path = Path(__file__).with_name("response_parameters.json")
    return json.loads(path.read_text())


@njit(cache=True)
def positive(rates, offsets, coefficients, source_fraction):
    """Return log density, conditional photon mean and no-source probability.

    Each receiver has its own compact range of photon counts. The coefficient
    is log[p(charge|n)/n!]; only n*log(lambda) changes during geometry fitting.
    Summation in log space also handles badly mismatched trial geometries.
    """
    log_density = np.empty(len(rates))
    photon_mean = np.empty(len(rates))
    absent = np.empty(len(rates))
    boundary = np.empty(len(rates))
    for i in range(len(rates)):
        rate = max(rates[i], 1e-300)
        log_rate = math.log(rate)
        begin, end = offsets[i], offsets[i + 1]
        maximum = -1e300
        for k in range(begin, end):
            value = coefficients[k] + (k - begin + 1) * log_rate
            if value > maximum:
                maximum = value
        total = 0.0
        mean = 0.0
        missing = 0.0
        tail = 0.0
        fraction = source_fraction[i]
        log_absence = math.log1p(-fraction) if fraction < 1.0 else -1e300
        for k in range(begin, end):
            n = k - begin + 1
            weight = math.exp(coefficients[k] + n * log_rate - maximum)
            total += weight
            mean += n * weight
            missing += weight * math.exp(n * log_absence)
            if k >= end - 4:
                tail += weight
        log_density[i] = -rate + maximum + math.log(total)
        photon_mean[i] = mean / total
        absent[i] = missing / total
        boundary[i] = tail / total
    return log_density, photon_mean, absent, boundary


def missed_probability(n, config):
    result = config["miss_one_photon"] ** np.asarray(n, float)
    return np.where(np.asarray(n) == 2, config["miss_two_photons"], result)


class Observation:
    def __init__(self, q):
        self.q = np.asarray(q, float)
        config = parameters()
        assert self.q.ndim == 1 and np.all(self.q > 0) and np.isfinite(self.q).all()
        self.shape = float(config["gamma_shape_per_photon"])
        self.gain = float(config["mean_charge_per_photon"])
        self.r = float(config["miss_one_photon"])
        self.p2 = float(config["miss_two_photons"])
        charge_grid = np.asarray(config["single_photon_charge_grid"])
        one_density = np.asarray(config["single_photon_positive_density"])
        coefficients = []
        offsets = [0]
        for charge in self.q:
            # A conservative saddle estimate at the maximum allowed total
            # intensity sets a data-independent truncation margin. Every use
            # also checks the final four posterior terms for convergence.
            estimate = charge / self.gain
            mode = max(
                estimate, math.exp((math.log(1e7) + self.shape * math.log(estimate)) / (self.shape + 1))
            )
            limit = int(math.ceil(mode + 16 * math.sqrt(mode / (self.shape + 1) + 1) + 60))
            n = np.arange(1, limit + 1, dtype=float)
            k = self.shape * n
            scale = self.gain / self.shape
            coefficient = (k - 1) * math.log(charge) - charge / scale - gammaln(k) - k * math.log(scale)
            coefficient += np.log1p(-missed_probability(n, config)) - gammaln(n + 1)
            density = float(np.interp(charge, charge_grid, one_density, left=0.0, right=0.0))
            coefficient[0] = math.log((1 - self.r) * density) if density > 0 else -np.inf
            coefficients.append(coefficient)
            offsets.append(offsets[-1] + len(n))
        self.coefficients = np.concatenate(coefficients)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.zero_source = np.zeros(len(self.q))

    def no_hit(self, rates):
        """Exactly sum the declared missed-burst probabilities over Poisson n.

        For n>=3, p(miss|n)=r**n. The measured two-photon exception is a small
        explicit correction to exp[-lambda*(1-r)], evaluated without overflow.
        """
        rates = np.asarray(rates, float)
        delta = self.p2 - self.r**2
        exponential = np.exp(-self.r * rates)
        correction = 0.5 * delta * rates**2 * exponential
        assert np.all(correction > -1)
        logp = -(1 - self.r) * rates + np.log1p(correction)
        residual = 1 - self.r - delta * (rates - 0.5 * self.r * rates**2) * exponential / (1 + correction)
        return logp, residual

    def score(self, means, hit_indices):
        """Log likelihood and gradient of its negative with respect to means."""
        logp, residual = self.no_hit(means)
        hit_means = np.maximum(means[hit_indices], 1e-300)
        observed, photons, _, tail = positive(hit_means, self.offsets, self.coefficients, self.zero_source)
        if float(tail.max()) > 1e-11:
            raise ArithmeticError("Photon-count summation did not converge")
        logp[hit_indices] = observed
        residual[hit_indices] = 1 - photons / hit_means
        return float(logp.sum()), residual

    def no_source(self, hit_means, fraction):
        """P(source contributes no photon | measured charge, observed hit)."""
        _, _, absent, tail = positive(
            np.asarray(hit_means), self.offsets, self.coefficients, np.asarray(fraction)
        )
        if float(tail.max()) > 1e-11:
            raise ArithmeticError("Photon-count summation did not converge")
        return absent


@lru_cache(maxsize=4)
def _observation(charge_bytes):
    return Observation(np.frombuffer(charge_bytes, dtype=np.float64))


def observation(q):
    return _observation(np.ascontiguousarray(q, dtype=np.float64).tobytes())


def profile(shapes, hit_indices, q, initial_amplitudes=None):
    """Profile nonnegative source intensities under the same calibrated score."""
    obs = observation(q)
    shapes = np.asarray(shapes, float)
    total = float(np.sum(q) / obs.gain)
    np.testing.assert_allclose(shapes.sum(1), 1.0, atol=1e-10, rtol=0)
    assert len(np.unique(hit_indices)) == len(hit_indices)

    def objective(x):
        amplitudes = np.exp(x)
        terms = amplitudes[:, None] * shapes
        ll, residual = obs.score(terms.sum(0), hit_indices)
        return -ll / max(total, 1.0), terms @ residual / max(total, 1.0)

    weights = np.ones(len(shapes)) / len(shapes)
    starts = [np.log(np.maximum(total * weights, 1e-6))]
    if initial_amplitudes is not None:
        weights = np.maximum(np.asarray(initial_amplitudes, float), 1e-10)
        weights /= weights.sum()
        starts.append(np.log(np.maximum(total * weights, 1e-6)))
    bounds = [(-18.0, np.log(1e6))] * len(shapes)
    fits = []
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=400, ftol=1e-13, gtol=1e-9),
        )
        gradient = objective(result.x)[1]
        gradient[(result.x <= -18 + 1e-8) & (gradient > 0)] = 0.0
        gradient[(result.x >= np.log(1e6) - 1e-8) & (gradient < 0)] = 0.0
        projected = float(abs(gradient).max())
        if projected <= 1e-7:
            fits.append((result.fun, result.x, projected))
    if not fits:
        return None, dict(qualified=False)
    value, x, error = min(fits, key=lambda v: v[0])
    amplitudes = np.exp(x)
    terms = amplitudes[:, None] * shapes[:, hit_indices]
    posterior = terms / terms.sum(0)
    return posterior, dict(
        qualified=True,
        ll=-float(value) * max(total, 1.0),
        amplitudes=amplitudes.tolist(),
        projected_gradient=error,
    )
