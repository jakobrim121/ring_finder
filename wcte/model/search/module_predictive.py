"""Allow extra diffuse light when comparing already fitted track geometries."""

import math

from numba import njit
import numpy as np

from compare_count_models import Comparison
from gated_photon_time import series
from predictive_score import log_probabilities


@njit(cache=True)
def log_charge(rate, hit, offsets, coefficients, missed_one, missed_two):
    if hit >= 0:
        return series(rate, coefficients, offsets[hit], offsets[hit + 1], 0, False)[0] - rate
    correction = 0.5 * (missed_two - missed_one**2) * rate**2 * math.exp(-missed_one * rate)
    return -(1.0 - missed_one) * rate + math.log1p(correction)


def background_rule(scale, order):
    """Integrate a zero-or-exponential prior, resolving its small-yield end."""
    # Integrate in PE units with a quadrature tuned to typical per-module
    # charge. A long upper interval contributes only on very bright modules.
    node, weight = np.polynomial.legendre.leggauss(order)
    limits = np.array([0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]) * min(1.0, scale)
    widths = 0.5 * np.diff(limits)
    levels = (0.5 * (limits[1:] + limits[:-1]))[:, None] + widths[:, None] * node
    log_prior = np.log(0.5 * widths[:, None] * weight / scale) - levels / scale
    return np.r_[0.0, levels.ravel()], np.r_[np.log(0.5), log_prior.ravel()]


@njit(cache=True)
def held_out(
    intensity, hit, modules, relative_bg, levels, prior, offsets, coefficients, missed_one, missed_two
):
    """Use other tubes in the same module to infer the diffuse PE yield.

    The track geometry and ordinary yields stay fixed. For each tube the
    predictive density is the ratio of two integrals over the same local
    background. The denominator omits that tube's measured charge.
    """
    sizes = np.bincount(modules)
    full = np.zeros((len(sizes), len(levels)))
    tubes = np.empty((len(intensity), len(levels)))
    for i in range(len(intensity)):
        for j in range(len(levels)):
            rate = intensity[i] + relative_bg[i] * levels[j]
            value = log_charge(rate, hit[i], offsets, coefficients, missed_one, missed_two)
            tubes[i, j] = value
            full[modules[i], j] += value
    normalizer = np.empty(len(sizes))
    for m in range(len(sizes)):
        joint = full[m] + prior
        best = joint.max()
        normalizer[m] = best + math.log(np.exp(joint - best).sum())
    by_tube = np.empty(len(intensity))
    extra = np.empty(len(intensity))
    for i in range(len(intensity)):
        m = modules[i]
        prior_only = full[m] - tubes[i] + prior
        best = prior_only.max()
        weights = np.exp(prior_only - best)
        mass = weights.sum()
        by_tube[i] = normalizer[m] - best - math.log(mass)
        extra[i] = (weights @ levels) * relative_bg[i] / mass
    return by_tube, extra


class ModulePredictive(Comparison):
    """Compare each stored track bank in the presence of patchy soft light."""

    def __init__(self, event, model, background_pe):
        super().__init__(event, model)
        self.hit = np.full(len(model.active), -1, dtype=np.int64)
        self.hit[model.ix] = np.arange(len(model.ix))
        # Use the existing diffuse fit to set one event-wide PE scale.
        # Each module can have zero extra light or an exponential yield.
        # Within a module the light follows the diffuse optical acceptance.
        normal = np.bincount(self.modules, weights=model.bg)
        self.relative_bg = model.bg / normal[self.modules]
        scale = max(1.0, background_pe / len(normal))
        self.levels, self.prior = background_rule(scale, 10)

    def measured(self, fractions, record, intensity, rates, time, dimension, masks):
        base = super().measured(fractions, record, intensity, rates, time, dimension, masks)
        model = self.model
        obs = model.obs
        by_tube, extra = held_out(
            intensity,
            self.hit,
            self.modules,
            self.relative_bg,
            self.levels,
            self.prior,
            obs.offsets,
            obs.coefficients,
            obs.r,
            obs.p2,
        )
        ordinary, _ = log_probabilities(obs, intensity, model.ix)
        # Keep the full first-pulse evidence. Increase the diffuse part of
        # the timing model only when testing whether a core is supported.
        # This prevents a source from counting PEs assigned to the inferred
        # background while leaving the reconstructed partition untouched.
        base.joint += np.bincount(self.modules, weights=by_tube - ordinary)
        base.rates = rates.copy()
        base.rates[:, 0] += extra[model.ix]
        return base


def optical_has_weak_core(model, geometry, shapes, amplitude):
    """Recheck a weak optical source or an opening beyond Cherenkov saturation."""
    from wctering.seeding.coherent_timing import N_PHASE

    if np.any(np.deg2rad(geometry["openings_deg"]) > np.arccos(1.0 / N_PHASE)):
        return True
    hit_rates = np.asarray(amplitude)[:, None] * shapes[:, model.ix]
    total = hit_rates.sum(0)
    support = [np.sum(1.0 - model.obs.no_source(total, core / total)) for core in hit_rates[1:]]
    # A core's few photons can be concentrated in one module or be
    # interleaved with a brighter source. Allow enough charge to cover two
    # visibility thresholds before dismissing the smaller model.
    return min(support) < 2.0 * 30.0
