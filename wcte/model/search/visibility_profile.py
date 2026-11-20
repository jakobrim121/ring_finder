"""Refit a competing geometry with a weak section below countable support."""

from math import exp, expm1

from numba import njit
import numpy as np

from analytic_knockon import render as knockon
from balanced_search import feasible
from bare_model import Model as Bare
from chain_sources import paths
from gated_photon_time import series
from junction_coordinates import Junction
from source import raw
from track_coordinates import fit as fit_chart


@njit(cache=True)
def support_score(mean, other, offsets, coefficients):
    """Count expected PMTs containing at least one PE from the weak core.

    The no-core conditional probability is the ratio of the two positive
    charge series. The common Poisson factor cancels. Differentiate that
    ratio with respect to the total and the non-core intensity separately.
    """
    support = 0.0
    d_total, d_other = np.empty(len(mean)), np.empty(len(mean))
    for i in range(len(mean)):
        begin, end = offsets[i], offsets[i + 1]
        log_total, total_slope = series(mean[i], coefficients, begin, end, 0)
        log_other, other_slope = series(other[i], coefficients, begin, end, 0)
        delta = min(0.0, log_other - log_total)
        probability = exp(delta)
        support -= expm1(delta)
        d_total[i] = probability * total_slope
        d_other[i] = -probability * other_slope
    return support, d_total, d_other


class BelowSupport(Bare):
    """Use the original absolute response and add one support constraint."""

    def __init__(self, ids, charges, background, particles, section, order):
        super().__init__(ids, charges, background, particles, 4, order)
        self.section = section
        self.countable_cache = None

    def primary_light(self, x):
        pieces, derivative = [], np.zeros((len(self.active), len(x)))
        for j, ((p, transform), mass) in enumerate(zip(paths(x), self.particles)):
            parts, gradient = raw(self.pos, self.normals, p, self.order, False, mass)
            length = 1000.0 * p[9] * p[10]
            d_length = 1000.0 * (p[10] * transform[9] + p[9] * transform[10])
            delta, d_delta = knockon(self.pos, self.normals, p, mass, self.order)
            derivative += (length * gradient.sum(0) + d_delta) @ transform
            derivative += parts.sum(0)[:, None] * d_length
            scaled = length * parts
            scaled[1] += delta
            pieces.append(scaled)
            if j == self.section:
                self.core_gradient = length * (gradient[0] @ transform) + parts[0, :, None] * d_length
        return np.asarray(pieces), derivative

    def support(self, x):
        previous = self.countable_cache
        if previous is not None and np.array_equal(x, previous[0]):
            return previous[1:]
        self(x, True)
        _, total, cores, d_total = self.primary_cache
        efficiency, background = np.exp(x[-2:])
        i = self.ix
        light = efficiency * total[i]
        core = efficiency * cores[self.section, i]
        diffuse = background * self.bg[i]
        other = efficiency * np.maximum(total[i] - cores[self.section, i], 0.0) + diffuse
        value, a, b = support_score(light + diffuse, other, self.obs.offsets, self.obs.coefficients)
        full = np.column_stack([efficiency * d_total[i], light, diffuse])
        weak = np.column_stack([efficiency * self.core_gradient[i], core, np.zeros(len(i))])
        gradient = (a + b) @ full - b @ weak
        self.countable_cache = x.copy(), value, gradient
        return value, gradient

    def all_constraints(self, x):
        margin, jac = super().all_constraints(x)
        support, gradient = self.support(x)
        return np.r_[margin, (29.9 - support) / 100.0], np.vstack([jac, -gradient / 100.0])


def fit(model, state, section, order, budget, *, target_count=1, timed_event=None, spread=False):
    """Find a physically admissible fit with similar light and fewer rings."""
    n = len(model.particles)
    if n < target_count + 1 or (timed_event is None and n != 2):
        return None
    args = model.active[model.ix], model.q, model.bg, model.particles, section
    coarse = BelowSupport(*args, order)
    initial = state["parameters"]
    original = coarse(initial, True)["ll"]
    initial_support = coarse.support(initial)[0]
    if initial_support < 30.0:
        return None
    chart = Junction(coarse, initial)
    result, x = fit_chart(
        coarse, initial, budget, budget, full_likelihood=True, chart=chart, check_initial=True
    )
    if not feasible(coarse, x) or coarse.hardware_margin(x) <= 1e-5:
        return None
    score = coarse(x, True)["ll"]
    if timed_event is None and original - score > 10.0:
        return None
    fine = BelowSupport(*args, 2 * order)
    final = fine(x, True)
    support = fine.support(x)[0]
    error = abs(final["ll"] - score)
    if error >= 0.01 or support >= 30.0:
        return None
    if n == 2:
        share = final["posterior"][2 - section]
        remaining = float(np.sum(1.0 - fine.obs.no_source(final["means_hit"], share)))
        if remaining < 30.0 or not np.any(share > 0.5):
            return None
    else:
        from chain_comparison import counted

        fractions = counted(fine, final)
        if fractions.shape[1] != target_count + 1 or not np.all(
            np.bincount(fractions.argmax(1), minlength=target_count + 1)[1:]
        ):
            return None
        remaining = [float(np.sum(1.0 - fine.obs.no_source(final["means_hit"], f))) for f in fractions.T[1:]]
    detail = dict(
        ll=float(final["ll"]),
        support=initial_support,
        competing_support=support,
        remaining_support=remaining,
        evaluations=int(result.nfev),
        quadrature_ll_error=error,
        competing_parameters=x.tolist(),
    )
    if timed_event is not None:
        from physical_pulse import profile

        timing = profile(timed_event, fine, final, separate_knockon=True, spread=spread)
        if timing is None:
            return None
        detail.update(timing)
    return detail
