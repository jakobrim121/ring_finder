"""Evaluate many trial rates with the same exact charge probability series."""

import math

from numba import njit
import numpy as np


@njit(cache=True)
def rank_mean_batch(means, hit, offsets, coefficients, miss_one, miss_two):
    """Compute each likelihood in active-PMT order, preserving ties."""
    rows, count = means.shape
    result = np.empty(rows)
    delta = miss_two - miss_one**2
    for j in range(rows):
        score = 0.0
        for i in range(count):
            rate = max(means[j, i], 1e-300)
            index = hit[i]
            if index < 0:
                correction = 0.5 * delta * rate**2 * math.exp(-miss_one * rate)
                score += -(1.0 - miss_one) * rate + math.log1p(correction)
                continue
            log_rate = math.log(rate)
            start, stop = offsets[index], offsets[index + 1]
            lo, hi = start + 2, stop - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if coefficients[mid + 1] - coefficients[mid] + log_rate > 0.0:
                    lo = mid + 1
                else:
                    hi = mid
            peak = max(
                coefficients[lo] + (lo - start + 1) * log_rate,
                coefficients[start] + log_rate,
                coefficients[start + 1] + 2.0 * log_rate,
            )
            prob = 0.0
            for k in range(start, stop):
                term = math.exp(coefficients[k] + (k - start + 1) * log_rate - peak)
                if k > lo and prob + term == prob:
                    break
                prob += term
            score += -rate + peak + math.log(prob)
        result[j] = score
    return result


def shortlist(model, means, penalty, count):
    """Refit the leading likelihood values with the scalar scoring path."""
    score = batch_scores(model, means) - penalty
    order = np.argsort(-score, kind="stable")
    if not len(order):
        return order
    # The vector score accumulates over PMTs in a different order from
    # NumPy's pairwise sum. Preserve near ties for exact scalar reranking.
    cutoff = score[order[min(count, len(order)) - 1]]
    return order[score[order] >= cutoff - 1e-7]


def batch_scores(model, means):
    """Use one hit lookup for every fixed charge hypothesis in this batch."""
    ix = model.ix
    hit = np.full(len(model.active), -1, dtype=int)
    hit[ix] = np.arange(len(ix))
    obs = model.obs
    return rank_mean_batch(means, hit, obs.offsets, obs.coefficients, obs.r, obs.p2)
