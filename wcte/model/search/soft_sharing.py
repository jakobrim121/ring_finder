"""Average plausible same-count charge partitions across geometry modes."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def average(winner, alternatives, charges, temperature=12.0):
    """Align rings by shared PE before averaging their probabilities.

    Closely scored geometry modes can disagree about the boundary between
    two rings. Require each mode to identify the same rings before it can
    affect that boundary. Count selection stays upstream of this step.
    """
    gain, fractions, record = winner
    count = record["count"]
    if count < 1:
        return fractions, []
    q = np.minimum(charges, 2.0)
    base = fractions[:, 1:]
    total = fractions.copy()
    weight_sum = 1.0
    weights = []
    for score, candidate, detail in alternatives:
        if detail is record or detail["count"] != count or score < gain - 4 * temperature:
            continue
        other = candidate[:, 1:]
        overlap = (base.T * q) @ other
        mass = (base.T @ q)[:, None] + (other.T @ q)[None, :]
        similarity = 2.0 * overlap / np.maximum(mass, 1e-15)
        _, match = linear_sum_assignment(-similarity)
        if np.min(similarity[np.arange(count), match]) < 0.38:
            continue
        weight = float(np.exp(min(0.0, (score - gain) / temperature)))
        total += weight * candidate[:, np.r_[0, 1 + match]]
        weight_sum += weight
        weights.append((detail["family"], weight))
    result = total / weight_sum
    if set(result.argmax(1)) - {0} != set(range(1, count + 1)):
        return fractions, []
    return result, weights
