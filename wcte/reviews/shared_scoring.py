"""Match ring columns between two predictions by shared prompt PE."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_charge_shares(predicted, actual, charge):
    """Return paired ring indices and their soft Dice overlaps."""
    predicted = predicted[:, 1:].T
    weight = np.minimum(charge, 2.0)
    overlap = np.minimum(predicted[:, None, :], actual[None, :, :]) @ weight
    total = (predicted @ weight)[:, None] + (actual @ weight)[None, :]
    dice = np.divide(2.0 * overlap, total, out=np.zeros_like(total), where=total > 0.0)
    row, col = linear_sum_assignment(-dice)
    return row, col, dice[row, col]
