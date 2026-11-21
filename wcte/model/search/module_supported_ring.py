"""Estimate support and its uncertainty across detector modules."""

import numpy as np


def lower_bound(ids, support):
    """Estimate a lower support limit with coherent tube responses per mPMT.

    Conditional photon responsibilities already describe independent pulse
    uncertainty. Geometry error is correlated across tubes in a module,
    so report a leave-one-module sensitivity as a second, separate check.
    """
    groups = np.unique(ids // 100, return_inverse=True)[1]
    by_module = np.bincount(groups, weights=support)
    total = float(by_module.sum())
    statistical = float(np.sqrt(support @ (1.0 - support)))
    largest = float(by_module.max(initial=0.0))
    return dict(
        pmt_support=total,
        lower_statistical=total - 1.645 * statistical,
        strongest_module=largest,
        leave_one_module=total - largest,
    )
