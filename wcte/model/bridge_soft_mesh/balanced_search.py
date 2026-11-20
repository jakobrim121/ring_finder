"""Refit connected cores with the already demonstrated soft-light background.

Only detector-fitted, archived parameters initialize this study. Each trial
has an explicit source-evaluation budget and records every completed stage.
Exhausted or unqualified modes never count as evidence for a reconstruction.
"""

import numpy as np


def feasible(model, x):
    return bool(
        np.isfinite(x).all() and model.inequality(x).min() >= -1e-6 and abs(model.equality(x)).max() < 1e-6
    )
