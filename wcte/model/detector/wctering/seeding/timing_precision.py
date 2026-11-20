"""Per-hit timing weights for the point-source fit and Fisher information."""

import numpy as np


def precision_weights(charges, power=1.0, reference_sigma_ns=1.0):
    effective = np.maximum(np.asarray(charges, dtype=float), 1.0e-6) ** float(power)
    return effective / float(reference_sigma_ns) ** 2
