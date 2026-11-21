"""Directional geometry and derivatives for a two-section trajectory."""

import numpy as np


def turn_cosine(x):
    from chain_sources import paths
    from source import trajectory

    (incoming, da), (outgoing, db) = paths(x[:-2])
    _, tangent, _, derivative = trajectory(incoming, np.ones(1))
    cosine = tangent[0] @ outgoing[3:6]
    slope = tangent[0] @ db[3:6] + outgoing[3:6] @ derivative[0] @ da
    return float(cosine), np.r_[slope, 0.0, 0.0]
