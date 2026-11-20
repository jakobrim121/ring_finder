"""Profile the same two-section history on the one-core side of 15 degrees."""

import numpy as np


def angle_constraint(x):
    from chain_sources import paths
    from source import trajectory

    (incoming, da), (outgoing, db) = paths(x[:-2])
    _, tangent, _, derivative = trajectory(incoming, np.ones(1))
    cosine = tangent[0] @ outgoing[3:6]
    slope = tangent[0] @ db[3:6] + outgoing[3:6] @ derivative[0] @ da
    return float(cosine - np.cos(np.radians(14.98))), np.r_[slope, 0.0, 0.0]
