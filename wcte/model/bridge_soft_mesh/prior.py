"""Normalized, moment-matched multiple-scattering direction prior."""

import numpy as np
import electron as base
from material import material
from mesh import boundaries
from stable_vmf import angular, spline_difference


def prior(x):
    """Log density per steradian of all directions after the first, and d/dx."""
    axes, jac = base.directions(x)
    pieces = len(axes)
    m = material("electron")
    s = m["scatter_at"]
    ds = s.derivative()
    edges, dr = boundaries(x[-1], pieces - 1)
    slopes = ds(edges)
    out = 0.0
    gradient = np.zeros_like(x)
    for j in range(1, pieces):
        q = spline_difference(s, edges[j - 1], edges[j])
        dq = slopes[j - 1] * (dr[j] - dr[j - 1]) + spline_difference(ds, edges[j - 1], edges[j]) * dr[j]
        value, da, db, dvariance = angular(axes[j - 1], axes[j], q)
        out += value
        gradient[3 + 3 * (j - 1) : 6 + 3 * (j - 1)] += jac[j - 1] @ da
        gradient[3 + 3 * j : 6 + 3 * j] += jac[j] @ db
        gradient[-1] += dvariance * dq
    return float(out), gradient
