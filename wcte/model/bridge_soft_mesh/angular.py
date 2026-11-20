"""Analytic narrow-cone normalization, with ordinary quadrature for broad light.

Integrate the Gaussian against the solid-angle measure sin(angle). For small
variance, incomplete Gaussian moments give a rapidly convergent sine series.
The omitted upper tail is below 1e-27 over the permitted Cherenkov angles.
This includes the near-threshold boundary at zero angle exactly.
"""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def normalization(theta, variance, angles, weights):
    if variance <= 0.04 and theta <= 0.9:
        moments = np.empty(28)
        edge = math.exp(-theta * theta / (2 * variance))
        moments[0] = math.sqrt(math.pi * variance / 2) * (1 + math.erf(theta / math.sqrt(2 * variance)))
        moments[1] = theta * moments[0] + variance * edge
        for k in range(2, len(moments)):
            moments[k] = theta * moments[k - 1] + (k - 1) * variance * moments[k - 2]
        sine = 0.0
        cosine = 0.0
        cs = 1.0
        cc = 1.0
        for k in range(13):
            sine += cs * moments[2 * k + 1]
            cosine += cc * moments[2 * k]
            cs /= -float((2 * k + 2) * (2 * k + 3))
            cc /= -float((2 * k + 1) * (2 * k + 2))
        # Integration by parts gives I_theta=I_cos and
        # I_variance=I/(2v)+(exp(-theta²/(2v))-I)/2.
        return 2 * math.pi * sine, cosine / sine, 0.5 / variance - 0.5 + 0.5 * edge / sine
    total = 0.0
    angle_derivative = 0.0
    variance_derivative = 0.0
    for j in range(len(angles)):
        residual = angles[j] - theta
        value = weights[j] * math.exp(-0.5 * residual * residual / variance)
        total += value
        angle_derivative += value * residual / variance
        variance_derivative += value * residual * residual / (2 * variance * variance)
    return total, angle_derivative / total, variance_derivative / total
