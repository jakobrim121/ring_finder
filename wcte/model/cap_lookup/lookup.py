"""Numerical acceleration of the existing analytic spherical-cap response.

The geometry and acceptance law are unchanged. Quintic Hermite intervals use
its exact value, first derivative, and analytic second derivative. Close to
the grazing endpoints we evaluate the original formula, because higher
derivatives there are singular. Return the derivative of the interpolating
polynomial, so the optimizer always differentiates the value it receives.
"""

import numpy as np
from numba import njit
from wctering.receiver.point_source import cap as exact, T, T2, Z0

COUNT = 2048
LOW, HIGH = -T + 0.005, T - 0.005
STEP = (HIGH - LOW) / COUNT
grid = np.linspace(LOW, HIGH, COUNT + 1)
values = np.array([exact(float(mu)) for mu in grid])
second = 2 * Z0 * np.sqrt(T2 - grid * grid) / (np.pi * T2 * (1 - grid * grid) ** 2)
COEFFICIENTS = np.empty((COUNT, 6))
for j in range(COUNT):
    a0 = values[j, 0]
    a1 = STEP * values[j, 1]
    a2 = 0.5 * STEP * STEP * second[j]
    v = values[j + 1, 0] - a0 - a1 - a2
    d = STEP * values[j + 1, 1] - a1 - 2 * a2
    c = STEP * STEP * second[j + 1] - 2 * a2
    COEFFICIENTS[j] = a0, a1, a2, 10 * v - 4 * d + 0.5 * c, -15 * v + 7 * d - c, 6 * v - 3 * d + 0.5 * c


@njit(cache=True)
def cap(mu):
    if mu <= LOW or mu >= HIGH:
        return exact(mu)
    index = min(int((mu - LOW) / STEP), COUNT - 1)
    t = (mu - (LOW + index * STEP)) / STEP
    a, b, c, d, e, f = COEFFICIENTS[index]
    value = a + t * (b + t * (c + t * (d + t * (e + t * f))))
    derivative = (b + t * (2 * c + t * (3 * d + t * (4 * e + t * 5 * f)))) / STEP
    return value, derivative
