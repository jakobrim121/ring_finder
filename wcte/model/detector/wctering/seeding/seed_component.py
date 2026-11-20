"""One seed cone's light and log-shape derivatives, without Python temporaries."""

import numpy as np
from numba import njit
from wctering.receiver.numpy_loops import unary, matmul
from wctering.logmath import logsumexp_vector
from wctering.seeding.optical_seed_fit import shape_factors


@njit(cache=True)
def component(function, loops, rays, axis, axis_derivative, log_exposure, opening, width, variance):
    cosine = matmul(function, loops[3], rays, axis.reshape(3, 1))[:, 0]
    cosine = np.clip(cosine, -1 + 1e-10, 1 - 1e-10)
    angle = unary(function, loops[6], cosine)
    projection = matmul(function, loops[3], rays, axis_derivative.T)
    logkernel, jacobian = shape_factors(angle, cosine, projection, log_exposure, opening, width, variance)
    normalizer, finite = logsumexp_vector(function, loops, logkernel)
    if not finite:
        raise FloatingPointError("Nonfinite seed normalization")
    shape = unary(function, loops[4], logkernel - normalizer)
    mean = matmul(function, loops[3], shape.reshape(1, len(shape)), jacobian)[0]
    for row in range(len(shape)):
        for column in range(4):
            jacobian[row, column] -= mean[column]
    return shape, jacobian
