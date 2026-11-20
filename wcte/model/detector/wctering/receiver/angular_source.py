"""Angular integration with the original finite-source endpoint certificate."""

import numpy as np
from wctering.receiver.normalize import normalize
from wctering.receiver.cache import evaluate
from wctering.receiver.finite_source import _source, quadrature
from wctering.receiver.angular_panels import finite_integrate


def source(positions, normals, parameters, order=16):
    def light(p, n, x):
        if x[6] == 0.0:
            return _source(p, n, x, order, scalar=True)
        from wctering.receiver.angular_shape import backend, evaluate as compiled

        functions = backend()
        if functions is not None:
            return compiled(
                *functions,
                np.ascontiguousarray(p),
                np.ascontiguousarray(n),
                np.ascontiguousarray(x),
                *quadrature(order),
            )
        raw, jacobian, _ = finite_integrate(p, n, x, *quadrature(order))
        return normalize(raw, jacobian)

    return evaluate(
        light,
        positions,
        normals,
        parameters,
        kind="finite_angular_checked",
        order=order,
    )
