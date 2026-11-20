"""Reuse exact geometric constraints when only the light scale changes."""

from contextlib import contextmanager
from functools import wraps

import numpy as np


def cached(function, *, margin):
    @wraps(function)
    def evaluate(model, parameters):
        # Serial models have two final amplitude coordinates. Their
        # containment and recoil constraints depend only on geometry.
        geom = parameters[:-2]
        name = "_last_geometry_margin" if margin else "_last_geometry_constraints"
        saved = getattr(model, name, None)
        if saved is not None and np.array_equal(geom, saved[0]):
            return saved[1]
        value = function(model, parameters)
        setattr(model, name, (geom.copy(), value))
        return value

    return evaluate


@contextmanager
def reuse_geometry(enabled):
    """Cache the unmodified base routines for this event's worker."""
    if not enabled:
        yield
        return
    from bare_model import Model

    old_constraints = Model.all_constraints
    old_margin = Model.hardware_margin
    Model.all_constraints = cached(old_constraints, margin=False)
    Model.hardware_margin = cached(old_margin, margin=True)
    try:
        yield
    finally:
        Model.all_constraints = old_constraints
        Model.hardware_margin = old_margin
