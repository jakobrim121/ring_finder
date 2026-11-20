"""Cache the material interpolant used to choose every optical quadrature."""

from contextlib import contextmanager
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def derivative(table):
    return table.derivative()


def panels(path, cut, water):
    """Keep the original nodes with one derivative spline per material."""
    range_mm = 1000.0 * path[9]
    length = range_mm * path[10]
    result = [0.0, 0.25, 0.5, 0.75, 1.0, cut]
    if length > 0:
        power = -float(derivative(water["scatter_at"])(range_mm))
        step = np.exp(2.0 * path[11]) / (power * length)
        while 0.0 < step < 1.0:
            result.append(step)
            step *= 4.0
        for relative in (0.5, 0.25, 0.1, 0.01, 0.001, 0.0001):
            node = (1.0 - relative) / path[10]
            if 0.0 < node < 1.0:
                result.append(node)
    return np.unique(result)


@contextmanager
def cached_panels(enabled):
    """Reuse derivative splines within this fit and restore them on exit."""
    if not enabled:
        yield
        return
    import physical_pulse
    import source

    previous = source.material_panels, physical_pulse.material_panels
    source.material_panels = physical_pulse.material_panels = panels
    try:
        yield
    finally:
        source.material_panels, physical_pulse.material_panels = previous
