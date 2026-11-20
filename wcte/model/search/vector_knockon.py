"""Render below-margin knock-on light for scoring a proposed track."""

import ctypes
from functools import lru_cache

import numpy as np

from fast_knockon import integrate as scalar
from lookup import COEFFICIENTS, COUNT, LOW, STEP
from wctering.receiver.build import ARRAY, INTEGERS, load_library


@lru_cache(maxsize=1)
def library():
    handle = load_library(("knockon_batch.c",), vector_math=True)
    if handle is not None:
        handle.knockon_batch.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            *([ARRAY] * 5),
            INTEGERS,
            ARRAY,
            ctypes.c_double,
            ARRAY,
            ctypes.c_size_t,
            ARRAY,
            ctypes.c_size_t,
            ARRAY,
            ARRAY,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ARRAY,
        ]
        handle.knockon_batch.restype = None
    return handle


def integrate(
    positions,
    normals,
    points,
    axes,
    jp,
    jt,
    openings,
    weights,
    length,
    opening_grid,
    polar_grid,
    coefficients,
):
    native = library()
    if native is None:
        # The portable integral supplies derivatives too. A seed score only
        # needs the value, so the omitted parameter derivatives are zero.
        return scalar(
            positions,
            normals,
            points,
            axes,
            jp,
            jt,
            openings,
            np.zeros((len(openings), 12)),
            weights,
            length,
            np.zeros(12),
            opening_grid,
            polar_grid,
            coefficients,
        )[0]
    cells = np.asarray(
        np.clip(np.searchsorted(opening_grid, openings) - 1, 0, len(opening_grid) - 2),
        dtype=np.int64,
    )
    positions, normals, points, axes, openings, weight, angles, polar, spline = (
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (positions, normals, points, axes, openings, weights, opening_grid, polar_grid, coefficients)
    )
    values = np.empty(len(positions))
    native.knockon_batch(
        len(positions),
        len(weight),
        positions,
        normals,
        points,
        axes,
        openings,
        cells,
        weight,
        length,
        angles,
        len(angles),
        polar,
        len(polar),
        spline,
        COEFFICIENTS,
        COUNT,
        LOW,
        STEP,
        values,
    )
    return values
