"""Physical WCTE inner-water volume and contained point-source refits."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


# Exact 16-sided inner-water prism used by WCSim/LicketyFit optical transport.
WCTE_PRISM_N_SIDES = 16
WCTE_PRISM_APOTHEM_MM = 3075.926 / 2.0
WCTE_PRISM_HEIGHT_MM = 2714.235
WCTE_PRISM_Y_CENTER_MM = 424.763
WCTE_PRISM_Y_MIN_MM = WCTE_PRISM_Y_CENTER_MM - 0.5 * WCTE_PRISM_HEIGHT_MM
WCTE_PRISM_Y_MAX_MM = WCTE_PRISM_Y_CENTER_MM + 0.5 * WCTE_PRISM_HEIGHT_MM

WCTE_PRISM_FACE_NORMALS_XZ = np.asarray(
    [
        [
            math.cos(2.0 * math.pi * index / WCTE_PRISM_N_SIDES),
            math.sin(2.0 * math.pi * index / WCTE_PRISM_N_SIDES),
        ]
        for index in range(WCTE_PRISM_N_SIDES)
    ],
    dtype=float,
)

# Containment is part of the v1.3-and-later model contract, not a tuning switch.
ENFORCE_WCTE_VERTEX_CONTAINMENT = True


def wcte_vertex_margin_mm(point: Sequence[float]) -> float:
    """Signed distance-like margin to the closest exact prism inequality."""

    vertex = np.asarray(point, dtype=float)
    if vertex.shape != (3,) or not np.isfinite(vertex).all():
        return -math.inf
    radial_margin = WCTE_PRISM_APOTHEM_MM - float(np.max(WCTE_PRISM_FACE_NORMALS_XZ @ vertex[[0, 2]]))
    return min(
        float(vertex[1] - WCTE_PRISM_Y_MIN_MM),
        float(WCTE_PRISM_Y_MAX_MM - vertex[1]),
        radial_margin,
    )


def wcte_contains_vertex(point: Sequence[float], *, tolerance_mm: float = 1.0e-6) -> bool:
    """Return whether a finite vertex lies in the exact WCTE water prism."""

    return bool(wcte_vertex_margin_mm(point) >= -abs(float(tolerance_mm)))


def project_vertex_into_wcte(point: Sequence[float], *, inset_mm: float = 1.0e-6) -> np.ndarray:
    """Return a finite feasible SLSQP start without coordinate-wise wall bias."""

    vertex = np.asarray(point, dtype=float).copy()
    if vertex.shape != (3,) or not np.isfinite(vertex).all():
        vertex = np.asarray([0.0, WCTE_PRISM_Y_CENTER_MM, 0.0], dtype=float)
    inset = max(float(inset_mm), 0.0)
    vertex[1] = np.clip(vertex[1], WCTE_PRISM_Y_MIN_MM + inset, WCTE_PRISM_Y_MAX_MM - inset)
    xz = vertex[[0, 2]]
    largest_face_coordinate = float(np.max(WCTE_PRISM_FACE_NORMALS_XZ @ xz))
    limit = WCTE_PRISM_APOTHEM_MM - inset
    if largest_face_coordinate > limit:
        # A radial contraction preserves the proposed direction from the detector
        # centre and supplies a feasible start. The constrained optimizer, not this
        # projection, determines the final boundary solution.
        xz *= limit / max(largest_face_coordinate, 1.0e-30)
        vertex[[0, 2]] = xz
    return np.ascontiguousarray(vertex, dtype=float)


def contained_point_source_refit(
    positions_mm: np.ndarray,
    times_ns: np.ndarray,
    sqrt_weights: np.ndarray,
    group_velocity_mm_ns: float,
    initial_parameters: Sequence[float],
    *,
    time_bound_ns: float = 1000.0,
    max_iterations: int = 300,
) -> dict[str, Any]:
    """Refit an outside point-source solution under exact linear prism bounds.

    The ordinary unconstrained solver remains the primary path. This function is
    called only when that result is outside the detector, preserving all interior
    numerical solutions exactly while refusing an unphysical vertex.
    """

    positions = np.ascontiguousarray(positions_mm, dtype=float)
    times = np.ascontiguousarray(times_ns, dtype=float)
    sqrtw = np.ascontiguousarray(sqrt_weights, dtype=float)
    initial = np.asarray(initial_parameters, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_mm must have shape (N,3)")
    if times.shape != (len(positions),) or sqrtw.shape != (len(positions),):
        raise ValueError("times and sqrt_weights must have shape (N,)")
    if initial.shape != (4,) or not np.isfinite(initial).all():
        raise ValueError("initial_parameters must be a finite length-four vector")
    if not np.isfinite(group_velocity_mm_ns) or group_velocity_mm_ns <= 0.0:
        raise ValueError("group_velocity_mm_ns must be finite and positive")
    if not np.isfinite(time_bound_ns) or time_bound_ns <= 0.0:
        raise ValueError("time_bound_ns must be finite and positive")

    start = initial.copy()
    start[:3] = project_vertex_into_wcte(start[:3], inset_mm=1.0e-5)
    start[3] = float(np.clip(start[3], -float(time_bound_ns) + 1.0e-6, float(time_bound_ns) - 1.0e-6))

    def residual_and_jacobian(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vertex = parameters[:3]
        vectors = positions - vertex[None, :]
        distances = np.maximum(np.linalg.norm(vectors, axis=1), 1.0e-12)
        residual = sqrtw * (times - parameters[3] - distances / float(group_velocity_mm_ns))
        jacobian = np.column_stack(
            [
                sqrtw[:, None] * vectors / distances[:, None] / float(group_velocity_mm_ns),
                -sqrtw,
            ]
        )
        return residual, jacobian

    def objective(parameters: np.ndarray) -> float:
        residual, _ = residual_and_jacobian(parameters)
        return float(np.dot(residual, residual))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        residual, jacobian = residual_and_jacobian(parameters)
        return np.ascontiguousarray(2.0 * (jacobian.T @ residual), dtype=float)

    face_matrix = np.zeros((WCTE_PRISM_N_SIDES, 4), dtype=float)
    face_matrix[:, 0] = WCTE_PRISM_FACE_NORMALS_XZ[:, 0]
    face_matrix[:, 2] = WCTE_PRISM_FACE_NORMALS_XZ[:, 1]
    face_constraint = LinearConstraint(
        face_matrix,
        np.full(WCTE_PRISM_N_SIDES, -np.inf, dtype=float),
        np.full(WCTE_PRISM_N_SIDES, WCTE_PRISM_APOTHEM_MM, dtype=float),
    )
    bounds = Bounds(
        [
            -WCTE_PRISM_APOTHEM_MM,
            WCTE_PRISM_Y_MIN_MM,
            -WCTE_PRISM_APOTHEM_MM,
            -float(time_bound_ns),
        ],
        [
            WCTE_PRISM_APOTHEM_MM,
            WCTE_PRISM_Y_MAX_MM,
            WCTE_PRISM_APOTHEM_MM,
            float(time_bound_ns),
        ],
    )
    result = minimize(
        objective,
        start,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=(face_constraint,),
        options={"maxiter": int(max_iterations), "ftol": 1.0e-12, "disp": False},
    )
    parameters = np.asarray(result.x, dtype=float)
    residual, jacobian = residual_and_jacobian(parameters)
    valid = bool(
        parameters.shape == (4,)
        and np.isfinite(parameters).all()
        and wcte_contains_vertex(parameters[:3], tolerance_mm=1.0e-5)
    )
    return {
        "valid": valid,
        "success": bool(result.success),
        "parameters": parameters,
        "chi2": float(np.dot(residual, residual)),
        "normal_matrix": np.ascontiguousarray(jacobian.T @ jacobian, dtype=float),
        "n_iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
        "vertex_margin_mm": float(wcte_vertex_margin_mm(parameters[:3])),
    }
