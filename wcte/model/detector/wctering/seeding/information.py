"""Nested shared-axis versus separate-axis information models.

The model treats the long multilateration covariance direction as an
undirected propagation-axis estimate.  It does not reward the normalized
anisotropy of each proposed partition.  Instead, it asks how much timing
information is incompatible with forcing all partitions to share one weak
information direction.
"""

from __future__ import annotations

from wctering.seeding.timing_precision import precision_weights

from typing import Any, Iterable

import numpy as np


C_LIGHT_MM_PER_NS = 299.792458


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize and remove only numerical negative eigenvalues."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-9 * scale:
        raise ValueError("information matrix has a material negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


def profiled_vertex_information(
    positions_mm: np.ndarray,
    charges_pe: np.ndarray,
    vertex_mm: np.ndarray,
    *,
    group_index: float,
    sigma_t_ns: float = 1.0,
    charge_weight_power: float = 1.0,
) -> np.ndarray:
    """Return the 3x3 vertex information after profiling the emission time.

    This is the Schur complement of the point-source timing Hessian for

        r_i = sqrt(w_i) [t_i - t0 - |x_i-v|/v_group].

    The hit times do not enter the local information matrix once the fitted
    vertex has been supplied. Hit precision scales with measured charge.
    """
    positions = np.asarray(positions_mm, dtype=float)
    charges = np.asarray(charges_pe, dtype=float)
    vertex = np.asarray(vertex_mm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_mm must have shape (N, 3)")
    if charges.shape != (len(positions),):
        raise ValueError("charges_pe must have shape (N,)")
    if len(positions) < 4:
        raise ValueError("at least four hits are required")
    if not np.isfinite(positions).all() or not np.isfinite(charges).all():
        raise ValueError("positions and charges must be finite")
    if not np.isfinite(vertex).all() or vertex.shape != (3,):
        raise ValueError("vertex_mm must be a finite length-three vector")
    if group_index <= 0.0 or sigma_t_ns <= 0.0:
        raise ValueError("group_index and sigma_t_ns must be positive")

    group_velocity = C_LIGHT_MM_PER_NS / float(group_index)
    weights = precision_weights(charges, charge_weight_power, sigma_t_ns)
    sqrt_weights = np.sqrt(weights)

    vectors = positions - vertex[None, :]
    distances = np.maximum(np.linalg.norm(vectors, axis=1), 1.0e-12)
    jacobian_xyz = sqrt_weights[:, None] * vectors / distances[:, None] / group_velocity
    jacobian_t0 = -sqrt_weights

    hessian_xyz = jacobian_xyz.T @ jacobian_xyz
    coupling = jacobian_xyz.T @ jacobian_t0
    hessian_t0 = float(jacobian_t0 @ jacobian_t0)
    if hessian_t0 <= 0.0:
        raise ValueError("time information is non-positive")
    profiled = hessian_xyz - np.outer(coupling, coupling) / hessian_t0
    return _positive_semidefinite(profiled)


def shared_axis_score(information_matrices: Iterable[np.ndarray]) -> dict[str, Any]:
    """Score the evidence that the partitions require distinct weak axes.

    For information matrices P_k, the cheapest shared weak direction is

        min_u sum_k u^T P_k u = lambda_min(sum_k P_k),

    whereas separate directions cost sum_k lambda_min(P_k).  Their nonnegative
    difference is the information incompatible with a shared axis.
    """
    matrices = [_positive_semidefinite(matrix) for matrix in information_matrices]
    if len(matrices) < 2:
        raise ValueError("at least two information matrices are required")
    if any(matrix.shape != (3, 3) for matrix in matrices):
        raise ValueError("every information matrix must have shape (3, 3)")

    eigenvalues = [np.linalg.eigvalsh(matrix) for matrix in matrices]
    total = _positive_semidefinite(np.sum(matrices, axis=0))
    total_eigenvalues, total_eigenvectors = np.linalg.eigh(total)

    separate_cost = float(sum(values[0] for values in eigenvalues))
    shared_cost = float(total_eigenvalues[0])
    delta = max(shared_cost - separate_cost, 0.0)
    trace = float(np.trace(total))

    weak_axes = []
    traces = []
    for matrix in matrices:
        _values, vectors = np.linalg.eigh(matrix)
        axis = vectors[:, 0]
        axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
        weak_axes.append(axis)
        traces.append(float(np.trace(matrix)))

    result: dict[str, Any] = {
        "n_partitions": int(len(matrices)),
        "partition_information_matrices_per_mm2": matrices,
        "partition_information_eigenvalues_per_mm2": eigenvalues,
        "partition_weak_axes": weak_axes,
        "partition_information_traces_per_mm2": traces,
        "total_information_matrix_per_mm2": total,
        "total_information_eigenvalues_per_mm2": total_eigenvalues,
        "best_shared_weak_axis": total_eigenvectors[:, 0],
        "shared_axis_cost_per_mm2": shared_cost,
        "separate_axes_cost_per_mm2": separate_cost,
        "incompatible_information_per_mm2": delta,
        "trace_normalized_incompatibility": delta / max(trace, 1.0e-30),
        "shared_cost_reduction_fraction": delta / max(shared_cost, 1.0e-30),
        "minimum_partition_information_fraction": min(traces) / max(trace, 1.0e-30),
    }
    if len(weak_axes) == 2:
        cosine = abs(float(np.dot(weak_axes[0], weak_axes[1])))
        result["information_axis_separation_deg"] = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
    return result


def incremental_axis_score(
    information_matrices: Iterable[np.ndarray],
) -> dict[str, Any]:
    """Score whether K partitions require K axes rather than K-1 axes.

    For K partition information matrices, the K-axis cost is the sum of their
    individual minimum eigenvalues.  The best K-1-axis model merges exactly
    one pair and assigns every other partition its own weak axis.  The merged
    pair is selected by the smallest resulting cost.  For K=2 this reduces
    exactly to :func:`shared_axis_score`; for K=3 it is the natural nested
    two-ring versus three-ring comparison.
    """

    matrices = [_positive_semidefinite(matrix) for matrix in information_matrices]
    if len(matrices) < 2:
        raise ValueError("at least two information matrices are required")
    if any(matrix.shape != (3, 3) for matrix in matrices):
        raise ValueError("every information matrix must have shape (3, 3)")

    eigenvalues = [np.linalg.eigvalsh(matrix) for matrix in matrices]
    separate_cost = float(sum(values[0] for values in eigenvalues))
    merge_candidates = []
    for first in range(len(matrices)):
        for second in range(first + 1, len(matrices)):
            merged_cost = float(np.linalg.eigvalsh(matrices[first] + matrices[second])[0])
            unmerged_cost = float(
                sum(eigenvalues[index][0] for index in range(len(matrices)) if index not in (first, second))
            )
            merge_candidates.append(
                {
                    "merged_partition_indices": [int(first), int(second)],
                    "merged_pair_axis_cost_per_mm2": merged_cost,
                    "unmerged_axes_cost_per_mm2": unmerged_cost,
                    "total_k_minus_one_axis_cost_per_mm2": (merged_cost + unmerged_cost),
                }
            )
    best = min(
        merge_candidates,
        key=lambda item: item["total_k_minus_one_axis_cost_per_mm2"],
    )
    previous_cost = float(best["total_k_minus_one_axis_cost_per_mm2"])
    delta = max(previous_cost - separate_cost, 0.0)
    total = _positive_semidefinite(np.sum(matrices, axis=0))
    trace = float(np.trace(total))
    traces = [float(np.trace(matrix)) for matrix in matrices]

    return {
        "n_partitions": int(len(matrices)),
        "null_axis_count": int(len(matrices) - 1),
        "alternative_axis_count": int(len(matrices)),
        "partition_information_matrices_per_mm2": matrices,
        "partition_information_eigenvalues_per_mm2": eigenvalues,
        "partition_information_traces_per_mm2": traces,
        "total_information_matrix_per_mm2": total,
        "separate_axes_cost_per_mm2": separate_cost,
        "best_k_minus_one_axis_cost_per_mm2": previous_cost,
        "best_merged_partition_pair": best["merged_partition_indices"],
        "merge_candidates": merge_candidates,
        "incremental_incompatible_information_per_mm2": delta,
        "trace_normalized_incremental_incompatibility": (delta / max(trace, 1.0e-30)),
        "incremental_cost_reduction_fraction": (delta / max(previous_cost, 1.0e-30)),
        "minimum_partition_information_fraction": (min(traces) / max(trace, 1.0e-30)),
    }
