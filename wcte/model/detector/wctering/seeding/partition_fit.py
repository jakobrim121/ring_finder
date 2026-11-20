"""Timing fits of a hit partition, vertex information, and axis matching.

These functions quantify a proposed partition. They neither choose the final
ring count nor assign final optical charge fractions.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import itertools
import time
import numpy as np
from wctering.seeding.information import profiled_vertex_information
from wctering.seeding.multilateration import PointSourceFitter

FIT_MAX_NFEV = 100


def _undirected_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1.0e-12)
    b /= max(float(np.linalg.norm(b)), 1.0e-12)
    cosine = abs(float(np.dot(a, b)))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def _fit_partition(
    fitter: PointSourceFitter,
    mask: np.ndarray,
    *,
    total_charge_pe: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    started = time.perf_counter()
    fit = fitter.point_source_fit(mask, max_nfev=FIT_MAX_NFEV)
    charge = float(np.sum(fitter.hit_charges[mask]))
    result: dict[str, Any] = {
        "valid": bool(fit.get("valid", False)),
        "n_hits": int(np.count_nonzero(mask)),
        "n_unique_pmts": int(len(np.unique(fitter.pmt_ids[mask]))),
        "n_unique_modules": int(len(np.unique(fitter.pmt_ids[mask] // 100))),
        "charge_pe": charge,
        "charge_fraction": charge / max(float(total_charge_pe), 1.0e-30),
        "fit_runtime_s": float(time.perf_counter() - started),
    }
    if not result["valid"]:
        result["reason"] = str(fit.get("reason", "invalid point-source fit"))
        return result

    # Symmetrize solver covariance before diagonalizing. The separately profiled
    # timing information removes the common emission-time nuisance parameter.
    covariance = 0.5 * (
        np.asarray(fit["cov"], dtype=float)[:3, :3] + np.asarray(fit["cov"], dtype=float)[:3, :3].T
    )
    covariance_eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    information = profiled_vertex_information(
        fitter.pmt_locs[mask],
        fitter.hit_charges[mask],
        np.asarray(fit["vertex"], dtype=float),
        group_index=float(fitter.n_group),
        sigma_t_ns=float(fitter.sigma_t_ns),
        charge_weight_power=float(fitter.charge_weight_power),
    )
    # eigh returns ascending eigenvalues: column zero is the least constrained
    # vertex direction, used as an undirected axis in the multiplicity evidence.
    information_eigenvalues, information_eigenvectors = np.linalg.eigh(information)
    weak_axis = information_eigenvectors[:, 0]
    weak_axis /= max(float(np.linalg.norm(weak_axis)), 1.0e-30)
    result.update(
        fit_success=bool(fit.get("success", True)),
        solver=str(fit.get("solver", "scipy_reference")),
        vertex_mm=np.asarray(fit["vertex"], dtype=float),
        t0_ns=float(fit["eps"]),
        chi2=float(fit["chi2"]),
        ndof=int(fit["ndof"]),
        reduced_chi2=float(fit["red_chi2"]),
        covariance_xyz_mm2=covariance,
        covariance_eigenvalues_mm2=covariance_eigenvalues,
        covariance_major_axis=np.asarray(fit["major_axis"], dtype=float),
        covariance_anisotropy=float(fit["anisotropy"]),
        covariance_condition_number=float(
            covariance_eigenvalues[-1] / max(float(covariance_eigenvalues[0]), 1.0e-30)
        ),
        profiled_information_matrix_per_mm2=information,
        profiled_information_eigenvalues_per_mm2=information_eigenvalues,
        information_weak_axis=weak_axis,
        covariance_information_axis_difference_deg=_undirected_angle_deg(fit["major_axis"], weak_axis),
        vertex_containment=dict(fit.get("vertex_containment", {})),
    )
    return result


def _axis_set_alignment(
    axes: Sequence[Sequence[float]], references: Sequence[Sequence[float]]
) -> dict[str, Any]:
    if len(axes) != len(references):
        return {"valid": False, "reason": "axis/reference count mismatch"}
    costs = np.asarray(
        [[_undirected_angle_deg(axis, reference) for reference in references] for axis in axes],
        dtype=float,
    )
    # Labels can permute between module-fold refits. Match the tiny axis sets
    # exhaustively (at most 3!), using mean angle then RMS to break ties.
    candidates = []
    for permutation in itertools.permutations(range(len(references))):
        errors = np.asarray(
            [costs[index, permutation[index]] for index in range(len(axes))],
            dtype=float,
        )
        candidates.append(
            (
                float(np.mean(errors)),
                float(np.sqrt(np.mean(errors**2))),
                permutation,
                errors,
            )
        )
    mean_error, rms_error, permutation, errors = min(candidates, key=lambda item: item[:2])
    return {
        "valid": True,
        "cost_matrix_deg": costs,
        "best_reference_index_for_each_axis": [int(value) for value in permutation],
        "matched_axis_errors_deg": errors,
        "mean_matched_axis_error_deg": mean_error,
        "rms_matched_axis_error_deg": rms_error,
        "maximum_matched_axis_error_deg": float(np.max(errors)),
    }
