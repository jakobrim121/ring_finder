"""Fit and certify the one-ring/crossing comparison hypotheses.

This shorter continuation policy belongs to the accepted 6.33-second model.
An incomplete comparison returns control to the complete candidate search.
"""

import math
import warnings

import numpy as np
from scipy.optimize import minimize

from wctering.optimization.checks import feasible
from wctering.likelihood.water import WaterOccupancyModel


def fit_source(event, geometry, background, kind):
    """Fit one occupancy hypothesis and certify its numerical endpoint."""
    model_class = WaterOccupancyModel
    model = model_class(event.pmt_ids, event.charges_pe, geometry, background, kind, 16)
    parameters = model.x.copy()
    if "initial_occupancy_intensity" in geometry:
        parameters[-1] = np.log(geometry["initial_occupancy_intensity"])
    lower, upper = np.asarray(model.bounds).T
    initial_ll = model(parameters, True)["ll"]
    attempts = []
    continuations = 3 if kind == "point" else 2
    # The final point retry rescales the same likelihood.
    for continuation in range(continuations):
        scale = model.total if continuation == 2 else 1.0

        def objective(values):
            value, gradient = model(values)
            return (value * scale, gradient * scale)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            optimized = minimize(
                objective,
                parameters,
                jac=True,
                method="SLSQP",
                bounds=model.bounds,
                constraints=model.constraints,
                options=dict(
                    maxiter=600 if continuation == 2 else 200,
                    ftol=1e-11 if continuation == 2 else 1e-13,
                    disp=False,
                ),
            )
        parameters = np.clip(optimized.x, lower, upper)
        result = model(parameters, True)
        # Certify the constrained endpoint again with twice as many nodes.
        coarse_kkt = model.kkt(parameters, result["gradient"])
        fine_model = model_class(event.pmt_ids, event.charges_pe, geometry, background, kind, 2 * model.order)
        fine_result = fine_model(parameters, True)
        fine_kkt = fine_model.kkt(parameters, fine_result["gradient"])
        error = dict(
            log_density=float(
                abs(np.log(result["expected_charge_all"]) - np.log(fine_result["expected_charge_all"])).max()
            ),
            ll=float(abs(result["ll"] - fine_result["ll"])),
            gradient=float(abs(result["gradient"] - fine_result["gradient"]).max()),
        )
        quadrature_ok = (
            error["log_density"] <= 1e-07 and error["ll"] <= 1e-05 and (error["gradient"] <= 1e-05)
        )
        support_ok = sorted(set(result["posterior"].argmax(0)) - {0}) == list(range(1, model.k + 1))
        qualified = bool(
            feasible(coarse_kkt)
            and feasible(fine_kkt)
            and (coarse_kkt["residual_inf"] <= 1e-06)
            and (fine_kkt["residual_inf"] <= 1e-06)
            and quadrature_ok
            and support_ok
            and np.isfinite(result["posterior"]).all()
            and (result["ll"] >= initial_ll - 1e-07)
        )
        attempts.append(
            dict(
                ll=result["ll"],
                qualified=qualified,
                coarse=coarse_kkt,
                fine=fine_kkt,
                error=error,
                iterations=int(optimized.nit),
                message=str(optimized.message),
            )
        )
        if qualified:
            break
        if not quadrature_ok:
            model = fine_model
    dimension = (9 if kind == "finite" else 8) * model.k + 1
    fitted_geometry = dict(model.geometry(parameters), weights=result["weights"])
    fitted_geometry["occupancy_intensity"] = float(result["expected_total_charge"])
    return dict(
        k=model.k,
        kind=kind,
        occupancy=True,
        crossing=False,
        qualified=bool(qualified),
        ll=result["ll"],
        score=2 * result["ll"] - dimension * math.log(1843),
        posterior=result["posterior"],
        geometry=fitted_geometry,
        attempts=attempts,
    )


def unresolved(fits, metric="score"):
    """A better failed endpoint makes this comparison incomplete."""
    best = best_fit(fits, metric)
    return best is None or any(
        not fit["qualified"] and (metric not in fit or fit[metric] > best[metric] + 1e-7) for fit in fits
    )


# Share the identical qualified-score selection used by the complete search.
from wctering.optimization.fit import best_fit
