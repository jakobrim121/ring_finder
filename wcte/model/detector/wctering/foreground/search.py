"""Refine detector-only cone starts for the one-ring charge fit."""

import numpy as np
from scipy.optimize import minimize

from wctering.seeding.optical_seed_fit import optical_fit
from wctering.seeding.profile_vertices import LOG_BAND, profile_seed_bank


def describe(objective, values, success, bounds, foreground):
    """Certify a finite endpoint and keep only its geometry and restart state."""
    x = np.asarray(values, float).copy()
    fit = objective(x, True)
    labels = fit["posterior"].argmax(axis=0)
    labels[foreground == 0] = 0
    finite = bool(np.isfinite(fit["ll"]) and np.isfinite(x).all() and np.isfinite(fit["posterior"]).all())
    return dict(
        ll=float(fit["ll"]),
        finite=finite,
        admissible=finite and set(map(int, labels)) - {0} == {1},
        converged=bool(success),
        parameters=x,
        bounds=bounds,
        geometry=dict(
            vertices=fit["vertices"],
            axes=fit["axes"],
            openings_deg=fit["openings_deg"],
            widths_deg=fit["widths_deg"],
            weights=fit["weights"],
        ),
    )


def fit_start(inputs, foreground, entry):
    """Refine one start and allow two retries when the endpoint is incomplete."""

    def callback():
        return optical_fit(*inputs, entry["seed"], _debug=True)

    def solve(objective, start, bounds):
        fitted = minimize(
            objective,
            np.asarray(start, float).copy(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=110, ftol=1e-8, gtol=1e-5, maxls=20),
        )
        return describe(objective, fitted.x, fitted.success, bounds, foreground)

    objective, start, bounds = callback()
    incumbent = solve(objective, start, bounds)
    for _ in range(2):
        if not incumbent["finite"]:
            break
        objective, _, bounds = callback()
        start = incumbent["parameters"]
        np.testing.assert_array_equal(np.asarray(bounds), incumbent["bounds"])
        np.testing.assert_allclose(objective(start, True)["ll"], incumbent["ll"], rtol=0, atol=1e-9)
        if incumbent["converged"]:
            break
        try:
            candidate = solve(objective, start, bounds)
        except Exception:
            continue
        if candidate["finite"] and candidate["ll"] >= incumbent["ll"]:
            incumbent = candidate
    return incumbent


def fit_starts(reference, inputs, foreground):
    candidates = profile_seed_bank(reference, 1, *inputs[1:], foreground)
    eligible = []
    for index, candidate in enumerate(candidates):
        fit = fit_start(inputs, foreground, candidate)
        if fit["admissible"]:
            eligible.append((2 * index, candidate["name"], fit))
    if not eligible:
        return []

    # Distinct near-optimal vertices can seed different downstream charge basins.
    best = max(fit["ll"] for _, _, fit in eligible)
    result = []
    for index, name, fit in eligible:
        if best - fit["ll"] > LOG_BAND + 1e-10:
            continue
        geometry = fit["geometry"]
        assert all(np.isfinite(np.asarray(values, float)).all() for values in geometry.values())
        assert np.all(np.asarray(geometry["weights"]) > 0)
        np.testing.assert_allclose(np.sum(geometry["weights"]), 1, atol=1e-12, rtol=0)
        # Give the charge fit its own arrays because it can normalize an axis in place.
        result.append((f"r330_core_{index}_{name}", {key: values.copy() for key, values in geometry.items()}))
    return result
