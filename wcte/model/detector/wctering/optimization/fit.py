"""Profile amplitudes and fit point or finite tracks at the selected seed count."""

import math
from functools import lru_cache

import numpy as np

from wctering.likelihood.charge import ChargeModel
from wctering.optimization.checks import convert, profile
from wctering.optimization.solve import solve
from wctering.serialization import copy_geometry


def compact_attempt(value):
    """Retain certificates and failed endpoint scores, without large duplicates.

    The selected fit already owns its parameters, geometry and posterior.
    Count selection needs each discarded stage's score and qualification, not
    another copy of its complete prediction. Keeping those scores is essential:
    a better but unqualified endpoint can block a count comparison.
    """
    if isinstance(value, dict):
        return {
            k: compact_attempt(v)
            for k, v in value.items()
            if k
            not in (
                "posterior",
                "expected_charge_all",
                "parameters",
                "geometry",
                "weights",
            )
        }
    if isinstance(value, list):
        return [compact_attempt(v) for v in value]
    return value


def fit_source(event, geometry, background, kind, *, cache):
    geometry = copy_geometry(geometry)

    @lru_cache(maxsize=None)
    def create(order):
        model = ChargeModel(event.pmt_ids, event.charges_pe, geometry, background, kind, order)
        model.prepared_cache = cache.setdefault("prepared_solutions", {})
        return model

    model = create(16)
    parameters = model.x.copy()
    if kind == "finite":
        for j, length in enumerate(geometry.get("lengths_mm", [])):
            if length is not None and length > 1e-6:
                parameters[9 * j : 9 * j + 3] = np.asarray(geometry["vertices"][j]) / 1000.0
                parameters[9 * j + 3 : 9 * j + 6] = geometry["axes"][j]
                parameters[9 * j + 6] = length / 1000.0
    initial_parameters = parameters.copy()
    fingerprint = kind, parameters.tobytes()
    if fingerprint in cache:
        return dict(cache[fingerprint], reused=True)

    # Only feasible seed geometries enter the optical optimizer.
    margin = float((model.water_upper - model.water_values(parameters)).min())
    box = min(min(parameters[i] - lo, hi - parameters[i]) for i, (lo, hi) in enumerate(model.bounds))
    axis = float(abs(model.axis_norms(parameters) - 1).max())
    if margin < -1e-7 or box < -1e-8 or axis > 1e-7:
        return dict(
            k=model.k,
            kind=kind,
            occupancy=False,
            qualified=False,
            excluded=True,
            water_margin=margin,
            box_margin=box,
            axis_error=axis,
        )

    # Retry the last finite endpoint with a rescaled objective if needed.
    attempts = []
    for scaled in (False, True):
        result = solve(create, parameters, profile, scaled=scaled, convert=convert)
        attempts.append(result)
        if result["usable"]:
            break
        if "parameters" in result:
            parameters = result["parameters"].copy()
    good = [r for r in attempts if r["usable"]]
    available = [r for r in attempts if "ll" in r]
    chosen = max(good or available, key=lambda r: r["ll"]) if available else attempts[-1]
    dimension = (9 if kind == "finite" else 8) * model.k + 1
    fit = dict(
        k=model.k,
        kind=kind,
        occupancy=False,
        crossing=False,
        qualified=bool(good) and chosen["hard_positive_labels"] == list(range(1, model.k + 1)),
        numerically_usable=bool(good),
        water=True,
        initial_parameters=initial_parameters,
        seed_geometry=geometry,
        attempts=compact_attempt(attempts),
        amplitude_profile=None,
    )
    if "ll" in chosen:
        fit.update(
            parameters=chosen["parameters"].copy(),
            order=chosen["order"],
            ll=chosen["ll"],
            score=2 * chosen["ll"] - dimension * math.log(1843),
            posterior=chosen["posterior"],
            geometry=dict(chosen["geometry"], weights=chosen["weights"]),
        )
    cache[fingerprint] = fit
    return fit


def best_fit(fits, metric="score"):
    """Highest qualified score, preserving the declared order on exact ties."""
    good = [fit for fit in fits if fit["qualified"]]
    return max(good, key=lambda fit: fit[metric]) if good else None
