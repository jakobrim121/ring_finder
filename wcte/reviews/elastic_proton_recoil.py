"""Constrain a fitted aperture-to-recoil history to elastic hydrogen kinematics."""

import numpy as np

from finite_chains import factory
from kink_boundary import turn_cosine
from lineage_support import compare as count_lineage
from wcte.counting import track_counting


def mass_shell(parameters, medium):
    """Return the child's on-shell kinetic energy and its parameter gradient."""
    parent = 1000.0 * parameters[9] * (1.0 - parameters[10])
    kinetic = float(medium["energy_at"](parent))
    d_parent = np.zeros_like(parameters)
    d_parent[9] = 1000.0 * (1.0 - parameters[10])
    d_parent[10] = -1000.0 * parameters[9]
    d_kinetic = float(medium["energy_at"].derivative()(parent)) * d_parent
    cosine, d_cosine = turn_cosine(parameters)
    m = medium["mass"]
    fraction = kinetic / (kinetic + 2.0 * m)
    u = cosine * cosine * fraction
    du = 2.0 * cosine * fraction * d_cosine + cosine**2 * 2.0 * m / (kinetic + 2.0 * m) ** 2 * d_kinetic
    e = 2.0 * m * u / (1.0 - u)
    return e, (2.0 * m / (1.0 - u) ** 2) * du, kinetic, cosine, d_cosine


def model_for(competition, order, medium):
    model = factory(competition.event, competition.model.bg, ("proton", "proton"), order)
    physical = model.all_constraints
    derivative = medium["energy_at"].derivative()

    def constraints(parameters):
        values, gradient = physical(parameters)
        fit, df, _, cosine, dc = mass_shell(parameters, medium)
        child = 1000.0 * parameters[18]
        kinetic = float(medium["energy_at"](child))
        delta = kinetic - fit
        dv = -df
        dv[18] += float(derivative(child)) * 1000.0
        # Scale to the same size as the fit's range and containment margins.
        return np.r_[values, (18.0 + delta) / 100.0, (18.0 - delta) / 100.0, cosine - 0.35], np.vstack(
            [gradient, dv / 100.0, -dv / 100.0, dc]
        )

    model.all_constraints = constraints
    return model


def revisit(competition, unconstrained, diagnostic):
    from balanced_search import feasible
    from chain_soft_search import canonical
    from matched_turn_split import seed_turn
    from material import material
    from track_coordinates import fit

    medium = material("proton")
    x = unconstrained["parameters"].copy()
    energy, _, parent, cosine, _ = mass_shell(x, medium)
    above = float(medium["energy_at"](medium["margin_mm"]))
    child = 1000.0 * x[18]
    former = float(medium["energy_at"](child))
    diagnostic.update(
        measured_parent=parent,
        on_shell=energy,
        measured_child=former,
        measured_cosine=cosine,
        threshold=above,
    )
    if energy < above or parent - energy >= above or cosine <= 0.35:
        diagnostic["reason"] = "On-shell energies fail the one-visible-daughter model"
        return None
    m = medium["mass"]
    target_cosine = np.sqrt(former * (parent + 2.0 * m) / (parent * (former + 2.0 * m)))
    if not 0.35 <= target_cosine <= 1.0:
        diagnostic["reason"] = "Measured outgoing range has no forward elastic angle"
        return None
    target_turn = float(np.degrees(np.arccos(target_cosine)))
    measured_turn = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if abs(target_turn - measured_turn) > 8.0 or target_turn < track_counting.get().turn_offset(0.02):
        diagnostic["reason"] = "Measured direction is far from hydrogen kinematics"
        return None
    x = seed_turn(x, target_turn)
    for order in (8, 16, 32):
        model = model_for(competition, order, medium)
        guess = canonical(model, x)
        if guess is None or not feasible(model, guess):
            diagnostic["reason"] = "Projected initial path failed constraints"
            return None
        _, x = fit(model, guess, 30, 46, full_likelihood=True, check_initial=True)
        precise = model_for(competition, 2 * order, medium)
        if not feasible(precise, x) or precise.hardware_margin(x) <= 1.0e-5:
            diagnostic["reason"] = "Refit failed containment or kinematics"
            return None
        state = precise(x, True)
        if abs(model(x, True)["ll"] - state["ll"]) > 0.01:
            continue
        result = count_lineage(competition, precise, state)
        if result is None or result["count"] != 2:
            diagnostic["reason"] = "Supported prompt count is below two"
            return None
        expected, _, initial, cosine, _ = mass_shell(x, medium)
        actual = float(medium["energy_at"](1000.0 * x[18]))
        if initial - actual >= above or abs(expected - actual) > 18.01:
            diagnostic["reason"] = "Kinematics failed the final tolerance"
            return None
        return dict(
            result,
            label="hydrogen_elastic_recoil",
            outgoing_kinetic=actual,
            energy_residual=actual - expected,
            other_kinetic=initial - actual,
            outgoing_angle=float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        )
    return None
