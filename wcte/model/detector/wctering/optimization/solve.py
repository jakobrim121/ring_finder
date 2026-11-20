"""Optimize optical charge geometry and certify the endpoint at two orders."""

import traceback
import copy
import numpy as np
from scipy.optimize import minimize

from wctering.optimization.checks import Stationary, feasible, could_be_stationary


def certify(create, parameters, order, initial_ll, convert):
    """Check constraints, stationarity and quadrature at the same endpoint."""
    model, fine = create(order), create(max(32, 2 * order))
    coarse_state = convert(model, model(parameters, True))
    coarse_kkt = model.kkt(parameters, coarse_state["gradient"])
    if model.quadrature_independent:
        # Two point-source quadrature orders are the very same calculation.
        fine_state = coarse_state
        fine_kkt = copy.deepcopy(coarse_kkt)
    else:
        fine_state = convert(fine, fine(parameters, True))
        fine_kkt = fine.kkt(parameters, fine_state["gradient"])
    error = dict(
        log_density=float(
            abs(np.log(coarse_state["expected_charge_all"]) - np.log(fine_state["expected_charge_all"])).max()
        ),
        ll=float(abs(coarse_state["ll"] - fine_state["ll"])),
        gradient=float(abs(coarse_state["gradient"] - fine_state["gradient"]).max()),
    )
    quadrature_ok = error["log_density"] <= 1e-7 and error["ll"] <= 1e-5 and error["gradient"] <= 1e-5
    usable = bool(
        feasible(coarse_kkt)
        and feasible(fine_kkt)
        and coarse_kkt["residual_inf"] <= 1e-6
        and fine_kkt["residual_inf"] <= 1e-6
        and quadrature_ok
        and coarse_state["ll"] >= initial_ll - 1e-7
    )
    np.testing.assert_allclose(coarse_state["posterior"].sum(0), 1, atol=2e-14, rtol=0)
    return dict(
        parameters=parameters.copy(),
        order=order,
        usable=usable,
        ll=coarse_state["ll"],
        initial_ll=initial_ll,
        stationarity=coarse_kkt,
        fine_stationarity=fine_kkt,
        refinement=error,
        quadrature_passed=quadrature_ok,
        geometry=model.geometry(parameters),
        posterior=coarse_state["posterior"],
        weights=coarse_state["weights"],
        expected_charge_all=coarse_state["expected_charge_all"],
        hard_positive_labels=sorted(int(v) for v in set(coarse_state["posterior"].argmax(0)) - {0}),
    )


def _run(
    create,
    start,
    initialize,
    stages,
    *,
    convert,
    scaled,
):
    """Run V101's initial fit or its failed-fit continuation on one CPU.

    Three attempts are available. Quadrature disagreement doubles the order;
    otherwise only one extra polishing attempt is allowed. The callback can
    stop a fit only after five iterations and a tighter KKT certificate.
    """
    parameters = np.asarray(start, float).copy()
    order = 16
    model = create(order)
    lower, upper = np.asarray(model.bounds).T
    violation = float(abs(np.clip(parameters, lower, upper) - parameters).max())
    if violation > 1e-8:
        raise ValueError(f"Restart outside original box: {violation}")
    parameters = np.clip(parameters, lower, upper)
    parameters, profile_note = initialize(model, parameters)
    polished = False
    for attempt in range(3):
        model = create(order)
        initial_ll = convert(model, model(parameters, True))["ll"]
        iterations = 0
        callback_clips = []
        kkt_checks_skipped = 0

        def callback(values):
            nonlocal iterations, kkt_checks_skipped
            bounded = np.clip(values, lower, upper)
            clip = float(abs(bounded - values).max())
            if clip:
                callback_clips.append(clip)
            if clip > 1e-8:
                return
            state = model(bounded, "optimizer")
            iterations += 1
            if iterations >= 5:
                if not could_be_stationary(
                    model,
                    bounded,
                    state["gradient"],
                    1e-7,
                    1e-9,
                ):
                    kkt_checks_skipped += 1
                    return
                certificate = model.kkt(bounded, state["gradient"])
                if (
                    certificate["residual_inf"] <= 1e-7
                    and feasible(certificate, 1e-9)
                    and convert(model, state)["ll"] >= initial_ll - 1e-7
                ):
                    raise Stationary(bounded)

        def objective(values):
            value, gradient = model(values)
            scale = model.total
            return value * scale, gradient * scale

        try:
            optimized = minimize(
                objective if scaled else model,
                parameters,
                jac=True,
                method="SLSQP",
                bounds=model.bounds,
                constraints=model.constraints,
                callback=callback,
                options=dict(
                    maxiter=600 if scaled else 300,
                    ftol=1e-11 if scaled else 1e-13,
                    disp=False,
                ),
            )
            parameters = optimized.x
            violation = float(abs(np.clip(parameters, lower, upper) - parameters).max())
            if violation > 1e-8:
                raise ValueError(f"Final box violation: {violation}")
            parameters = np.clip(parameters, lower, upper)
            optimizer = dict(
                success=bool(optimized.success),
                nit=int(optimized.nit),
                nfev=int(optimized.nfev),
                message=str(optimized.message),
                final_box_clip=violation,
            )
        except Stationary as stopped:
            parameters = stopped.x
            optimizer = dict(
                success=None,
                nit=iterations,
                nfev=model.calls,
                message="Certified stationarity callback",
                final_box_clip=0.0,
            )
        optimizer["callback_box_clips"] = callback_clips
        optimizer["kkt_checks_skipped"] = kkt_checks_skipped
        stage = certify(create, parameters, order, initial_ll, convert)
        stage.update(stage=attempt, optimizer=optimizer)
        stages.append(stage)
        if stage["usable"]:
            break
        if (
            not stage["quadrature_passed"]
            or stage["stationarity"]["residual_inf"] <= 1e-6 < stage["fine_stationarity"]["residual_inf"]
        ):
            order *= 2
        elif polished:
            break
        else:
            polished = True
    good = [stage for stage in stages if stage["usable"]]
    chosen = max(good or stages, key=lambda stage: stage["ll"])
    result = dict(chosen, stages=stages, amplitude_initialization=profile_note)
    return dict(result, status="completed")


class PreparedSolution(Exception):
    def __init__(self, result, note):
        self.result, self.note = result, note


def _solve(create, start, initialize, stages, **options):
    """Reuse a certified solve in the same rounded-start bucket.

    FitSearch fixes the event and background. The key also distinguishes the
    physical domain, model family, bounds, training mask, precision and solver
    mode. Initialized parameters are rounded to six decimal places only for lookup;
    original unrounded values enter every new solve. This equates nearby starts
    and is an approximation, not arithmetic identity. No ring permutation or
    event identity participates. Full count/F1 regression is required.
    """
    prepared = {}

    def initialize_once(model, parameters):
        values, note = initialize(model, parameters)
        cache = getattr(model, "prepared_cache", None)
        if cache is not None:
            key = (
                type(model),
                model.kind,
                model.k,
                model.order,
                model.training_mask.tobytes(),
                np.asarray(model.bounds).tobytes(),
                model.water_upper.tobytes(),
                np.round(values, 6).tobytes(),
                options["scaled"],
            )
            if key in cache:
                raise PreparedSolution(cache[key], note)
            prepared.update(cache=cache, key=key)
        return values, note

    try:
        result = _run(create, start, initialize_once, stages, **options)
    except PreparedSolution as reused:
        result = copy.deepcopy(reused.result)
        result["amplitude_initialization"] = reused.note
        stages.extend(result["stages"])
        return result
    if prepared and result.get("usable"):
        prepared["cache"][prepared["key"]] = copy.deepcopy(result)
    return result


def solve(
    create,
    start,
    initialize,
    *,
    convert,
    scaled=False,
):
    """Return a completed fit or an explicit failure with all earlier stages.

    In particular, an exception during a continuation must not erase a higher
    failed endpoint. The count decision still needs that endpoint for its veto.
    """
    stages = []
    try:
        return _solve(
            create,
            start,
            initialize,
            stages,
            convert=convert,
            scaled=scaled,
        )
    except Exception as error:
        return dict(
            status="failed",
            usable=False,
            stages=stages,
            error=repr(error),
            traceback=traceback.format_exc(),
        )
