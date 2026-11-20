"""Guard seed optimizations against lost incumbents and non-finite endpoints."""

import numpy as np


def finite_fit(r):
    return np.isfinite(r["ll"]) and np.all(np.isfinite(r["posterior"]))


def retain_incumbent(raw, optimized=None, error=None):
    """Never mark an unevolved changed-objective incumbent converged."""
    raw = dict(raw, converged=False, iterations=0, retained_incumbent=True)
    if optimized is not None and finite_fit(optimized) and optimized["ll"] >= raw["ll"]:
        return optimized, dict(
            accepted_optimization=True,
            ll_change=float(optimized["ll"] - raw["ll"]),
            error=error,
        )
    change = (
        None if optimized is None or not np.isfinite(optimized["ll"]) else float(optimized["ll"] - raw["ll"])
    )
    return raw, dict(accepted_optimization=False, ll_change=change, error=error)


def guarded_optimizer(original):
    """An explicit supplied-parameter continuation retains its finite start."""

    def fit(*args, **kwargs):
        if kwargs.get("_initial_parameters") is None or kwargs.get("_debug", False):
            return original(*args, **kwargs)
        debug = dict(kwargs, _debug=True)
        callback, _, _ = original(*args, **debug)
        raw = callback(np.asarray(kwargs["_initial_parameters"]), True)
        if not finite_fit(raw):
            raise ValueError("Nonfinite continuation incumbent")
        optimized = None
        error = None
        try:
            optimized = original(*args, **kwargs)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            error = repr(exc)
        result, record = retain_incumbent(raw, optimized, error)
        result = dict(result, continuation_record=record)
        result["labels"] = result["posterior"].argmax(axis=0)
        return result

    return fit


def reevaluate(optical, inputs, fit):
    callback, _, _ = optical.optical_fit(
        *inputs, fit["seed"], _debug=True, _initial_parameters=fit["parameters"]
    )
    r = callback(np.asarray(fit["parameters"]), True)
    r.update(
        labels=r["posterior"].argmax(axis=0),
        converged=False,
        iterations=0,
        family="reevaluated_flat_incumbent",
        changed_objective=True,
    )
    if not finite_fit(r):
        raise ValueError("Nonfinite changed-objective incumbent")
    return r


def continue_fit(optical, inputs, fit, family, maxiter=75):
    r = optical.optical_fit(*inputs, fit["seed"], maxiter=maxiter, _initial_parameters=fit["parameters"])
    r["family"] = family
    return r


def supplement_bank(optical, inputs, base_bank, base_summaries):
    bank = {}
    summaries = {}
    for k, base in base_bank.items():
        initial = base_summaries[k]
        candidates = list(initial["_candidates"]) + [base]
        if k > 1:
            start = max(candidates, key=lambda r: r["ll"])
            for seed in optical._residual_proposals(inputs[1], inputs[2], start):
                try:
                    r = optical.optical_fit(*inputs, seed, maxiter=75)
                    r["family"] = "supplemental_residual"
                    if finite_fit(r):
                        candidates.append(r)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
        best = max(candidates, key=lambda r: r["ll"])
        if not best["converged"]:
            polished = continue_fit(optical, inputs, best, "supplemental_polish", 150)
            candidates.append(polished)
            if polished["ll"] >= best["ll"]:
                best = polished
        if best["ll"] < base["ll"]:
            raise AssertionError("Lost same-objective base incumbent")
        # These TWO head inputs retain the literal original-three-family meaning.
        summaries[k] = dict(
            candidate_ll=list(initial["candidate_ll"]),
            admissible_count=initial["admissible_count"],
            _candidates=candidates,
        )
        bank[k] = best
    return bank, summaries


def partition_fit(optical, inputs, partition):
    """Recover saved fit parameters; timing strength stays exactly zero."""
    if not partition.get("applied", False):
        return None
    seed = [
        dict(
            vertex=np.array(c["vertex_mm"]),
            axis=np.array(c["direction"]),
            opening=c["opening_deg"],
            t0=0.0,
            sigma_t=1.0,
        )
        for c in partition["components"]
    ]
    x = []
    for c in partition["components"]:
        x.extend(
            [
                *optical.axis_angles(c["direction"]),
                np.deg2rad(c["opening_deg"]),
                np.log(np.deg2rad(c["angular_width_deg"])),
            ]
        )
    pi = np.array(partition["mixture_weights"])
    x.extend(np.log(pi[1:] / pi[0]))
    x = np.array(x)
    callback, _, _ = optical.optical_fit(*inputs, seed, _debug=True, _initial_parameters=x)
    r = callback(x, True)
    r.update(
        labels=r["posterior"].argmax(axis=0),
        converged=False,
        iterations=0,
        family="reevaluated_final_partition",
    )
    return r


def final_incumbents(optical, inputs, partition):
    """Recover the selected partition on the identical charge objective."""
    own = partition_fit(optical, inputs, partition)
    if own is None:
        return []
    np.testing.assert_allclose(own["ll"], partition["log_likelihood"], rtol=0, atol=1e-7)
    own.update(
        converged=bool(partition["optimizer_converged"]),
        iterations=int(partition["optimizer_iterations"]),
        family="same_objective_final_incumbent",
    )
    return [own]
