"""Retain matched-objective incumbents while extending the optical seed bank."""

import numpy as np

from wctering.seeding.continuation import (
    reevaluate,
    continue_fit,
    finite_fit,
    partition_fit,
)


def matched_bank(optical, inputs, base, summaries, incumbent_bank):
    bank = {}
    output = {}
    for k, initial in base.items():
        summary = summaries[k]
        pool = list(summary["_candidates"]) + [initial]
        extra = []
        # Put the retained incumbent on the same objective as the new starts
        # before comparing likelihoods. This prevents an apparent gain caused
        # only by changing the receiver/background convention.
        raw = reevaluate(optical, inputs, incumbent_bank[k])
        raw.update(changed_objective=False, family="frozen_residual_incumbent")
        incumbent = incumbent_bank[k]
        np.testing.assert_allclose(raw["ll"], incumbent["ll"], rtol=0, atol=1e-7)
        raw.update(converged=bool(incumbent["converged"]), iterations=int(incumbent["iterations"]))
        extra.append(raw)
        extra.append(continue_fit(optical, inputs, raw, "frozen_residual_incumbent_continuation", 75))
        pool += extra
        # Residual proposals move a component toward unexplained light. They
        # supplement the incumbent and three original seed families.
        if k > 1:
            start = max(pool, key=lambda x: x["ll"])
            for seed in optical._residual_proposals(inputs[1], inputs[2], start):
                try:
                    r = optical.optical_fit(*inputs, seed, maxiter=75)
                    r["family"] = "supplemental_residual"
                    if finite_fit(r):
                        pool.append(r)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
        best = max(pool, key=lambda x: x["ll"])
        if not best["converged"]:
            r = continue_fit(optical, inputs, best, "supplemental_polish", 150)
            pool.append(r)
            if r["ll"] >= best["ll"]:
                best = r
        assert best["ll"] >= raw["ll"] and best["ll"] >= initial["ll"]
        bank[k] = best
        # The classifier learned the original family-summary features. Preserve
        # those summaries even when a supplemental fit becomes the geometry seed.
        output[k] = dict(
            candidate_ll=list(summary["candidate_ll"]),
            admissible_count=summary["admissible_count"],
            _candidates=pool,
        )
    return bank, output


def final_pool(optical, inputs, reference):
    """Preserve the saved incumbent and its continuation on the same objective."""
    partition = reference.get("analytic_partition", {})
    raw = partition_fit(optical, inputs, partition)
    if raw is None:
        return []
    np.testing.assert_allclose(raw["ll"], partition["log_likelihood"], rtol=0, atol=1e-7)
    raw.update(
        changed_objective=False,
        family="frozen_residual_final_incumbent",
        converged=bool(partition["optimizer_converged"]),
        iterations=int(partition["optimizer_iterations"]),
    )
    continued = continue_fit(optical, inputs, raw, "frozen_residual_final_incumbent_continuation", 75)
    return [raw, continued]
