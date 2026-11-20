"""Detector timing proposals for the one-ring comparison."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def ranked_timing_starts(event, diagnostics, labels, background, count):
    """Rank timing-vertex proposals after fitting only weights and intensity.

    Geometry stays fixed in this inexpensive screen. The returned geometries
    still require the full constrained fit in fitting.py before they can win.
    Rows are (negative likelihood, stable name, geometry, displaced component).
    Component 0 identifies an original seed rather than a displaced ring.
    """
    from wctering.seeding.profile_vertices import profile_seed_bank
    from wctering.likelihood.water import WaterOccupancyModel

    proposals = profile_seed_bank(
        diagnostics,
        count,
        event.pmt_positions_mm,
        event.charges_pe,
        event.times_ns,
        labels,
    )
    ranked = []
    for proposal in proposals:
        rings = proposal["seed"]
        geometry = dict(
            vertices=[ring["vertex"] for ring in rings],
            axes=[ring["axis"] for ring in rings],
            openings_deg=[ring["opening"] for ring in rings],
            widths_deg=[8.0] * count,
            weights=[0.2] + [0.8 / count] * count,
        )
        model = WaterOccupancyModel(event.pmt_ids, event.charges_pe, geometry, background, "point", 16)
        shapes = model(model.x, True)["shapes"]
        observation = model.observation

        def objective(parameters):
            weights = softmax(np.r_[0.0, parameters[:-1]])
            intensity = np.exp(parameters[-1])
            density = weights @ shapes
            mean = intensity * density
            hit = -np.expm1(-mean)
            likelihood = np.sum(observation * np.log(hit) - (1 - observation) * mean)
            derivative = 1 - observation / hit
            residual = np.sum(derivative * mean)
            # The last derivative is with respect to log(intensity). The others
            # include the softmax normalization relative to background.
            if count == 1:
                # Keep the frozen single-ring vector-dot reduction order.
                gradient = np.array(
                    [
                        weights[1] * (intensity * (shapes[1] @ derivative) - residual),
                        residual,
                    ]
                )
            else:
                gradient = np.r_[
                    weights[1:] * (intensity * (shapes[1:] @ derivative) - residual),
                    residual,
                ]
            return -likelihood / model.total, gradient / model.total

        fitted = minimize(
            objective,
            model.x[-count - 1 :],
            jac=True,
            method="L-BFGS-B",
            bounds=model.bounds[-count - 1 :],
            options=dict(maxiter=80, ftol=1e-12, gtol=1e-8),
        )
        geometry["weights"] = softmax(np.r_[0.0, fitted.x[:-1]])
        geometry["initial_occupancy_intensity"] = float(np.exp(fitted.x[-1]))
        ranked.append(
            (
                objective(fitted.x)[0],
                proposal["name"],
                geometry,
                proposal.get("profile_ring", 0),
            )
        )
    return sorted(ranked, key=lambda row: (row[0], row[1]))


def single_ring_starts(event, diagnostics, labels, background, count=3):
    """Refine the three most promising K=1 timing proposals."""
    ranked = ranked_timing_starts(event, diagnostics, labels, background, 1)
    return [row[2] for row in ranked[:count]]
