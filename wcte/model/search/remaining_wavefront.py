"""Check for a third measured wavefront after a connected two-track refit."""

import numpy as np
from scipy.optimize import minimize

from chain_comparison import MARGIN
from charge_starts import weighted_quantile
from finite_residual_fit import local_fit, optical_shape
from gated_photon_time import evaluate
from joint_prompt_fit import wavefront
from module_optics import conditional
from module_supported_ring import lower_bound
from resolved_ring_evidence import first_pulse_support
from wcte.counting import track_counting


def remains(compare, fit, mean, path, photons, budget, *, separate_yields=False):
    """Ask whether an additional cone retains calibrated pulse evidence."""
    from diffuse_timing import fit_mixture

    model, event = compare.model, compare.event
    if separate_yields:
        from timed_two_track import restore
        from transport_count_review import fit_physical, responses

        restored = restore(event, model.bg, fit.record)
        charged, state, _ = restored
        response = responses(event, model, fit.record, restored)
        if response is None:
            return False, None
        calibrated = fit_physical(
            event, model, fit.record, restored, compare, rate_free=True, response=response
        )
        if calibrated is None:
            return False, None
        fit, mean, multiplier = calibrated
        # Keep core and slowing tail tied to their track. This permits
        # light to transfer from a charged leg to its third-source rival,
        # rather than forcing the third source to add on top of an already
        # saturated fit. Relative clocks of the two tracks stay shared.
        strength = response.shape * multiplier[response.groups, None]
        groups = np.r_[response.groups, len(multiplier)]
        basis = np.zeros((len(multiplier), len(model.active)))
        np.add.at(basis, response.groups, strength)
        amp = basis.sum(1)
        bounds = np.exp(np.asarray(charged.bounds[-2]) - state["parameters"][-2])
        ranges = {j: tuple(amp[j] * bounds / multiplier[j]) for j in range(1, len(amp))}
    refined = local_fit(model, path, photons, mean, budget)
    seeds = [(path, photons)] if refined is None else [(path, photons), refined[:2]]
    best = None
    for path, photons in seeds:
        profile = optical_shape(model, path, 32)[0]
        hit = profile[model.ix]
        if not hit.any():
            continue
        flight = wavefront(event.pmt_positions_mm, path)[0]
        if separate_yields:
            shape = np.vstack([basis / amp[:, None], profile])
            pieces = np.vstack([strength / amp[response.groups, None], profile])
            delays = np.r_[np.zeros(response.time.shape[1], dtype=int), 1]
            mixed = fit_mixture(
                event,
                model,
                shape,
                np.r_[amp, photons],
                np.column_stack([response.time, flight]),
                return_detail=True,
                amplitude_bounds=ranges,
                clock_groups=delays,
                timing_components=(groups, pieces),
            )
            if mixed is None:
                continue
            _, light, arrival = mixed
            expectation = light @ shape
            rates = (light[groups, None] * pieces[:, model.ix]).T
            report = evidence(compare, fit, mean, expectation, rates, arrival)
            # The measured reference has one unconstrained normalization
            # per leg as well, so this tests the extra geometry fairly.
            if best is None or report["gain"] > best["gain"]:
                best = report
            if (
                report["gain"] > MARGIN
                and report["module_gain"] > 0.0
                and report["support"]["lower_statistical"] >= track_counting.get().min_track_pmts
            ):
                return True, report
            continue
        report = fixed_extra(compare, fit, mean, profile, flight, photons)
        if report is None:
            continue
        if best is None or report["gain"] > best["gain"]:
            best = report
        if (
            report["gain"] > MARGIN
            and report["module_gain"] > 0.0
            and report["support"]["lower_statistical"] >= track_counting.get().min_track_pmts
        ):
            return True, report
    return False, best


def fixed_extra(compare, fit, mean, profile, flight, photons):
    """Profile one extra source's PE yield and relative birth time."""
    model = compare.model
    hit = profile[model.ix]
    offset = weighted_quantile(compare.time - flight, compare.q * hit, [0.5])[0]

    def measure(x, detail=False):
        a = np.exp(x[0])
        expectation = mean + a * profile
        rates = np.column_stack([fit.rates, a * hit])
        arrival = np.column_stack([fit.times, flight + x[1]])
        if detail:
            return expectation, rates, arrival
        charge, slope = model.obs.score(expectation, model.ix)
        timing, d_rate, d_clock = evaluate(
            compare.time,
            *compare.gate,
            arrival,
            rates,
            compare.sigma,
            model.obs.offsets,
            model.obs.coefficients,
        )
        gradient = np.array([a * (profile @ slope - d_rate[:, -1] @ hit), -d_clock[:, -1].sum()])
        return -(charge + timing.sum()) / model.total, gradient / model.total

    peak = max(model.total, 1.0)
    initial = np.clip(photons, 0.01, peak)
    solved = minimize(
        measure,
        np.r_[np.log(initial), offset],
        jac=True,
        bounds=[(-18.0, np.log(peak)), (offset - 4.0, offset + 4.0)],
        method="L-BFGS-B",
        options=dict(maxiter=24, ftol=1e-9),
    )
    if not np.isfinite(solved.fun):
        return None
    return evidence(compare, fit, mean, *measure(solved.x, detail=True))


def evidence(compare, fit, mean, expectation, rates, arrival):
    """Evaluate one additional ring against its matched two-ring control."""
    model, event = compare.model, compare.event
    mask = 1 << (arrival.shape[1] - 1)
    supported = first_pulse_support(
        compare.time,
        compare.gate,
        rates,
        arrival,
        compare.sigma,
        -1,
        model.obs.offsets,
        model.obs.coefficients,
        excluded_mask=mask,
    )
    light = lower_bound(event.pmt_ids, supported)
    measured = compare.measured(
        fit.fractions,
        fit.record,
        expectation,
        rates,
        arrival,
        fit.penalty / compare.log_n + 10.0,
        (*fit.core_masks, mask),
    )
    contrast = conditional(model.active, model.ix, event.charges_pe, expectation)
    contrast -= conditional(model.active, model.ix, event.charges_pe, mean)
    gap = measured.score - fit.score - 2.0 * (measured.joint - fit.joint).max(initial=0.0)
    within = contrast.sum() - contrast.max(initial=0.0)
    return dict(gain=float(gap), module_gain=float(within), support=light)
