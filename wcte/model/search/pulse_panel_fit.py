"""Score fixed physical endpoints by separate detector module evidence."""

import numpy as np

from module_optics import conditional, shapes
from physical_pulse import components, profile
from predictive_score import log_probabilities
from gated_photon_time import log_density


def scores(event, model, state, *, clock=None, light=None):
    """Use the same observed first-time likelihood and calibrated charge."""
    gate = event.timing_selection["prompt_window_ns"]
    if gate is None:
        return None
    rate = shapes(model, state)
    if light is None:
        light = components(event, model, state, separate_knockon=True)
    if clock is None:
        fitted = profile(event, model, state, precomputed=light)
        if fitted is None:
            return None
        clock = fitted["clock_ns"]
    rates, centers = light
    center = np.median(event.times_ns)
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    timing = log_density(
        event.times_ns - center,
        *(np.asarray(gate) - center),
        centers + clock,
        rates,
        sigma,
        model.obs.offsets,
        model.obs.coefficients,
    )
    charge = log_probabilities(model.obs, rate, model.ix)[0]
    module = np.unique(model.active // 100, return_inverse=True)[1]
    n = int(module.max()) + 1
    joint = np.bincount(module, weights=charge)
    joint += np.bincount(module[model.ix], weights=timing, minlength=n)
    within = conditional(model.active, model.ix, event.charges_pe, rate)
    return joint, within, clock, np.bincount(module[model.ix], weights=timing, minlength=n)


def within_module_wavefront(event, current, proposed, weak, clocks):
    """Compare relative PMT illumination and first times in the same modules."""
    base, state = current
    trial, new = proposed
    a = scores(event, base, state, clock=clocks[0])
    b = scores(event, trial, new, clock=clocks[1])
    if a is None or b is None:
        return None
    # Profiling out module brightness loses information about total PE,
    # but the calibrated charge-conditioned first times remain available.
    conditional = b[1] - a[1] + b[3] - a[3]
    joint = b[0] - a[0]
    groups = np.unique(base.active // 100, return_inverse=True)[1][base.ix]
    exposure = np.bincount(groups, weights=np.minimum(event.charges_pe, 2.0) * weak, minlength=len(joint))
    exposed = conditional[exposure >= 2.0]
    dimension = (trial.dimension - base.dimension) * np.log(len(base.active))
    return dict(
        within_module_time=float(conditional.sum()),
        exposed_modules=int(len(exposed)),
        positive_modules=int(np.count_nonzero(exposed > 0.0)),
        exposed_gain_without_best=float(exposed.sum() - exposed.max(initial=0.0)),
        joint_gain_without_best=float(2.0 * (joint.sum() - joint.max(initial=0.0)) - dimension),
    )


def compare(event, current, proposed, reference_shares, trial_shares, clocks):
    """Check whether likelihood gain is distributed over separate modules."""
    base, state = current
    alternative, trial = proposed
    a = scores(event, base, state, clock=clocks[0])
    b = scores(event, alternative, trial, clock=clocks[1])
    if a is None or b is None:
        return None
    physical = b[0] - a[0]
    conditional = b[1] - a[1]
    modules = np.unique(base.active // 100, return_inverse=True)[1][base.ix]
    excess = np.maximum(trial_shares.sum(1) - reference_shares.sum(1), 0.0)
    charge = np.minimum(event.charges_pe, 2.0)
    assigned = np.bincount(modules, weights=charge * excess, minlength=len(physical))
    exposed = assigned >= 1.5
    if exposed.sum() < 3:
        return None
    measured = physical[exposed]
    # Keep the global improvement after dropping the single most favorable
    # module. This exposes alternatives that mainly fit one local pattern.
    return dict(
        joint_ll=float(physical.sum()),
        within_module=float(conditional.sum()),
        joint_without_best_module=float(physical.sum() - physical.max()),
        explained_modules=int(exposed.sum()),
        positive_modules=int((measured > 0.0).sum()),
        recovered_pe=float(assigned.sum()),
        relative_clock_ns=float(b[2]),
    )
