"""Compare a faint ring with delayed and angularly diffuse emission."""

import numpy as np
from scipy.optimize import minimize

from charge_starts import weighted_quantile
from gated_photon_time import evaluate


def intensity_fit(
    event,
    model,
    basis,
    initial,
    arrivals,
    delayed=0,
    *,
    amplitude_bounds=None,
    local_clock_ring=None,
    timing_components=None,
):
    """Fit measured PE and first times with a shared relative event clock.

    A physical source can contribute several arrival distributions. Their
    charge shapes sum to its basis row and all use that row's amplitude.
    """
    gate = event.timing_selection["prompt_window_ns"]
    if gate is None:
        return None
    center = np.median(event.times_ns)
    time = event.times_ns - center
    low, high = np.asarray(gate) - center
    if np.any(time < low - 1e-9) or np.any(time > high + 1e-9):
        return None
    q = np.minimum(event.charges_pe, 2.0)
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    ix = model.ix
    if timing_components is None:
        observed = basis[:, ix].T
        groups = np.arange(len(basis))
    else:
        groups, shapes = timing_components
        observed = shapes[:, ix].T
    weight = q[:, None] * observed[:, 1:] * initial[groups[1:]]
    if not weight.any():
        return None
    clock = weighted_quantile((time[:, None] - arrivals).ravel(), weight.ravel(), [0.5])[0]
    start = np.r_[np.log(np.maximum(initial, 1e-8)), clock]
    extent = [(-18.0, np.log(1e6))] * len(basis) + [(clock - 5.0, clock + 5.0)]
    if amplitude_bounds is not None:
        for i, bounds in amplitude_bounds.items():
            extent[i] = tuple(np.log(bounds))
            start[i] = np.clip(start[i], *extent[i])
    # Emission centroids need not be simultaneous. Give the compared
    # component the same small transport-time freedom in both models.
    if local_clock_ring is not None:
        clock_columns = np.atleast_1d(local_clock_ring)
        start = np.r_[start, 0.0]
        extent.append((-2.5, 2.5))
    n = len(basis)
    timing_arguments = sigma, model.obs.offsets, model.obs.coefficients

    def objective(x):
        amp = np.exp(x[:n])
        rates = observed * amp[groups]
        ll, residual = model.obs.score(amp @ basis, ix)
        means = arrivals + x[n]
        if local_clock_ring is not None:
            means[:, clock_columns] += x[n + 1]
        lt, rate_derivative, clock_derivative = evaluate(
            time, low, high, means, rates, *timing_arguments, delayed
        )
        timing_gradient = (rates * rate_derivative).sum(0)
        if timing_components is not None:
            timing_gradient = np.bincount(groups, timing_gradient, minlength=n)
        gradient = amp * (basis @ residual) - timing_gradient
        clock_gradient = -clock_derivative.sum()
        gradient = np.r_[gradient, clock_gradient]
        if local_clock_ring is not None:
            gradient = np.r_[gradient, -clock_derivative[:, clock_columns].sum()]
        value = -(ll + lt.sum()) / model.total
        return value, gradient / model.total

    fit = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=extent,
        options=dict(maxiter=28, ftol=2e-9, gtol=1e-6),
    )
    if not np.isfinite(fit.fun) or fit.fun > objective(start)[0]:
        return None
    return -float(fit.fun) * model.total, np.exp(fit.x[:n]), float(fit.x[n])
