"""Fit compact sources of diffuse light using charge and relative time."""

import numpy as np
from scipy.optimize import minimize

from charge_starts import weighted_quantile
from gated_photon_time import evaluate
from geometry import hardware_margin
from residual_axes import timed_residual_vertices
from timed_charged_light import projected_aperture
from timed_optical import arrival
from wctering.seeding.coherent_timing import C, N_GROUP
from wcte.counting import track_counting


def isotropic_shape(model, point):
    """An unresolved scattering region has no fitted Cherenkov axis."""
    delta = model.pos - 1000.0 * point
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
    incidence = -np.sum(model.normals * delta, axis=1) / distance
    aperture = projected_aperture(incidence[:, None])[:, 0]
    shape = np.maximum(aperture, 1e-10) / (distance**2 + 10000.0)
    return shape / shape.sum(), N_GROUP * distance[model.ix] / C


def rank_sources(event, model, geometry, shape, amplitude, limit):
    """Keep early diffuse clusters supported on several detector modules."""
    expected = amplitude @ shape
    _, residual = model.obs.score(expected, model.ix)
    charge = np.minimum(event.charges_pe, 2.0)
    time = event.times_ns - np.median(event.times_ns)
    positions = [np.asarray(p) / 1000.0 for p in geometry["vertices"]]
    positions.extend(timed_residual_vertices(event, expected[model.ix], limit=4))
    records = []
    for p in positions:
        if any(np.linalg.norm(p - q) < 0.13 for _, q, *_ in records):
            continue
        # A zero-length straight segment checks water and hardware.
        probe = np.r_[p, 0.0, 0.0, 1.0, np.zeros(3), 0.05, 0.0, -3.0]
        if hardware_margin(probe) <= 1e-5:
            continue
        light, travel = isotropic_shape(model, p)
        strength = -float(light @ residual)
        if strength <= 0.0:
            continue
        info = float(light**2 @ (1.0 / np.maximum(expected, 0.1)))
        w = charge * light[model.ix] / np.maximum(expected[model.ix], 0.3)
        clock = weighted_quantile(time - travel, w, [0.5])[0]
        prompt = np.exp(-0.5 * ((time - clock - travel) / 1.8) ** 2)
        if w @ prompt < 0.30 * w.sum():
            continue
        # Include the unhit receivers through the charge likelihood.
        initial = np.clip(strength / max(info, 1e-20), 4.0, 0.35 * model.total)
        improvement = model.obs.score(expected + initial * light, model.ix)[0]
        records.append((improvement, p, light, travel, initial))
    return sorted(records, key=lambda row: row[0], reverse=True)[:limit]


def fit_mixture(
    event,
    model,
    basis,
    amplitude,
    times,
    *,
    diffuse=0,
    soft_mask=0,
    amplitude_bounds=None,
    return_detail=False,
    clock_groups=None,
    timing_components=None,
    core_constraints=(),
    variance=None,
    iterations=32,
    relative_bounds=None,
    clock_bounds=None,
):
    """Profile light yields, a shared clock, and local production delays."""
    clock_origin = np.median(event.times_ns)
    relative = event.times_ns - clock_origin
    gate = np.asarray(event.timing_selection["prompt_window_ns"]) - clock_origin
    if np.any(relative < gate[0] - 1e-9) or np.any(relative > gate[1] + 1e-9):
        return None
    if timing_components is None:
        rate_groups, sources = np.arange(len(basis)), basis
    else:
        rate_groups, sources = timing_components
    weights = (
        np.minimum(event.charges_pe, 2.0)[:, None] * sources[1:, model.ix].T * amplitude[rate_groups[1:]]
    )
    if np.any(weights.sum(0) <= 0.0):
        return None
    offsets = np.array(
        [weighted_quantile(r, w, [0.5])[0] for r, w in zip((relative[:, None] - times).T, weights.T)]
    )
    clock = float(np.median(offsets))
    n = len(basis)
    # Core and slowing tail of the same trajectory can share its offset.
    if clock_groups is None:
        groups = np.arange(len(offsets))
        count, start = len(offsets), offsets[1:]
    else:
        groups = np.asarray(clock_groups, dtype=int)
        count = int(groups.max()) + 1
        start = np.array([np.median(offsets[groups == j]) for j in range(1, count)])
    # Some physical rivals come with a shared interaction clock from the
    # fitted parent. Other callers retain the original weighted estimate.
    if clock_bounds is None:
        clock_bounds = clock - 5.0, clock + 5.0
    else:
        clock = float(np.clip(clock, *clock_bounds))
    if relative_bounds is None:
        relative_bounds = [(-4.0, 4.0)] * (count - 1)
        initial = np.clip(start - clock, -4.0, 4.0)
    else:
        relative_bounds = list(relative_bounds)
        if len(relative_bounds) != count - 1:
            raise ValueError("Provide one relative time bound per independent delay")
        initial = np.clip(start - clock, *np.asarray(relative_bounds).reshape(-1, 2).T)
    x0 = np.r_[np.log(np.maximum(amplitude, 0.01)), clock, initial]
    bounds = [(-18.0, np.log(1e6))] * n + [clock_bounds, *relative_bounds]
    if amplitude_bounds is not None:
        for column, (lower, upper) in amplitude_bounds.items():
            bounds[column] = (np.log(max(lower, np.exp(-18.0))), np.log(upper))
            x0[column] = np.clip(x0[column], *bounds[column])
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    hit_shapes = sources[:, model.ix].T

    def objective(x):
        amps = np.exp(x[:n])
        rates = hit_shapes * amps[rate_groups]
        ll, residual = model.obs.score(amps @ basis, model.ix)
        arrivals = times + x[n] + np.r_[0.0, x[n + 1 :]][groups]
        lt, d_rate, d_clock = evaluate(
            relative,
            *gate,
            arrivals,
            rates,
            sigma,
            model.obs.offsets,
            model.obs.coefficients,
            diffuse,
            soft_mask,
            variance=variance,
        )
        timing_gradient = (rates * d_rate).sum(0)
        if timing_components is not None:
            timing_gradient = np.bincount(rate_groups, weights=timing_gradient, minlength=n)
        gradient = amps * (basis @ residual) - timing_gradient
        if clock_groups is None:
            other = d_clock[:, 1:].sum(0)
        else:
            other = np.bincount(groups, weights=d_clock.sum(0), minlength=count)[1:]
        g = np.r_[gradient, -d_clock.sum(), -other] / model.total
        return -(ll + lt.sum()) / model.total, g

    fit = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=iterations, ftol=1e-9, gtol=1e-6),
    )
    if not np.isfinite(fit.fun) or fit.fun > objective(x0)[0]:
        return None
    if core_constraints:
        from visibility_profile import support_score

        # The stopping core is a subset of one amplitude. Profile a
        # boundary optimum when a free fit moves to the other count.
        components = basis[:, model.ix]
        cores = [(j, row[model.ix], sense) for j, row, sense in core_constraints]

        def support(x):
            terms = np.exp(x[:n, None]) * components
            total = terms.sum(0)
            margins = np.empty(len(cores))
            gradient = np.zeros((len(cores), len(x)))
            for k, (source, core, sense) in enumerate(cores):
                signal = np.exp(x[source]) * core
                hits, full, other = support_score(
                    total, total - signal, model.obs.offsets, model.obs.coefficients
                )
                gradient[k, :n] = terms @ (full + other)
                gradient[k, source] -= signal @ other
                gradient[k] *= sense / 100.0
                margins[k] = (sense * (hits - 30.0) - 0.005) / 100.0
            return margins, gradient

        if support(fit.x)[0].min() < 0.0:
            fit = minimize(
                objective,
                fit.x,
                jac=True,
                method="SLSQP",
                bounds=bounds,
                constraints=[dict(type="ineq", fun=lambda x: support(x)[0], jac=lambda x: support(x)[1])],
                options=dict(maxiter=24, ftol=2e-9),
            )
            if not np.isfinite(fit.fun) or support(fit.x)[0].min() < -1e-6:
                return None
    ll = -float(fit.fun) * model.total
    a = np.exp(fit.x[:n])
    if not return_detail:
        return ll, a
    arrivals = times + fit.x[n] + np.r_[0.0, fit.x[n + 1 :]][groups]
    return ll, a, arrivals


def fit(event, model, geometry, shapes, amplitudes, incumbent):
    """Preserve ring multiplicity while allowing timed diffuse charge."""
    count = len(shapes) - 1
    if event.timing_selection["prompt_window_ns"] is None:
        return
    centers = rank_sources(event, model, geometry, shapes, amplitudes, 2)
    if not centers:
        return
    direct = arrival(event, geometry)
    control = fit_mixture(event, model, shapes, amplitudes, direct)
    if control is None:
        return
    q = np.minimum(event.charges_pe, 2.0)
    labels = incumbent.argmax(1)
    for _, point, light, travel, photons in centers:
        basis = np.vstack([shapes, light])
        trial = fit_mixture(
            event,
            model,
            basis,
            np.r_[amplitudes, photons],
            np.column_stack([direct, travel]),
            diffuse=count + 1,
        )
        if trial is None:
            continue
        ll, amp = trial
        # Three vertex coordinates, one intensity, one relative clock.
        gain = 2.0 * (ll - control[0]) - 5.0 * np.log(len(model.active))
        if gain <= 0.0:
            continue
        mean = (amp @ basis)[model.ix]
        portions = (amp[:, None] * basis[:, model.ix] / mean).T
        core = portions[:, 1 : count + 1]
        fractions = np.column_stack([1.0 - core.sum(1), core])
        hard = fractions.argmax(1)
        if set(hard) - {0} != set(range(1, count + 1)):
            continue
        support = [float(np.sum(1.0 - model.obs.no_source(mean, f))) for f in core.T]
        retain = [
            float(q[(hard == j) & (labels == j)].sum() / q[labels == j].sum()) for j in range(1, count + 1)
        ]
        if min(support) < track_counting.get().min_track_pmts or min(retain) < 0.65:
            continue
        yield (
            gain,
            fractions,
            dict(
                diffuse_center_m=point.tolist(),
                diffuse_photons=float(amp[-1]),
                joint_ll=float(ll),
                reference_joint_ll=float(control[0]),
                retained_charge=retain,
                observed_support=support,
            ),
        )
