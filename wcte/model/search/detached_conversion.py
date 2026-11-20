"""Test whether a second cone comes from two below-margin conversion arms."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

from balanced_search import feasible
from chain_sources import paths
from charge_starts import weighted_quantile
from conversion_arms import optimize as refine_pair
from conversion_light_budget import render as render_pair
from coupled_model import Model as Coupled
from four_hit_vertices import intersections
from gated_photon_time import evaluate
from geometry import hardware_margin
from material import material
from physical_pulse import components, flight_integral, wavefront
from residual_timing_vertices import refine as refine_vertices
from source import rule, trajectory
from track_coordinates import fit as refine_track
from wctering.likelihood.geometry import NORMALS, UPPER
from wctering.seeding.coherent_timing import C, N_GROUP


# The fitted directions describe mean light profiles. Their connection to
# the physical conversion momentum has angular uncertainty from scattering.
PHOTON_CONCENTRATION = 1.0 / np.deg2rad(15.0) ** 2


class Parent(Coupled):
    """Refit the counted path with the extra charge held in the background."""

    conversion = None

    def primary_light(self, x):
        parts, jac = super().primary_light(x)
        if self.conversion is not None:
            parts[0, 1] += self.conversion
        return parts, jac


@dataclass(frozen=True)
class Timing:
    time: np.ndarray
    low: float
    high: float
    sigma: np.ndarray

    @classmethod
    def from_event(cls, event):
        center = float(np.median(event.times_ns))
        low, high = np.asarray(event.timing_selection["prompt_window_ns"]) - center
        sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
        return cls(event.times_ns - center, low, high, sigma)

    def profile(self, model, signal, rates, arrivals, initial, signal_diffuse):
        """Fit shared response, diffuse charge and one event-relative clock."""
        weight = rates * np.minimum(model.q, 2.0)[:, None]
        clock = weighted_quantile((self.time[:, None] - arrivals).ravel(), weight.ravel(), [0.5])[0]
        bounds = [*model.bounds[-2:], (clock - 4.0, clock + 4.0)]
        initial = np.r_[np.clip(initial, *np.asarray(bounds[:2]).T), clock]
        sky = model.bg[model.ix]

        def objective(x):
            response, diffuse = np.exp(x[:2])
            mean = response * signal + diffuse * model.bg
            ll, slope = model.obs.score(mean, model.ix)
            intensities = response * rates
            full = np.column_stack([diffuse * sky + response * signal_diffuse, intensities])
            time_ll, d_rate, d_clock = evaluate(
                self.time,
                self.low,
                self.high,
                arrivals + x[2],
                full,
                self.sigma,
                model.obs.offsets,
                model.obs.coefficients,
            )
            grad = np.array(
                [
                    response * (signal @ slope) - np.sum(d_rate[:, 1:] * intensities),
                    diffuse * (model.bg @ slope - d_rate[:, 0] @ sky),
                    -d_clock.sum(),
                ]
            )
            grad[0] -= response * (d_rate[:, 0] @ signal_diffuse)
            return -(ll + time_ll.sum()) / model.total, grad / model.total

        fit = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=24, ftol=2e-9),
        )
        best = fit.x if np.isfinite(fit.fun) and fit.fun <= objective(initial)[0] else initial
        return -objective(best)[0] * model.total, best

    def likelihood(self, model, total, rates, arrivals):
        """Evaluate the common charge and first-arrival law by detector module."""
        from gated_photon_time import log_density
        from predictive_score import log_probabilities

        by_pmt = log_probabilities(model.obs, total, model.ix)[0]
        by_pmt[model.ix] += log_density(
            self.time,
            self.low,
            self.high,
            arrivals,
            rates,
            self.sigma,
            model.obs.offsets,
            model.obs.coefficients,
        )
        groups = np.unique(model.active // 100, return_inverse=True)[1]
        return np.bincount(groups, weights=by_pmt)


def residual_vertices(event, fraction, count=4):
    """Triangulate the residual pulse without using the trigger's origin time."""
    pos = event.pmt_positions_mm / 1000.0
    t = event.times_ns - np.median(event.times_ns)
    weight = np.minimum(event.charges_pe, 2.0) * fraction
    groups = np.unique(event.pmt_ids // 100, return_inverse=True)[1]
    module_charge = np.bincount(groups, weights=weight)
    balanced = weight / np.sqrt(np.maximum(module_charge[groups], 0.5))
    if (module_charge >= 1.0).sum() < 4:
        return []
    draws = np.random.default_rng(271828).choice(len(t), size=(1024, 4), p=balanced / balanced.sum())
    distinct = np.all(np.diff(np.sort(groups[draws], axis=1), axis=1) != 0, axis=1)
    travel = 1000.0 * N_GROUP / C
    points, birth = intersections(pos, t / travel, draws[distinct])
    if not len(points):
        return []
    flight = np.linalg.norm(points[:, None] - pos, axis=2) * travel
    residual = t - flight - travel * birth[:, None]
    gain = (np.exp(-0.5 * (residual / 1.4) ** 2) - 0.35 * np.exp(-0.5 * (residual / 3.6) ** 2)) @ weight
    modes = []
    for j in np.argsort(-gain, kind="stable"):
        if gain[j] < 7.0:
            break
        if all(np.linalg.norm(points[j] - p) > 0.16 for p in modes):
            modes.append(points[j])
        if len(modes) == 2 * count:
            break
    return refine_vertices(modes, pos, t, weight, maximum=count)


def photon_origins(parent):
    """Sample possible photon origins along the measured charged path."""
    u = np.linspace(0.0, 1.0, 9)
    return u, trajectory(parent, u)[0] / 1000.0


def direction_penalty(pair, charts, origins):
    """Use measured momentum closure as a smooth prior on conversion geometry."""
    water = material("electron")
    r = 1000.0 * pair[:, 9]
    energy = water["energy_at"](r)
    momentum = np.sqrt(energy * (energy + 2.0 * water["mass"]))
    slope = (energy + water["mass"]) / momentum * water["energy_at"].derivative()(r) * r
    total = momentum @ pair[:, 3:6]
    norm = np.linalg.norm(total)
    axis = total / max(norm, 1e-20)
    delta = pair[0, :3] - origins
    # A short flight has a poorly constrained direction at finite vertex
    # resolution. Include that uncertainty in both the loss and derivative.
    distance = np.sqrt(np.sum(delta * delta, axis=1) + 0.035**2)
    alignment = delta @ axis / distance
    selection = np.where(distance < 0.055, 1.0, alignment)
    j = int(np.argmax(selection))
    if distance[j] < 0.055:
        return 0.0, np.zeros(9), j, float(alignment[j])
    ray = delta[j] / distance[j]
    d_axis = (ray - axis * alignment[j]) / norm
    angular = np.concatenate([weight * (d_axis @ chart) for weight, chart in zip(momentum, charts)])
    derivative = np.r_[(axis - ray * alignment[j]) / distance[j], angular, slope * (pair[:, 3:6] @ d_axis)]
    return (
        float(PHOTON_CONCENTRATION * (1.0 - alignment[j])),
        -PHOTON_CONCENTRATION * derivative,
        j,
        float(alignment[j]),
    )


def photon_flight(parent, mass, pair):
    """Back-project the conversion to the measured charged path."""
    u, origins = photon_origins(parent)
    penalty, _, j, alignment = direction_penalty(pair, [np.zeros((3, 2))] * 2, origins)
    distance = 1000.0 * np.linalg.norm(pair[0, :3] - origins[j])
    water = material(mass)
    nodes, weights = rule(5)
    length = 1000.0 * parent[9] * parent[10] * u[j]
    remaining = 1000.0 * parent[9] - length * nodes
    gamma = 1.0 + water["energy_at"](remaining) / water["mass"]
    beta = np.sqrt(np.maximum(1.0 - gamma**-2, 1e-8))
    return float((length * (weights @ (1.0 / beta)) + distance) / C), penalty, alignment


def pair_starts(event, strong, weak, fraction):
    """Try residual vertices and the backward extension of the weak cone."""
    lepton = material("electron")
    edge = lepton["energy_at"](lepton["margin_mm"])
    source = weak[:3]
    direction = weak[3:6]
    points = residual_vertices(event, fraction)
    end = trajectory(strong, np.ones(1))[0][0] / 1000.0
    points.extend(source + v * direction for v in (-0.20, 0.0, 0.23, 0.50, 0.85))
    for vertex in points:
        if np.min(UPPER - NORMALS @ vertex) < 0.012:
            continue
        gamma_axis = vertex - end
        norm = np.linalg.norm(gamma_axis)
        axes = [direction]
        if norm > 0.08 and 0.55 < gamma_axis @ direction / norm < 0.999:
            axes.append(gamma_axis / norm)
        for axis, energies in product(axes, ((0.60, 0.28), (0.78, 0.40), (0.92, 0.76))):
            lengths = lepton["range_at"](edge * np.asarray(energies)) / 1000.0
            pair = np.array(
                [np.r_[vertex, axis, np.zeros(3), r, 1.0, np.log(np.deg2rad(2.0))] for r in lengths]
            )
            if min(hardware_margin(p) for p in pair) > 1e-5:
                yield pair


def parent_starts(incumbent, strong, mass, logs, event, background):
    """Keep the best single and recast the dominant arc at the same speed."""
    one, state = incumbent
    if len(one.particles) == 1:
        model = Parent(event.pmt_ids, event.charges_pe, background, one.particles, 4, 8)
        start = state["parameters"]
        if feasible(model, start) and model.hardware_margin(start) > 1e-5:
            yield model, start
    length = strong[9] * strong[10]
    old = material(mass)
    energy_per_mass = old["energy_at"](1000.0 * (strong[9] - 0.5 * length)) / old["mass"]
    for p in dict.fromkeys((mass, "proton", "pion")):
        m = material(p)
        energy = np.clip(m["mass"] * energy_per_mass, m["energy"][0] + 1.01, m["energy"][-1])
        ran = min(
            m["maximum_range_mm"] / 1000.0, max(1.001 * length, m["range_at"](energy) / 1000.0 + 0.5 * length)
        )
        if ran <= length:
            continue
        start = np.r_[strong[:9], ran, length / ran, strong[11], logs]
        model = Parent(event.pmt_ids, event.charges_pe, background, (p,), 4, 8)
        if feasible(model, start) and model.hardware_margin(start) > 1e-5:
            yield model, start


def score(
    event,
    time,
    model,
    primary,
    pair,
    initial,
    order,
    *,
    return_modules=False,
    split_arms=False,
    profile_delay=False,
):
    """Evaluate both conversion arms at a common physical birth time."""
    parent, mass = primary[:12], model.particles[0]
    parts, _ = model.primary_light(parent)
    sections, _ = render_pair(model, pair, order)
    extra = sections.sum(1)
    delay, penalty, alignment = photon_flight(parent, mass, pair)
    pos, normals = event.pmt_positions_mm, model.normals[model.ix]
    arrivals, parent_rates, _ = wavefront(pos, normals, parent, mass)
    arm_times = []
    margin = material("electron")["margin_mm"] / 1000.0
    for p in pair:
        if split_arms:
            arrival = wavefront(pos, normals, p, "electron", order=8)[0].T
        elif return_modules:
            center, nominal, _ = wavefront(pos, normals, p, "electron", order=8)
            weight = nominal.sum(0)
            arrival = np.divide(
                np.sum(center * nominal, axis=0), weight, out=center[0].copy(), where=weight > 1e-30
            )
        elif p[9] <= margin:
            arrival = wavefront(pos, normals, p, "electron")[0][1]
        else:
            from timed_charged_light import photon_time

            arrival = photon_time(event, normals, [p], "electron")
        arm_times.append(arrival + delay)
    if split_arms:
        total = np.vstack([parts[0], sections.reshape(-1, sections.shape[-1])])
    else:
        total = np.vstack([parts[0], extra])
    rates = total[:, model.ix].copy()
    diffuse = np.maximum(rates[1] - parent_rates[1], 0.0)
    rates[1] -= diffuse
    centers = np.column_stack([arrivals.T, *arm_times])
    ll, nuisance = time.profile(model, total.sum(0), rates.T, centers, initial, diffuse)
    if profile_delay:
        # Direction matching picks a photon origin along the charged path.
        # A curved track can leave several nearby origins plausible. Fit
        # the shared conversion clock over their causal flight envelope.
        from scipy.optimize import minimize_scalar

        u, points = photon_origins(parent)
        distance = 1000.0 * np.linalg.norm(pair[0, :3] - points, axis=1)
        p0 = 1000.0 * parent[9]
        primitive = flight_integral(mass)
        flight = (primitive(p0) - primitive(p0 * (1.0 - parent[10] * u)) + distance) / C
        bound = float(flight.min()) - delay, float(flight.max()) - delay
        if bound[1] - bound[0] > 0.02:

            def objective(offset):
                offsets = np.r_[0.0, 0.0, np.full(centers.shape[1] - 2, offset)]
                value, _ = time.profile(model, total.sum(0), rates.T, centers + offsets, initial, diffuse)
                return -value

            fitted = minimize_scalar(
                objective, bounds=bound, method="bounded", options={"xatol": 0.04, "maxiter": 9}
            )
            if np.isfinite(fitted.fun) and -fitted.fun > ll:
                delay += float(fitted.x)
                centers[:, 2:] += fitted.x
                ll, nuisance = time.profile(model, total.sum(0), rates.T, centers, initial, diffuse)
    ownership_light = np.vstack([parts[0], extra]) if split_arms else total
    result = ll, nuisance, ownership_light, delay, alignment, penalty
    if not return_modules:
        return result
    response, background = np.exp(nuisance[:2])
    mean = response * total.sum(0) + background * model.bg
    observed = np.column_stack([background * model.bg[model.ix] + response * diffuse, response * rates.T])
    modules = time.likelihood(model, mean, observed, centers + nuisance[2])
    return (*result, modules)


def contest(event, incumbent, trial, state, fractions, weak, budget):
    """Compare a counted split with a track and a physically timed conversion."""
    time = Timing.from_event(event)
    rates, arrivals = components(event, trial, state, separate_knockon=True)
    response, background = np.exp(state["parameters"][-2:])
    diffuse = np.maximum(rates[:, 0] - background * trial.bg[trial.ix], 0.0) / response
    trial_score, _ = time.profile(
        trial, trial.primary_cache[1], rates[:, 1:] / response, arrivals, state["parameters"][-2:], diffuse
    )
    geometry = paths(state["parameters"][: trial.ng])
    strong, secondary = geometry[1 - weak][0], geometry[weak][0]
    logs = state["parameters"][-2:]
    result = best_conversion(
        event,
        time,
        trial,
        incumbent,
        strong,
        secondary,
        trial.particles[1 - weak],
        logs,
        fractions[:, 1 + weak],
        budget,
    )
    if result is None:
        return None
    result.pop("fractions")
    # The pair adds one birth point, two directions and two ranges.
    result["penalized_gap"] = float(
        2.0 * (trial_score - result["joint_ll"] + result["direction_penalty"])
        + (result["dimension"] - trial.dimension) * np.log(len(trial.active))
    )
    result["trial_joint_ll"] = float(trial_score)
    return result


def best_conversion(event, time, prototype, incumbent, strong, weak, mass, logs, fraction, budget):
    """Rank cheap starts by charge, then fit the shared-clock alternative."""
    response, diffuse = np.exp(logs)
    pair_bank = [
        (p, render_pair(prototype, p, 4)[0].sum((0, 1))) for p in pair_starts(event, strong, weak, fraction)
    ]
    ranked = []
    for model, start in parent_starts(incumbent, strong, mass, logs, event, prototype.bg):
        model(start, True)
        _, light, _, _ = model.primary_cache
        _, origins = photon_origins(start)
        for pair, extra in pair_bank:
            penalty, _, _, _ = direction_penalty(pair, [np.zeros((3, 2))] * 2, origins)
            mean = response * (light + extra) + diffuse * model.bg
            ranked.append((model.obs.score(mean, model.ix)[0] - penalty, model, start, pair))
    chosen = []
    for _, model, start, pair in sorted(ranked, key=lambda r: r[0], reverse=True):
        if any(
            m.particles == model.particles and np.linalg.norm(p[0, :3] - pair[0, :3]) < 0.12
            for m, _, p in chosen
        ):
            continue
        chosen.append((model, start, pair))
        if len(chosen) == 3:
            break
    results = []
    for model, start, pair in chosen:
        fit = refine(event, time, model, start, pair, budget, fraction)
        if fit is not None:
            results.append(fit)
    return max(results, key=lambda x: x["joint_ll"] - x["direction_penalty"], default=None)


def refine(
    event,
    time,
    model,
    start,
    pair,
    budget,
    weak,
    *,
    maximum_range="margin",
    count_arms=False,
    support_sides=None,
    min_overlap=0.40,
    return_modules=False,
    split_arms=False,
    profile_delay=False,
):
    """Refit the parent and pair in turn, then profile the first-arrival law."""
    args = event.pmt_ids, event.charges_pe, model.bg, model.particles, 4
    parent = Parent(*args, 8)
    parent.conversion = render_pair(parent, pair, 8)[0].sum((0, 1))
    _, x = refine_track(parent, start, budget, budget, full_likelihood=True, check_initial=True)
    if not feasible(parent, x) or parent.hardware_margin(x) <= 1e-5:
        return None
    coarse = Parent(*args, 8)
    coarse(x, True)
    response, diffuse = np.exp(x[-2:])
    fixed = response * coarse.primary_cache[1] + diffuse * coarse.bg
    maximum = material("electron")["margin_mm"] / 1000.0 if maximum_range == "margin" else maximum_range
    _, origins = photon_origins(x)

    def prior(paths, charts):
        return direction_penalty(paths, charts, origins)[:2]

    improved = refine_pair(
        coarse,
        pair,
        fixed,
        response,
        8,
        budget,
        maximum_range=maximum,
        prior=prior,
        support_sides=support_sides,
    )
    if improved is None:
        return None
    parent = Parent(*args, 16)
    parent.conversion = render_pair(parent, improved, 16)[0].sum((0, 1))
    _, x = refine_track(parent, x, budget, budget, full_likelihood=True, check_initial=True)
    if not feasible(parent, x) or parent.hardware_margin(x) <= 1e-5:
        return None
    fine = Parent(*args, 32)
    fitted = score(
        event,
        time,
        fine,
        x,
        improved,
        x[-2:],
        32,
        return_modules=return_modules,
        split_arms=split_arms,
        profile_delay=profile_delay,
    )
    if fitted is None:
        return None
    ll, nuisance, parts, delay, alignment, penalty, *modules = fitted
    response, diffuse = np.exp(nuisance[:2])
    mean = response * parts.sum(0) + diffuse * fine.bg
    own = response * parts[:, fine.ix] / mean[fine.ix]
    weight = np.minimum(event.charges_pe, 2.0) * weak
    overlap = float(weight @ own[2:].sum(0) / max(weight.sum(), 1e-15))
    support = float(np.sum(1.0 - fine.obs.no_source(mean[fine.ix], own[0])))
    if overlap < min_overlap or support < 30.0 or not np.any(own[0] > 0.5):
        return None
    charge = float(fine.obs.score(mean, fine.ix)[0])
    # The 16-node parent already includes the conversion charge.
    coarse_main = parent.primary_light(x[:12])[0].sum((0, 1))
    error = abs(fine.obs.score(response * coarse_main + diffuse * fine.bg, fine.ix)[0] - charge)
    if error >= 0.03:
        return None
    arm_support = None
    if count_arms or support_sides is not None:
        core = render_pair(fine, improved, 32)[0][:, 0][:, fine.ix]
        arm_support = [
            float(np.sum(1.0 - fine.obs.no_source(mean[fine.ix], response * row / mean[fine.ix])))
            for row in core
        ]
        if count_arms and max(arm_support) < 30.0:
            return None
        if support_sides is not None and any(
            s * (v - 30.0) < 0.0 for s, v in zip(support_sides, arm_support)
        ):
            return None
    result = dict(
        joint_ll=float(ll),
        charge_ll=charge,
        pair=improved,
        parent=x,
        material=fine.particles[0],
        core_pmts=support,
        conversion_ownership=overlap,
        photon_delay_ns=delay,
        flight_alignment=alignment,
        direction_penalty=penalty,
        clock_ns=float(nuisance[2]),
        dimension=fine.dimension + 9 + int(profile_delay),
        quadrature_ll_error=float(error),
        arm_support=arm_support,
        fractions=np.column_stack([1.0 - own[0], own[0]]),
    )
    if modules:
        result["module_scores"] = modules[0]
    return result
