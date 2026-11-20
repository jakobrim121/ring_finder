"""Test prompt electromagnetic light from charge exchange at a pion stop."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from balanced_search import feasible
from chain_comparison import MARGIN
from charge_starts import allowed_chord
from diffuse_timing import fit_mixture
from directional_shower import angular_density
from entrance_track import Aperture, EntranceChart, rank as entry_starts, wavefronts as entry_wavefronts
from geometry import hardware_margin
from material import material, stopping
from physical_pulse import flight_integral, wavefront
from recoil_competitor import render as track_light
from source import trajectory
from timed_charged_light import projected_aperture
from timed_optical import arrival
from track_coordinates import fit as optimize
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


PION_ZERO_MEV = 134.9768
ELECTRON_MEV = 0.51099895
INITIAL_MEV = 139.57039 + 938.27208816
NEUTRON_MEV = 939.56542052
PION_ENERGY_MEV = (INITIAL_MEV**2 + PION_ZERO_MEV**2 - NEUTRON_MEV**2) / (2.0 * INITIAL_MEV)
PION_MOMENTUM_MEV = np.sqrt(PION_ENERGY_MEV**2 - PION_ZERO_MEV**2)
CONVERSION_LENGTH_MM = 9.0 / 7.0 * 360.8


@lru_cache(maxsize=1)
def dark_range():
    """Approximate the distance and flight time below the optical threshold."""
    from scipy.integrate import quad

    water = material("pion")
    mass = water["mass"]
    threshold = water["energy"][0]

    def speed(kinetic):
        return C * np.sqrt(1.0 - (1.0 + kinetic / mass) ** -2)

    # The Bethe model is unreliable at very low energy. The last MeV
    # contributes little distance and is outside this transport integral.
    length = quad(lambda energy: 1.0 / stopping(energy, mass), 1.0, threshold)[0]
    time = quad(lambda energy: 1.0 / (speed(energy) * stopping(energy, mass)), 1.0, threshold)[0]
    return length, time


def water_stop(path):
    """Continue the slowing path through its optically dark final range."""
    end, direction, *_ = trajectory(path[:12], np.ones(1))
    extension, delay = dark_range()
    origin = end[0] + extension * direction[0]
    primitive = flight_integral("pion")
    clock = float((primitive(1000.0 * path[9]) - primitive(0.0)) / C) + delay
    return origin, clock


def frame(axis):
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def second_direction(first, azimuth, energy):
    """Conserve two-photon invariant mass and the charge-exchange energy."""
    opposite_energy = PION_ENERGY_MEV - energy
    cosine = 1.0 - PION_ZERO_MEV**2 / (2.0 * energy * opposite_energy)
    if not -1.0 <= cosine <= 1.0:
        return None
    return cosine * first + np.sqrt(1.0 - cosine**2) * azimuth


def decay_energy(total_energy, cosine):
    """Recover the positive energy pair from total energy and opening angle."""
    product = PION_ZERO_MEV**2 / (2.0 * max(1.0 - cosine, 1e-12))
    discriminant = total_energy**2 - 4.0 * product
    if discriminant < 0.0:
        return None
    return 0.5 * (total_energy + np.array([-1.0, 1.0]) * np.sqrt(discriminant))


@lru_cache(maxsize=32)
def electron_response(energy):
    """Integrate the core and tail of a lepton over its slowing range."""
    water = material("electron")
    energy = min(energy, water["energy"][-1])
    distance = float(water["range_at"](energy))
    margin = water["margin_mm"]
    grid = np.linspace(-1.0, 1.0, 1025)
    coefficients = np.zeros((2, len(grid)))
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(12)
    for j, (low, high) in enumerate(((min(margin, distance), distance), (0.0, min(margin, distance)))):
        if high <= low:
            continue
        remain = low + 0.5 * (nodes + 1.0) * (high - low)
        beta = np.sqrt(1.0 - (1.0 + water["energy_at"](remain) / ELECTRON_MEV) ** -2)
        opening = np.minimum(1.0 / (N_PHASE * beta), 1.0)
        variance = np.maximum(water["scatter_at"](remain) - water["scatter_at"](distance), 0.0)
        concentration = 1.0 / (variance + np.deg2rad(2.0) ** 2)
        strength = 0.5 * (high - low) * weights * (1.0 - opening**2)
        coefficients[j] = angular_density(grid[:, None], opening, concentration) @ strength
    return grid, coefficients


@lru_cache(maxsize=1)
def response_table():
    """Interpolate the smooth pair response in energy during local searches."""
    energy = np.linspace(6.0, 90.0, 169)
    responses = [electron_response(float(e)) for e in energy]
    return energy, responses[0][0], np.asarray([row[1] for row in responses])


@dataclass(frozen=True)
class Shower:
    axis: np.ndarray
    distance_mm: float
    energy_mev: float
    split: float
    light: np.ndarray
    arrival: np.ndarray


def photon_geometry(model, point, axis):
    """Reuse PMT polar angles and projected exposure across trial energies."""
    delta = model.pos - point
    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    direction = delta / length[:, None]
    incidence = projected_aperture(-np.sum(direction * model.normals, axis=1)[:, None])[:, 0]
    exposure = incidence / (length**2 + 10000.0)
    polar = direction @ axis
    return polar, exposure, length[model.ix]


def shower(model, origin, axis, distance, energy, split, *, projected=None, tabulated=False):
    """Render one conversion at a latent distance along the photon ray."""
    if projected is None:
        projected = photon_geometry(model, origin + distance * axis, axis)
    polar, exposure, length = projected
    parts = []
    for share in (split, 1.0 - split):
        kinetic = (energy - 2.0 * ELECTRON_MEV) * share
        if tabulated:
            energies, grid, table = response_table()
            index = np.clip(np.searchsorted(energies, kinetic) - 1, 0, len(energies) - 2)
            left, right = energies[index : index + 2]
            fraction = np.clip((kinetic - left) / (right - left), 0.0, 1.0)
            values = table[index] + fraction * (table[index + 1] - table[index])
        else:
            grid, values = electron_response(kinetic)
        parts.extend(np.interp(polar, grid, row) * exposure for row in values)
    arrival = (distance + N_GROUP * length) / C
    return Shower(axis, distance, energy, split, np.asarray(parts), arrival)


def distributed_shower(model, origin, arm, *, compton=False):
    """Render the first pair over its track length and allow radiation in water."""
    from electromagnetic_cascade import render_fast

    point = origin + arm.distance_mm * arm.axis
    chord = allowed_chord(point / 1000.0, arm.axis)
    if chord is None or chord[0] > 0.0 or chord[1] <= 0.0:
        return None
    split = 0.88 if compton else arm.split
    light, arrival = render_fast(
        model, point, arm.axis, arm.energy_mev, 1000.0 * chord[1], split=split, compton=compton
    )
    return Shower(
        arm.axis, arm.distance_mm, arm.energy_mev, arm.split, light[::-1], arrival + arm.distance_mm / C
    )


def refine_photons(
    model, origin, fixed, first, second, *, distributed=False, evaluations=90, tabulated=False
):
    """Refine both photon flights while preserving the pion decay kinematics."""
    from scipy.optimize import minimize

    tangent = np.column_stack(frame(first.axis))
    perpendicular = second.axis - (second.axis @ first.axis) * first.axis
    norm = np.linalg.norm(perpendicular)
    if norm < 1e-9:
        return None
    perpendicular /= norm
    azimuth = np.arctan2(perpendicular @ tangent[:, 1], perpendicular @ tangent[:, 0])
    fraction = (2.0 * first.energy_mev - PION_ENERGY_MEV) / PION_MOMENTUM_MEV
    start = np.array(
        [
            0.0,
            0.0,
            azimuth,
            np.arctanh(np.clip(fraction, -0.999, 0.999)),
            np.log(first.distance_mm),
            np.log(second.distance_mm),
        ]
    )
    base = model.obs.score(fixed, model.ix)[0]
    best = np.inf, None

    def evaluate(x):
        nonlocal best
        axis = first.axis + tangent @ x[:2]
        axis /= np.linalg.norm(axis)
        # Transport the old azimuth basis smoothly around the refined ray.
        a = tangent[:, 0] - (axis @ tangent[:, 0]) * axis
        a /= np.linalg.norm(a)
        b = np.cross(axis, a)
        phi = x[2]
        energy = 0.5 * (PION_ENERGY_MEV + PION_MOMENTUM_MEV * np.tanh(x[3]))
        other = second_direction(axis, np.cos(phi) * a + np.sin(phi) * b, energy)
        if other is None:
            return 1e8
        arms = []
        for direction, length, e, split in zip(
            (axis, other), np.exp(x[4:]), (energy, PION_ENERGY_MEV - energy), (first.split, second.split)
        ):
            chord = allowed_chord(origin / 1000.0, direction)
            if chord is None or chord[0] > 0.0 or length >= 995.0 * chord[1]:
                return 1e8
            candidate = shower(model, origin, direction, length, e, split, tabulated=tabulated)
            arms.append(candidate)
        signal = np.sum([arm.light.sum(0) for arm in arms], axis=0)
        penalty = sum(arm.distance_mm for arm in arms) / CONVERSION_LENGTH_MM
        loss = base - model.obs.score(fixed + model.response * signal, model.ix)[0] + penalty
        if loss < best[0]:
            best = loss, (*arms, penalty)
        return loss

    limits = [
        (-0.26, 0.26),
        (-0.26, 0.26),
        (azimuth - 0.7, azimuth + 0.7),
        (-2.8, 2.8),
        (np.log(45.0), np.log(2300.0)),
        (np.log(45.0), np.log(2300.0)),
    ]
    evaluate(start)
    start = np.clip(start, *np.asarray(limits).T)
    minimize(evaluate, start, method="Powell", bounds=limits, options=dict(maxfev=evaluations, ftol=1e-5))
    if best[1] is None:
        return None
    if distributed:
        first, second, penalty = best[1]
        first, second = (distributed_shower(model, origin, arm) for arm in (first, second))
        return (first, second, penalty) if first is not None and second is not None else None
    return best[1]


def project_pair(projected, energy, split):
    """Rank a point conversion with the summed response of its two arms."""
    polar, exposure, _ = projected
    left = float((energy - 2.0 * ELECTRON_MEV) * split)
    right = float((energy - 2.0 * ELECTRON_MEV) * (1.0 - split))
    grid, response = pair_angular(left, right)
    return np.interp(polar, grid, response) * exposure


@lru_cache(maxsize=64)
def pair_angular(left, right):
    """Cache the angular density summed over both charged trajectories."""
    grid, a = electron_response(left)
    _, b = electron_response(right)
    return grid, a.sum(0) + b.sum(0)


def charge_screen(model, fixed):
    """Use the local charge score only to choose which starts to evaluate."""
    _, derivative = model.obs.score(fixed, model.ix)
    inverse = 1.0 / np.maximum(fixed, 0.15)

    def score(light, distance):
        rate = model.response * light
        # Keep the local Poisson Fisher curvature. A saturating
        # correction prevents a bright, poorly fit trial from dominating
        # the expansion just because an incumbent rate was near zero.
        gain = -(rate @ derivative) - 0.5 * ((rate * rate) @ (inverse / (1.0 + 0.5 * rate * inverse)))
        return float(gain - distance / CONVERSION_LENGTH_MM)

    return score


def screen_first(model, remaining, rows, *, limit=12):
    """Compare every ray before rendering the exact charge shortlist."""
    trial_score = charge_screen(model, remaining)
    ranked = []
    for axis, distance, projection, energy, split, azimuths in rows:
        light = project_pair(projection, energy, split)
        ranked.append((trial_score(light, distance), axis, distance, projection, energy, split, azimuths))
    return sorted(ranked, key=lambda row: -row[0])[:limit]


def photon_starts(
    model,
    origin,
    geometry,
    remaining,
    event=None,
    *,
    reuse_geometry=False,
    paired_starts=False,
    charge_shortlist=False,
    initial_pairs=None,
    batch_rank=False,
    batch_pairs=False,
):
    """Search gamma directions through the measured lobes and their partners."""
    from residual_search import axis_grid

    points = np.asarray(geometry["vertices"])
    fitted_axes = np.asarray(geometry["axes"])
    lengths = np.asarray([v or 0.0 for v in geometry.get("lengths_mm", [None] * len(points))])
    if event is not None:
        # A gamma can convert upstream of its apparent optical center.
        # A clock-independent residual vertex provides a second estimate
        # when the measured cone is too broad to locate the conversion.
        from detached_conversion import residual_vertices

        residual = np.clip(1.0 - remaining[model.ix] / np.maximum(event.charges_pe, 0.25), 0.0, 1.0)
        vertices = np.asarray(residual_vertices(event, residual, count=2)).reshape(-1, 3)
        points = np.vstack([points, points - 0.5 * lengths[:, None] * fitted_axes, 1000.0 * vertices])
    lines = points - origin
    axis = lines / np.maximum(np.linalg.norm(lines, axis=1, keepdims=True), 1e-20)
    axes = np.vstack([axis, axis_grid(30)])
    imbalance = [-0.60, 0.0, 0.60] if event is None else [-0.95, -0.60, 0.0, 0.60, 0.95]
    energy_grid = 0.5 * (PION_ENERGY_MEV + PION_MOMENTUM_MEV * np.asarray(imbalance))
    model_score, _ = model.obs.score(remaining, model.ix)
    candidates = []
    shortlist_size = 5
    proposals = []
    for first in axes:
        if first @ first < 0.99:
            continue
        offsets = np.maximum(lines @ first, 0.0)
        side = lines - offsets[:, None] * first
        directed = np.argsort(np.linalg.norm(side, axis=1))[:1]
        unit, other = frame(first)
        azimuths = [unit, -unit, other, -other]
        tangent = side[directed[0]]
        if np.linalg.norm(tangent) > 1e-8:
            azimuths.append(tangent / np.linalg.norm(tangent))
        near = float(np.clip(offsets[directed[0]], 120.0, 950.0))
        stop = allowed_chord(origin / 1000.0, first)
        if stop is None or stop[0] > 0.0 or stop[1] < 0.075:
            continue
        distances = [near] if event is None else [near, 0.62 * near, 1.42 * near]
        distances = np.unique(np.minimum(distances, 995.0 * stop[1]))
        local = []
        for distance in distances:
            projection = photon_geometry(model, origin + distance * first, first) if reuse_geometry else None
            for energy, split in product(energy_grid, (0.50, 0.76)):
                if charge_shortlist or batch_rank:
                    proposals.append((first, distance, projection, energy, split, azimuths))
                    continue
                leading = shower(model, origin, first, distance, energy, split, projected=projection)
                prediction = remaining + model.response * leading.light.sum(0)
                ll = model.obs.score(prediction, model.ix)[0]
                # The first arm alone must explain some measured residual.
                local.append((ll - model_score - distance / CONVERSION_LENGTH_MM, leading, azimuths, split))
        # No candidate below the fifth from this ray can enter the
        # global top five. Release its large PMT arrays immediately.
        if reuse_geometry:
            candidates = sorted(candidates + local, key=lambda row: -row[0])[:shortlist_size]
        else:
            candidates.extend(local)
    if batch_rank and proposals:
        from charge_series_batch import shortlist

        projected = np.asarray(
            [remaining + model.response * project_pair(row[2], row[3], row[4]) for row in proposals]
        )
        penalty = np.asarray([row[1] / CONVERSION_LENGTH_MM for row in proposals])
        retained = shortlist(model, projected, penalty, shortlist_size)
        screened = [proposals[j] for j in sorted(retained)]
    elif charge_shortlist:
        screened = [r[1:] for r in screen_first(model, remaining, proposals)]
    else:
        screened = []
    if screened:
        for axis0, distance, projection, energy, split, azimuths in screened:
            leading = shower(model, origin, axis0, distance, energy, split, projected=projection)
            prediction = remaining + model.response * leading.light.sum(0)
            ll = model.obs.score(prediction, model.ix)[0]
            candidates.append((ll - model_score - distance / CONVERSION_LENGTH_MM, leading, azimuths, split))
    trials = []
    measured_pairs = []
    # A back-to-back pair can place the two conversions in very different
    # detector regions. Let two measured sites nominate both rays at once.
    if event is not None and paired_starts:
        for j, first in enumerate(axis):
            for k in range(j + 1, len(axis)):
                second = axis[k]
                if min(first @ first, second @ second) < 0.99:
                    continue
                energies = decay_energy(PION_ENERGY_MEV, first @ second)
                if energies is None:
                    continue
                distance = np.linalg.norm(lines[[j, k]], axis=1)
                chords = [allowed_chord(origin / 1000.0, ray) for ray in (first, second)]
                if any(chord is None or chord[0] > 0.0 or chord[1] < 0.075 for chord in chords):
                    continue
                distance = np.clip(distance, 75.0, [995.0 * chord[1] for chord in chords])
                penalty = distance.sum() / CONVERSION_LENGTH_MM
                for pair in (energies, energies[::-1]):
                    arms = [
                        shower(model, origin, ray, length, energy, 0.50)
                        for ray, length, energy in zip((first, second), distance, pair)
                    ]
                    mean = remaining + model.response * sum(arm.light.sum(0) for arm in arms)
                    score = model.obs.score(mean, model.ix)[0]
                    trials.append((score - penalty, *arms, penalty))
    for _, leading, azimuths, split in sorted(candidates, key=lambda row: -row[0])[:shortlist_size]:
        for azimuth in azimuths:
            second = second_direction(leading.axis, azimuth, leading.energy_mev)
            if second is None:
                continue
            end = allowed_chord(origin / 1000.0, second)
            if end is None or end[0] > 0.0 or end[1] <= 0.04:
                continue
            along = np.maximum(lines @ second, 0.0)
            error = np.linalg.norm(lines - along[:, None] * second, axis=1)
            project = float(along[np.argmin(error)])
            distances = [220.0, 550.0, max(75.0, project)]
            if event is not None:
                distances.extend([max(90.0, 0.65 * project), max(130.0, 1.35 * project)])
            for distance in np.unique(np.minimum(distances, 995.0 * end[1])):
                other = shower(model, origin, second, distance, PION_ENERGY_MEV - leading.energy_mev, split)
                combined = leading.light.sum(0) + other.light.sum(0)
                mean = remaining + model.response * combined
                penalty = (leading.distance_mm + distance) / CONVERSION_LENGTH_MM
                if batch_pairs:
                    measured_pairs.append((mean, leading, other, penalty))
                else:
                    ll = model.obs.score(mean, model.ix)[0]
                    trials.append((ll - penalty, leading, other, penalty))
    if measured_pairs:
        from charge_series_batch import shortlist

        # The full pair is used only by the best three starts and the
        # likelihood screen. Recompute their scores in the original order.
        values = np.array([row[0] for row in measured_pairs])
        penalties = np.array([row[-1] for row in measured_pairs])
        for j in sorted(shortlist(model, values, penalties, 3)):
            mean, leading, other, penalty = measured_pairs[j]
            trials.append((model.obs.score(mean, model.ix)[0] - penalty, leading, other, penalty))
    if initial_pairs is not None:
        # Save a detector-only screening statistic before expensive
        # angular refinement. It does not enter the physics likelihood.
        initial_pairs.append((model_score, max((r[0] for r in trials), default=-np.inf)))
    return sorted(trials, key=lambda row: -row[0])[:3]


def pion_paths(
    event,
    model,
    geometry,
    basis,
    amplitude,
    budget,
    *,
    timed_starts=False,
    single_geometry=False,
    charge_shortlist=False,
):
    """Fit the pion's entrance path while retaining the other observed light."""
    from detached_conversion import Parent

    port = Aperture()
    options = dict(masses=("pion",)) if charge_shortlist else {}
    entries = [
        (coarse, x)
        for _, coarse, x in entry_starts(event, model, geometry, amplitude, port, **options)
        if coarse.particles == ("pion",)
    ]
    if timed_starts:
        # The surveyed normal provides an independent primary start when
        # a brighter conversion dominates the unrestricted charge rank.
        # The normal is only an initialization. It can turn in the fit.
        normal = dict(vertices=[], axes=[], openings_deg=[])
        water = material("pion")
        for point, axis, opening, _ in entry_wavefronts(event, normal, port):
            beta = 1.0 / (N_PHASE * np.cos(opening))
            kinetic = water["mass"] * (1.0 / np.sqrt(1.0 - min(beta, 0.999) ** 2) - 1.0)
            ran = float(water["range_at"](min(kinetic, water["energy"][-1]))) / 1000.0
            if not np.isfinite(ran) or ran < 0.02:
                continue
            trial = Parent(event.pmt_ids, event.charges_pe, model.bg, ("pion",), 4, 4)
            x = np.r_[
                point,
                axis,
                np.zeros(3),
                ran,
                1.0,
                np.log(np.deg2rad(2.0)),
                np.log([model.response, amplitude[0]]),
            ]
            if feasible(trial, x) and hardware_margin(x[:12]) > 1e-5:
                entries.append((trial, x))
                break
    for coarse, x in entries:
        # Preserve the measured light outside the entrance cone while
        # finding its path, since a conversion can be the brightest lobe.
        coarse(x, True)
        core = coarse.primary_cache[2][0, model.ix]
        # Profile every distinct cone as a possible primary. The optical
        # search can place the real beam ring in a faint component.
        terms = amplitude[1:, None] * basis[1:]
        order = np.argsort(-np.minimum(terms[:, model.ix], model.response * core).sum(1))
        if single_geometry:
            order = order[:1]
        paths = []
        for keep in order:
            extra = terms.sum(0) - terms[keep]
            fit = Parent(event.pmt_ids, event.charges_pe, model.bg, ("pion",), 4, 8)
            fit.conversion = extra / model.response
            seed = x.copy()
            seed[10] = 1.0
            # A pion that stops traverses the entire visible range.
            # Remove the radial fraction degree of freedom in the chart.
            stop = StoppingChart(fit, seed, port)
            _, candidate = optimize(
                fit, seed, budget, budget, full_likelihood=True, chart=stop, check_initial=True
            )
            if (
                candidate[:2] @ candidate[:2] > port.radius**2 + 1e-9
                or not feasible(fit, candidate)
                or hardware_margin(candidate[:12]) <= 1e-5
            ):
                continue
            quality = fit(candidate, True)["ll"]
            if all(
                np.linalg.norm(candidate[:3] - old[:3]) > 0.01
                or abs(candidate[9] - old[9]) > 0.04
                or candidate[3:6] @ old[3:6] < 0.999
                for _, old in paths
            ):
                paths.append((quality, candidate))
        for _, candidate in sorted(paths, key=lambda row: -row[0])[:2]:
            yield candidate
        if timed_starts and isinstance(coarse, Parent):
            # Evaluate the initial timed line as a separate basin. Pure
            # charge optimization can move it onto a nearby gamma lobe.
            yield x


class StoppingChart(EntranceChart):
    """Tie emitting length to the remaining range."""

    def __init__(self, model, x, port):
        super().__init__(model, x, port)
        self.initial[8] = 0.0
        self.bounds[8] = 0.0, 0.0

    def full(self, y):
        mapped = y.copy()
        mapped[8] = mapped[7]
        x, jac, values, slopes = super().full(mapped)
        mapping = jac.copy()
        mapping[:, 7] += mapping[:, 8]
        mapping[:, 8] = 0.0
        slopes[:, 7] += slopes[:, 8]
        slopes[:, 8] = 0.0
        return x, mapping, values[2:], slopes[2:]


def fit(event, model, geometry, basis, amplitude, incumbent, budget):
    """Compare a stopped pion plus two gamma showers with the measured cones."""
    gate = event.timing_selection["prompt_window_ns"]
    if gate is None or len(basis) not in (3, 4):
        return (None, [])
    # A one-ring capture hypothesis competes with the two-ring interpretation.
    if len(basis) != 3:
        return (None, [])
    if geometry.get("emission_model") == "full_water_line":
        return (None, [])
    paths = list(
        pion_paths(
            event,
            model,
            geometry,
            basis,
            amplitude,
            budget,
            timed_starts=True,
            single_geometry=False,
            charge_shortlist=True,
        )
    )
    if not paths:
        return (None, [])
    control = fit_mixture(event, model, basis, amplitude, arrival(event, geometry), return_detail=True)
    if control is None:
        return (None, [])
    old_ll, a, old_time = control
    screen = True
    reference_charge = model.obs.score(a @ basis, model.ix)[0] if screen else None
    initial_pairs = [] if screen else None
    compare = None
    from detached_conversion import Timing

    compare = Timing.from_event(event)
    old_mean = a @ basis
    old_modules = compare.likelihood(model, old_mean, (a[:, None] * basis[:, model.ix]).T, old_time)
    q = np.minimum(event.charges_pe, 2.0)
    reports = []
    options = []
    for path in paths:
        core, total, _ = track_light(model, path[:12], "pion", 16)
        _, finer, _ = track_light(model, path[:12], "pion", 32)
        if np.sum(np.abs(total - finer)) * model.response > 0.02:
            continue
        if model.response * core.sum() < 30.0:
            continue
        origin, clock = water_stop(path)
        flight = wavefront(event.pmt_positions_mm, model.normals[model.ix], path[:12], "pion", order=8)[0]
        initial = a[0] * model.bg + model.response * finer
        seeds = photon_starts(
            model,
            origin,
            geometry,
            initial,
            event,
            reuse_geometry=True,
            paired_starts=False,
            charge_shortlist=False,
            initial_pairs=initial_pairs,
            batch_rank=True,
            batch_pairs=False,
        )
        for score, first, second, penalty in seeds:
            options.append((score, core, finer, flight, clock, origin, first, second, penalty))
    # Keep the best starts and any separated stopping locations for the timed fit.
    finalists = sorted(options, key=lambda r: -r[0])[:3]
    for candidate in sorted(options, key=lambda r: -r[0]):
        if all((np.linalg.norm(candidate[5] - old[5]) > 80.0 for old in finalists)):
            finalists.append(candidate)
    if initial_pairs and any((np.isfinite(score) for _, score in initial_pairs)):
        seed_gain = float(max((score for _, score in initial_pairs)) - reference_charge)
        reports.append(
            dict(
                family="capture_seed_evidence",
                count=len(basis) - 1,
                supported=False,
                best_gain=seed_gain,
                delta_per_path=[
                    [float(base - reference_charge), float(score - reference_charge)]
                    for base, score in initial_pairs
                    if np.isfinite(score)
                ],
            )
        )
        if seed_gain < -16.0 * np.sqrt(len(model.ix)):
            return (None, reports)
    best = None
    for _, core, primary, flight, clock, origin, first, second, penalty in finalists:
        rays = refine_photons(
            model,
            origin,
            a[0] * model.bg + model.response * primary,
            first,
            second,
            distributed=True,
            evaluations=90,
            tabulated=False,
        )
        if rays is not None:
            first, second, penalty = rays
        if rays is None:
            first = distributed_shower(model, origin, first)
            second = distributed_shower(model, origin, second)
            if first is None or second is None:
                continue
        diffuse = np.maximum(primary - core, 0.0)
        radiation = [arm.light.sum(0) for arm in (first, second)]
        if any((row.sum() <= 1e-16 for row in [core, *radiation])):
            continue
        signal = np.asarray([model.bg, primary, *radiation])
        skies = np.asarray([model.bg, core, diffuse, *radiation])
        groups = np.array([0, 1, 1, 2, 3])
        ray = np.column_stack([flight.T, first.arrival + clock, second.arrival + clock])
        visible = signal.sum(1) * np.r_[a[0], np.repeat(model.response, 3)] > np.exp(-18.0)
        # Keep the background and primary while omitting fully dark conversion arms.
        visible[:2] = True
        timing = visible[groups] & (skies[:, model.ix].sum(1) > 0.0)
        timing[0] = True
        groups = np.cumsum(visible)[groups[timing]] - 1
        skies = skies[timing]
        ray = ray[:, timing[1:]]
        signal = signal[visible]
        norm = signal.sum(1)
        initial = norm * np.r_[a[0], np.repeat(model.response, 3)][visible]
        kwargs = dict(
            return_detail=True,
            clock_groups=np.zeros(ray.shape[1], dtype=int),
            timing_components=(groups, skies / norm[groups, None]),
        )
        bounds = {1: (0.75 * initial[1], 1.25 * initial[1])}
        bounds.update({j: (0.4 * initial[j], 2.5 * initial[j]) for j in range(2, len(initial))})
        spectrum = signal / norm[:, None]
        fit = fit_mixture(event, model, spectrum, initial, ray, amplitude_bounds=bounds, **kwargs)
        if fit is None:
            continue
        ll, intensity, fitted_time = fit
        full_means = intensity @ spectrum
        means = full_means[model.ix]
        share = intensity[1] / norm[1] * core[model.ix] / means
        scale = np.zeros(len(visible))
        scale[visible] = intensity / norm
        support = [
            float(
                np.sum(1.0 - model.obs.no_source(means, np.clip(scale[j] * arm[model.ix] / means, 0.0, 1.0)))
            )
            for j, source in enumerate((first, second), 2)
            for arm in source.light[::2]
        ]
        old_signal = incumbent[:, 1:]
        retained = q @ np.minimum(old_signal, share[:, None]) / np.maximum(q @ old_signal, 1e-20)
        main = int(np.argmax(retained))
        record = dict(
            family="stopped_pion_photons",
            count=1,
            gain=float(2.0 * (ll - old_ll - penalty)),
            supported=False,
            primary_retention=float(retained[main]),
            conversion_arm_pmts=support,
            background_photons=intensity[2:],
            photon_distances_mm=[first.distance_mm, second.distance_mm],
            photon_energies_mev=[first.energy_mev, second.energy_mev],
            interaction_mm=origin.tolist(),
            photon_axes=[first.axis.tolist(), second.axis.tolist()],
        )
        reports.append(record)
        if compare is None or record["gain"] <= 0.0 or retained[main] < 0.65:
            continue
        # A resolved conversion arm must retain its counted light.
        if max(support, default=0.0) >= 30.0:
            continue
        primary_support = float(np.sum(1.0 - model.obs.no_source(means, np.clip(share, 0.0, 1.0))))
        if primary_support < 30.0 or not np.any(share > 0.5):
            continue
        rates = (skies[:, model.ix] * (intensity / norm)[groups, None]).T
        modules = compare.likelihood(model, full_means, rates, fitted_time)
        delta = modules - old_modules
        gain = float(2.0 * (delta.sum() - delta.max(initial=0.0) - penalty))
        record.update(robust_gain=gain, primary_support=primary_support)
        if gain > MARGIN and (best is None or gain > best[0]):
            best = gain, np.column_stack([1.0 - share, share]), record
    if best is None:
        return (None, reports)
    from gated_optical_fit import fit as optical_fit

    rival = optical_fit(event, geometry, model.bg, budget, strength=1.0)
    if rival is None:
        return (None, reports)
    gain, fractions, record = best
    advantage = min(gain, record["gain"] + 2.0 * (old_ll - rival[2]["joint_score"]))
    record["refitted_optical_gain"] = advantage
    if advantage <= MARGIN:
        return (None, reports)
    record.update(gain=advantage, supported=True)
    return ((advantage, fractions, record), reports)
