"""Profile below-margin conversion light with the existing two direct cores."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment

from conversion_arms import optimize as fit_conversion
from detached_conversion import Timing, direction_penalty, photon_flight, photon_origins
from detached_track_fit import ApertureCharts, profile as fit_clocks
from geometry import hardware_margin
from independent_arcs import Independent
from material import material
from module_optics import conditional
from module_supported_ring import lower_bound
from recoil_competitor import render
from track_coordinates import fit as fit_geometry
from wcte.counting import track_counting


@dataclass(frozen=True)
class Conversion:
    path: np.ndarray
    mean: float


def short_partner(model, parent, lepton, response, fixed):
    """Find a conversion partner sharing the fitted electron's birth."""
    water = material("electron")
    margin = water["margin_mm"] / 1000.0
    a = lepton[3:6]
    side = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    side /= np.linalg.norm(side)
    other = np.cross(a, side)
    best = []
    origins = photon_origins(parent)[1]
    for offset, ratio, deflection in product(
        (-0.07, 0.0, 0.09),
        (0.55, 0.85, 0.99),
        (np.zeros(3), 0.22 * side, -0.22 * side, 0.22 * other, -0.22 * other),
    ):
        # Nearby conversions can overlap one broad measured ring. The
        # two charged paths originate at the same point in this hypothesis.
        axis = a + deflection
        axis /= np.linalg.norm(axis)
        origin = lepton[:3] + offset * a
        partner = np.r_[origin, axis, np.zeros(3), ratio * margin, 1.0, np.log(np.deg2rad(1.7))]
        countable = lepton.copy()
        countable[:3] = origin
        pair = np.stack([countable, partner])
        if min(hardware_margin(p) for p in pair) <= 1e-5:
            continue
        main = render(model, countable, "electron", 4)[1]
        background = render(model, partner, "electron", 4)[1]
        ll = model.obs.score(fixed + response * (main + background), model.ix)[0]
        prior = direction_penalty(pair, [np.zeros((3, 2))] * 2, origins)[0]
        best.append(Conversion(pair, ll - prior))
    return max(best, key=lambda row: row.mean, default=None)


def candidate(event, model, compare, primary, parameters, port, old, budget, *, visible=False):
    """Alternate the shared conversion geometry and calibrated beam trajectory."""
    x = np.asarray(parameters)
    p, lepton = x[:24].reshape(2, 12)
    water = material("electron")
    margin, maximum = [water[key] / 1000.0 for key in ("margin_mm", "maximum_range_mm")]
    # Represent the direct electron and positron as stopping arcs. Two
    # resolved arms will also be profiled as a rival count when eligible.
    lepton = lepton.copy()
    lepton[9] = min(lepton[9], maximum * 0.999)
    lepton[10] = 1.0
    if lepton[9] <= margin:
        return None
    response, sky = np.exp(x[-2:])
    fixed = response * render(model, p, primary, 8)[1] + sky * model.bg
    mode = short_partner(model, p, lepton, response, fixed)
    if mode is None:
        return None
    origins = photon_origins(p)[1]

    def prior(paths, frames):
        return direction_penalty(paths, frames, origins)[:2]

    start = mode.path.copy()
    if visible:
        # Give a second countable conversion arm the same geometry freedom.
        # Start just above the margin so its support gradient is nonzero.
        start[1, 9] = max(start[1, 9], 1.22 * margin)
    pair = fit_conversion(
        model, start, fixed, response, 8, budget, prior=prior, support_sides=(1, 1 if visible else -1)
    )
    if pair is None:
        return None
    args = event.pmt_ids, event.charges_pe, model.bg, (primary, "electron", "electron")
    tracks = np.r_[p, pair.ravel(), x[-2:]]
    # Only the beam geometry moves here. Its nuisance response should see
    # both arms, while the common conversion origin stays tied exactly.
    search = Independent(*args, 8)
    chart = ApertureCharts(search, tracks, port)
    chart.bounds[10:30, 0] = chart.initial[10:30]
    chart.bounds[10:30, 1] = chart.initial[10:30]
    _, z = fit_geometry(search, tracks, budget, budget, chart=chart, full_likelihood=True, check_initial=True)
    if not np.isfinite(z).all() or search.hardware_margin(z) <= 1e-5 or z[:2] @ z[:2] > port.radius**2 + 1e-8:
        return None
    comparison = Timing.from_event(event)
    fine = Independent(*args, 32)
    low = Independent(*args, 16)
    error = abs(fine(z, True)["ll"] - low(z, True)["ll"])
    if error >= 0.01:
        return None
    # Constrain the arms to one causal time after the fitted beam endpoint.
    # The charged and neutral parts of that flight use their own velocities.
    delay, penalty, alignment = photon_flight(z[:12], primary, pair)
    timing = fit_clocks(event, fine, z, comparison, fixed_delays=np.repeat(delay, 2))
    if timing is None:
        return None
    mean, rates, clocks, shares, offsets, norm = timing
    direct = shares if visible else shares[:, :2]
    shares = np.column_stack([1.0 - direct.sum(1), direct])
    record = dict(
        materials=(primary, "electron", "electron"),
        parameters=np.r_[z[:-2], norm].tolist(),
        clock_ns=offsets.tolist(),
        conversion_delay_ns=float(delay),
        flight_alignment=float(alignment),
        direction_penalty=float(penalty),
        quadrature_ll_error=float(error),
    )
    dimension = 9 + 9 + 3 + 2 + 1
    measured = compare.measured(shares, record, mean, rates, clocks, dimension, (1, 4, 16))
    modules = 2.0 * (measured.joint - old.joint)
    gain = float(measured.score - old.score - modules.max(initial=0.0) - 2.0 * penalty)
    if visible:
        minimum = min(
            lower_bound(event.pmt_ids, row)["lower_statistical"] for row in compare.supports(measured).T
        )
        if minimum < track_counting.get().min_track_pmts or set(shares.argmax(1)) - {0} != {1, 2, 3}:
            return None
        record["lowest_core_support"] = minimum
    return gain, measured, mean, record


def coverage(compare, old, measured, mean, old_mean, *, predictive=False):
    """Verify countable cores and model the other light without counting it."""
    model, event = compare.model, compare.event
    support = compare.supports(measured)
    main, secondary, partner = support.T
    limits = [lower_bound(event.pmt_ids, p) for p in (main, secondary)]
    unresolved = lower_bound(event.pmt_ids, partner)
    previous = compare.supports(old)
    # Above-margin new charge must reproduce two distinct observed
    # ridges. All prompt light from the conversion may explain what
    # remains in the matched optical band without becoming a new ring.
    hit = mean[model.ix]
    total_secondary = measured.rates[:, 3:].sum(1) / hit
    features = np.column_stack([measured.fractions[:, 1], total_secondary])
    shares = old.fractions[:, 1:]
    common = (compare.q[:, None, None] * np.minimum(shares[:, :, None], features[:, None, :])).sum(0)
    i, j = linear_sum_assignment(-common)
    retained = common[i, j] / np.maximum(compare.q @ shares[:, i], 1e-15)
    (missing,) = set(range(previous.shape[1])) - set(i)
    novel = lower_bound(
        event.pmt_ids,
        np.maximum(previous[:, missing] - np.maximum(np.maximum(main, secondary), total_secondary), 0.0),
    )
    delta = conditional(model.active, model.ix, event.charges_pe, mean)
    delta -= conditional(model.active, model.ix, event.charges_pe, old_mean)
    within = float(delta.sum() - delta.max(initial=0.0))
    accepted = (
        min(x["lower_statistical"] for x in limits) >= track_counting.get().min_track_pmts
        and unresolved["pmt_support"] < track_counting.get().min_track_pmts
        and novel["pmt_support"] < track_counting.get().min_track_pmts
        and retained.min() >= 0.60
        and (predictive or within > 0.0)
        and set(measured.fractions.argmax(1)) - {0} == {1, 2}
    )
    return accepted, dict(
        retained=retained.tolist(),
        support=limits,
        partner=unresolved,
        missing=novel,
        within_module_gain=within,
    )
