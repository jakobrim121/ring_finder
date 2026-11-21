"""Fit the light from two sections of one charged track through the window."""

import numpy as np
from scipy.special import softmax

from analog_optical import endpoint
from aperture_pair_review import track_fit
from balanced_search import feasible
from charge_starts import recoil_start
from coupled_model import Model as Coupled
from entrance_track import Aperture
from gated_optical_fit import PromptOptics
from low_angle_continuation import verified
from material import material
from wctering.likelihood.charge import direction
from wctering.seeding.coherent_timing import C, N_PHASE
from wcte.counting import track_counting


def tangent_frame(axis):
    tangent = np.cross(axis, np.eye(3)[np.argmin(abs(axis))])
    tangent /= np.linalg.norm(tangent)
    return np.column_stack([tangent, np.cross(axis, tangent)])


class ThroughWindow:
    """Keep both endpoints of the incoming section on the charged trajectory."""

    def __init__(self, optics, port):
        self.optics = optics
        self.port = port
        self.total = optics.total
        self.water_upper = np.r_[optics.water_upper, 0.0, 0.0, 0.0, 0.0]
        path = optics.x[:18].reshape(2, 9)
        self.axes = path[:, 3:6].copy()
        self.frames = [tangent_frame(a) for a in self.axes]
        entrance = path[0, :2] - 0.5 * path[0, 6] * path[0, 3:5]
        self.x = np.r_[entrance, np.zeros(4), path[:, 6], path[:, 7], path[:, 8], optics.x[18:-1]]
        self.bounds = [
            *[(-port.radius, port.radius)] * 2,
            *[(-0.65, 0.65)] * 4,
            *[(0.002, 4.8)] * 2,
            optics.bounds[7],
            optics.bounds[16],
            optics.bounds[8],
            optics.bounds[17],
            *optics.bounds[18:-1],
        ]
        self.constraints = [
            dict(
                type="ineq",
                fun=lambda x: self.water_upper - self.water_values(x),
                jac=lambda x: -self.water_jacobian(x),
            )
        ]
        self.last = None

    def mapped(self, x):
        if self.last is not None and np.array_equal(x, self.last[0]):
            return self.last[1:]
        z = np.r_[np.zeros(18), x[12:], 0.0]
        jac = np.zeros((len(z), len(x)))
        jac[18:-1, 12:] = np.eye(4)
        point = np.r_[x[:2], self.port.z]
        downstream = np.zeros((3, len(x)))
        downstream[:2, :2] = np.eye(2)
        for j, (pole, frame) in enumerate(zip(self.axes, self.frames)):
            block = slice(2 + 2 * j, 4 + 2 * j)
            axis, slope = direction(pole + frame @ x[block])
            derivative = slope @ frame
            length, opening = x[6 + j], x[8 + j]
            i = 9 * j
            z[i : i + 9] = np.r_[point + 0.5 * length * axis, axis, length, opening, x[10 + j]]
            jac[i : i + 3] = downstream
            jac[i : i + 3, block] += 0.5 * length * derivative
            jac[i : i + 3, 6 + j] += 0.5 * axis
            jac[i + 3 : i + 6, block] = derivative
            jac[i + 6, 6 + j] = jac[i + 7, 8 + j] = jac[i + 8, 10 + j] = 1.0
            point += length * axis
            downstream[:, block] += length * derivative
            downstream[:, 6 + j] += axis
            # A calibrated clock at the incoming midpoint determines both
            # arrivals. The particle travels half of each interval between
            # midpoints, with beta inferred from the opening of each cone.
            flight = 500.0 * N_PHASE / C
            z[-1] += flight * length * np.cos(opening)
            jac[-1, 6 + j] = flight * np.cos(opening)
            jac[-1, 8 + j] = -flight * length * np.sin(opening)
        self.last = x.copy(), z, jac
        return z, jac

    def water_values(self, x):
        z, _ = self.mapped(x)
        # The path enters forwards, loses energy, and turns far enough to
        # count as two sections under the existing lineage definition.
        return np.r_[
            self.optics.water_values(z),
            (x[:2] @ x[:2] - self.port.radius**2) / self.port.radius,
            -z[5],
            z[16] - z[7],
            z[3:6] @ z[12:15] - track_counting.get().merge_cosine,
        ]

    def water_jacobian(self, x):
        z, jac = self.mapped(x)
        disc = np.r_[2.0 * x[:2] / self.port.radius, np.zeros(len(x) - 2)]
        turn = z[12:15] @ jac[3:6] + z[3:6] @ jac[12:15]
        return np.vstack([self.optics.water_jacobian(z) @ jac, disc, -jac[5], jac[16] - jac[7], turn])

    def axis_norms(self, x):
        return self.optics.axis_norms(self.mapped(x)[0])

    def evaluate(self, x):
        z, jac = self.mapped(x)
        ll, slope, posterior, mean, design = self.optics.evaluate(z)
        return ll, slope @ jac, posterior, mean, design @ jac


def fit(event, geometry, background, budget):
    """Refine a window seed and verify the joint light calculation."""
    try:
        optics = PromptOptics(event, geometry, background, 1.0)
    except (AssertionError, ValueError):
        return None
    chart = ThroughWindow(optics, Aperture())
    x, calls = endpoint(chart, budget)
    if x is None:
        return None
    z, _ = chart.mapped(x)
    state = verified(optics, z)
    if state is None or set(state[2].argmax(0)) - {0} != {1, 2}:
        return None
    support = [np.sum(1.0 - optics.obs.no_source(state[3][optics.ix], w)) for w in state[2][1:]]
    if min(support) < track_counting.get().min_track_pmts:
        return None
    paths = z[:18].reshape(2, 9)
    angle = np.degrees(np.arccos(np.clip(paths[0, 3:6] @ paths[1, 3:6], -1.0, 1.0)))
    if angle <= track_counting.get().track_merge_angle_deg:
        return None
    photons = softmax(np.r_[0.0, z[18:20]]) * np.exp(z[20])
    return (
        paths,
        state[2].T,
        dict(photons=photons, offset_ns=float(z[-1]), angle_deg=float(angle), evaluations=calls),
    )


def transports(
    event, model, paths, photons, budget, *, limit=1, module_weight=0.0, mixed=False, value_only=False
):
    """Ask whether both cone intervals fit one charged particle's transport."""
    port = Aperture()
    entrance = paths[0, :3] - 0.5 * paths[0, 6] * paths[0, 3:6]
    norms = np.log([model.response, max(photons[0], 0.01)])
    ranked = []

    def section(p, species):
        water = material(species)
        # Infer a first range from each fitted mean opening, then check
        # the loss through the whole interval and the nuclear recoil.
        beta = min(1.0 / (N_PHASE * np.cos(p[7])), 0.999)
        ke = water["mass"] * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
        ke = np.clip(ke, water["energy"][0] + 1.0, water["energy"][-1])
        remaining = water["range_at"](ke) / 1000.0
        ran = max(remaining + 0.5 * p[6], 1.001 * p[6])
        width = np.clip(p[8], np.log(np.deg2rad(0.5)), np.log(np.deg2rad(2.0)))
        return np.r_[p[3:6], np.zeros(3), ran, p[6] / ran, width]

    layouts = [(p, p) for p in ("pion", "proton", "muon")]
    if mixed:
        layouts.extend([("proton", "pion"), ("pion", "proton")])
    for masses in layouts:
        legs = [section(p, m) for p, m in zip(paths, masses)]
        scout = Coupled(event.pmt_ids, event.charges_pe, model.bg, masses, 4, 4)
        x = recoil_start(scout, np.r_[entrance, *legs, norms])
        if x is not None and feasible(scout, x) and scout.hardware_margin(x) > 1e-5:
            if value_only:
                from charge_probe import log_likelihood

                score = log_likelihood(scout, x, knockon=True)
            else:
                score = scout(x, True)["ll"]
            ranked.append((score, masses, x))
    for _, masses, x in sorted(ranked, key=lambda row: -row[0])[:limit]:
        result = track_fit(
            event, model, masses, x, port, budget, module_weight=module_weight, value_only=value_only
        )
        if result is not None:
            yield result
