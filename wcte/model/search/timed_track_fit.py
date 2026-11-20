"""Refit physical tracks with charge and the first arrival in one objective."""

import numpy as np

from analytic_knockon import render as knockon
from balanced_search import feasible
from chain_sources import paths
from coupled_model import Model as Coupled
from gated_photon_time import evaluate
from joint_prompt_fit import wavefront
from junction_coordinates import Junction
from material import material
from physical_pulse import profile
from source import raw, trajectory
from track_coordinates import Chart, fit as optimize
from wctering.seeding.coherent_timing import C, N_PHASE


def interval_flight(event, path, mass):
    """Approximate each charged arc by its midpoint tangent for the refit."""
    water = material(mass)
    position, axis, dp, da = trajectory(path, np.array([0.5]))
    length = 1000.0 * path[9] * path[10]
    d_length = np.r_[np.zeros(9), 1000.0 * path[10], 1000.0 * path[9], 0.0]
    d_range = np.eye(12)[9] * 1000.0
    phases = np.array([0.0, 0.5, 1.0])
    remaining = 1000.0 * path[9] - phases * length
    energies = water["energy_at"](remaining)
    gamma = 1.0 + energies / water["mass"]
    beta_squared = 1.0 - gamma**-2
    beta = np.sqrt(np.maximum(beta_squared, 1e-8))
    slope = water["energy_at"].derivative()(remaining) / water["mass"] / (gamma**3 * beta)
    slope[beta_squared <= 1e-8] = 0.0
    d_beta = slope[:, None] * (d_range - phases[:, None] * d_length)
    # The Cherenkov angle is allowed to approach zero for a weak tail.
    # The floor keeps the approximate wavefront finite at threshold.
    cosine = np.clip(1.0 / (N_PHASE * beta[1]), 0.001, np.cos(0.08))
    opening = np.arccos(cosine)
    d_angle = cosine / (beta[1] * np.sin(opening)) * d_beta[1] if cosine < np.cos(0.08) else np.zeros(12)
    optic = np.r_[position[0] / 1000.0, axis[0], length / 1000.0, opening, path[11]]
    jac = np.vstack([dp[0] / 1000.0, da[0], d_length / 1000.0, d_angle, np.eye(12)[11]])
    arrival, d_arrival = wavefront(event.pmt_positions_mm, optic)
    # Trapezoidal speed integration. The separate final check restores
    # arclength, spectral and slowing-tail timing moments.
    beta_inverse = 1.0 / beta
    d_inverse = -d_beta / beta[:, None] ** 2
    weights = np.array([[0.25, 0.25, 0.0], [0.25, 0.50, 0.25]]) / C
    flight = length * (weights @ beta_inverse)
    derivative = (weights @ beta_inverse)[:, None] * d_length + length * (weights @ d_inverse)
    return arrival + flight[0], d_arrival @ jac + derivative[0], flight[1], derivative[1]


class Timed(Coupled):
    """Use accurate charge geometry and a cheap differentiable timing law."""

    def __init__(self, event, background, particles, order, clock):
        super().__init__(event.pmt_ids, event.charges_pe, background, particles, 4, order)
        self.event = event
        self.clock = clock
        center = np.median(event.times_ns)
        self.time = event.times_ns - center
        self.gate = np.asarray(event.timing_selection["prompt_window_ns"]) - center
        self.sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
        self.timing_cache = None
        self.timed_state = None

    def primary_light(self, x):
        cores, derivatives, signals = [], [], []
        for (path, mapping), mass in zip(paths(x), self.particles):
            pieces, jac = raw(self.pos, self.normals, path, self.order, False, mass)
            soft, d_soft = knockon(self.pos, self.normals, path, mass, self.order)
            length = 1000.0 * path[9] * path[10]
            d_length = 1000.0 * (path[10] * mapping[9] + path[9] * mapping[10])
            cores.append((length * jac[0, self.ix]) @ mapping + pieces[0, self.ix, None] * d_length)
            signals.append([length * pieces[0], length * pieces[1] + soft])
            derivatives.append((length * jac.sum(0) + d_soft) @ mapping + pieces.sum(0)[:, None] * d_length)
        self.core_jacobian = np.asarray(cores)
        return np.asarray(signals), sum(derivatives)

    def timing(self, x):
        old = self.timing_cache
        if old is not None and np.array_equal(old[0], x):
            return old[1:]
        mean, slope = [], []
        flight = self.clock
        elapsed = np.zeros(len(x))
        for (path, transform), mass in zip(paths(x), self.particles):
            times, d_time, delay, d_delay = interval_flight(self.event, path, mass)
            mean.append(times + flight)
            slope.append(d_time @ transform + elapsed)
            flight += delay
            elapsed += d_delay @ transform
        self.timing_cache = (x.copy(), np.column_stack(mean), np.asarray(slope))
        return self.timing_cache[1:]

    def __call__(self, x, detail=False):
        old = self.timed_state
        if old is not None and np.array_equal(old["parameters"], x):
            return old if detail else (old["objective"], old["gradient"])
        result = dict(super().__call__(x, True))
        _, light, core, derivative = self.primary_cache
        response, diffuse = np.exp(x[-2:])
        other = response * np.maximum(light[self.ix] - core[:, self.ix].sum(0), 0.0)
        sky = diffuse * self.bg[self.ix]
        rings = response * core[:, self.ix]
        rates = np.column_stack([sky + other, rings.T])
        means, time_jacobian = self.timing(x[: self.ng])
        lt, d_rate, d_mean = evaluate(
            self.time, *self.gate, means, rates, self.sigma, self.obs.offsets, self.obs.coefficients
        )
        # The charge term and core rates use the exact transport. Timing
        # uses the midpoint wavefront plus a diffuse tail during this
        # fit, so use half weight until the independent full check.
        core_delta = d_rate[:, 1:] - d_rate[:, :1]
        spatial = response * (
            d_rate[:, 0] @ derivative[self.ix] + np.einsum("ik,kin->n", core_delta, self.core_jacobian)
        )
        spatial += np.einsum("ik,kin->n", d_mean, time_jacobian)
        efficiency = (core_delta * rings.T).sum() + d_rate[:, 0] @ (other + rings.sum(0))
        background = d_rate[:, 0] @ sky
        total = result["ll"] + 0.5 * lt.sum()
        gradient = result["gradient"] - 0.5 / self.total * np.r_[spatial, efficiency, background]
        result.update(objective=-total / self.total, gradient=gradient, timing_approximation=float(lt.sum()))
        self.timed_state = result
        return result if detail else (result["objective"], gradient)


def endpoint(event, model, state, budget, *, aperture=None):
    """Optimize near a detector-derived track and verify with the full timing."""
    gate = event.timing_selection["prompt_window_ns"]
    if gate is None:
        return None
    pulse = profile(event, model, state, separate_knockon=True)
    if pulse is None:
        return None
    x = state["parameters"]
    fit = Timed(event, model.bg, model.particles, 8, pulse["clock_ns"])
    if not feasible(fit, x) or fit.hardware_margin(x) <= 1e-5:
        return None
    if aperture is None:
        chart = Junction(fit, x) if len(model.particles) > 1 else Chart(fit, x)
    else:
        from entrance_track import EntranceChart

        chart = EntranceChart(fit, x, aperture)
    opt, trial = optimize(fit, x, budget, budget, full_likelihood=True, chart=chart)
    if not feasible(fit, trial) or fit.hardware_margin(trial) <= 1e-5:
        return None
    if aperture is not None and trial[:2] @ trial[:2] > aperture.radius**2 + 1e-9:
        return None
    coarse = Coupled(event.pmt_ids, event.charges_pe, model.bg, model.particles, 4, 16)
    fine = Coupled(event.pmt_ids, event.charges_pe, model.bg, model.particles, 4, 32)
    improved = fine(trial, True)
    error = abs(coarse(trial, True)["ll"] - improved["ll"])
    if error >= 0.01:
        return None
    return fine, improved, dict(evaluations=int(opt.nfev), quadrature_ll_error=error, order=16)
