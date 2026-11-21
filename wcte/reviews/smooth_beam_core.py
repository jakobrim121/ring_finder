"""Fit a continuous primary with two curved sections and one counted core.

Join the arcs halfway through the above-margin arclength. Position, tangent, material range and angular width
are continuous there. Two extra bend coordinates describe the second section.
All light, readout, numerical checks and count support use the original model.
"""

import time
import numpy as np
from wcte.counting import track_counting


class SmoothChart:
    def __init__(self, model, start):
        from terminated_primary_chart import TerminatedPrimaryChart
        from material import material

        self.model = model
        self.margin = float(material(model.particle)["margin_mm"]) / 1000.0
        self.base = TerminatedPrimaryChart(model, start)
        self.initial = np.r_[self.base.initial, 0.0, 0.0]
        self.bounds = np.vstack([self.base.bounds, [(-1.0, 1.0)] * 2])
        self.frame = self.base.base.frame
        self.cached = None
        self.evaluated = {}

    def full(self, y):
        if self.cached is not None and np.array_equal(y, self.cached[0]):
            return self.cached[1:]
        from source import trajectory

        original, mapping, margins, constraints = self.base.full(y[:-2])
        jac = np.pad(mapping, ((0, 0), (0, 2)))
        ran, fraction = original[9:11]
        length = ran * fraction
        dlength = fraction * jac[9] + ran * jac[10]
        # Both sections contain a coherent-core interval. Keeping the split
        # above the margin also preserves the slowing-tail diffusion origin.
        if length <= ran - self.margin:
            core, dcore = length, dlength
        else:
            core, dcore = ran - self.margin, jac[9]
        if core <= 0.0 or length <= 0.0:
            raise ArithmeticError("Smooth primary requires a positive core interval")
        first, dfirst = 0.5 * core, 0.5 * dcore
        second, dsecond = length - first, dlength - dfirst
        u = first / length
        du = dfirst / length - first * dlength / length**2
        p = original[:12].copy()
        dp = jac[:12].copy()
        p[6:9] *= u
        dp[6:9] = u * jac[6:9] + np.outer(original[6:9], du)
        p[10] = first / ran
        dp[10] = dfirst / ran - first * jac[9] / ran**2
        _, tangents, _, dt = trajectory(p, np.ones(1))
        tangent, dtangent = tangents[0], dt[0] @ dp

        # Parallel transport of the first circular arc's total bend.
        # The zero-correction shape is exactly the unsplit circular path.
        bend, dbend = p[6:9], dp[6:9]
        power = float(bend @ bend)
        dPower = 2.0 * bend @ dbend
        if power < 1e-8:
            cosine = 1.0 - power / 2.0 + power**2 / 24.0
            theta_sine = power - power**2 / 6.0 + power**3 / 120.0
            dc = -0.5 + power / 12.0 - power**2 / 240.0
            ds = 1.0 - power / 3.0 + power**2 / 40.0
        else:
            theta = np.sqrt(power)
            cosine, sine_over_theta = np.cos(theta), np.sin(theta) / theta
            theta_sine = theta * np.sin(theta)
            dc, ds = -0.5 * sine_over_theta, 0.5 * (sine_over_theta + cosine)
        rotated = cosine * bend - theta_sine * p[3:6]
        drotated = (
            cosine * dbend
            + np.outer(bend, dc * dPower)
            - theta_sine * dp[3:6]
            - np.outer(p[3:6], ds * dPower)
        )
        ratio = second / first
        dratio = dsecond / first - second * dfirst / first**2
        correction = self.frame @ y[-2:]
        dcorrection = np.zeros((3, len(y)))
        dcorrection[:, -2:] = self.frame
        along = float(tangent @ correction)
        dalong = correction @ dtangent + tangent @ dcorrection
        bend2 = ratio * rotated + correction - along * tangent
        dbend2 = (
            ratio * drotated
            + np.outer(rotated, dratio)
            + dcorrection
            - along * dtangent
            - np.outer(tangent, dalong)
        )
        remaining = ran - first
        dremaining = jac[9] - dfirst
        fraction2 = second / remaining
        dfraction2 = dsecond / remaining - second * dremaining / remaining**2
        x = np.r_[p, tangent, bend2, remaining, fraction2, original[11:14]]
        dx = np.vstack([dp, dtangent, dbend2, dremaining, dfraction2, jac[11:14]])

        # Same smooth-core convention as the existing incoming chart:
        # neither arc can hide a countable direction change inside 1 cm.
        turn2 = 0.01**2 * (bend2 @ bend2) / second**2
        dturn2 = 2.0 * 0.01**2 * (bend2 @ dbend2) / second**2 - 2.0 * turn2 * dsecond / second
        values = np.r_[margins, track_counting.get().merge_angle_rad**2 - turn2]
        derivatives = np.vstack([np.pad(constraints, ((0, 0), (0, 2))), -dturn2])
        self.cached = y.copy(), x, dx, values, derivatives
        self.evaluated[x.tobytes()] = y.copy()
        return self.cached[1:]

    def retained_coordinates(self, full_parameters):
        return self.evaluated[np.asarray(full_parameters).tobytes()].copy()


def make_model(event, background, particle, order):
    from coupled_model import Model
    from chain_sources import paths
    from material import material

    class Continuous(Model):
        delta_scale = 0.0

        def __init__(self):
            super().__init__(event.pmt_ids, event.charges_pe, background, (particle, particle), 4, order)
            self.particle = particle
            lower, upper = self.bounds[9]
            self.range_bounds = (max(lower, material(particle)["margin_mm"] / 1000.0 + 1e-6), upper)
            self.width_bounds = tuple(self.bounds[11])
            self.response_bounds = self.bounds[-2:]

        @property
        def dimension(self):
            # Existing surveyed one-arc convention (10) plus two bends.
            return 12

        def all_constraints(self, x):
            # A smooth join is not an interaction and has no recoil target.
            # Material continuity is exact in the chart; retain containment
            # and each original per-path geometric/material constraint.
            cached = getattr(self, "_smooth_constraints", None)
            if cached is not None and np.array_equal(x, cached[0]):
                return cached[1:]
            values, derivatives = [], []
            for path, transform in paths(x[: self.ng]):
                value, gradient = self.primary_constraints(path, particle)
                values.append(value)
                derivatives.append(np.pad(gradient @ transform, ((0, 0), (0, 2))))
            result = np.concatenate(values), np.vstack(derivatives)
            self._smooth_constraints = (x.copy(), *result)
            return result

    return Continuous()


def counted(model, state):
    # The two sections are a numerical representation of ONE continuous
    # emitting core. Assess support after adding their source intensities.
    core = state["posterior"][1:].sum(0)
    support = float((1.0 - model.obs.no_source(state["means_hit"], core)).sum())
    fractions = (
        np.column_stack([1.0 - core, core])
        if support >= track_counting.get().min_track_pmts
        else np.ones((len(core), 1))
    )
    return fractions, support


def fit(event, background, beam, *, budget=72):
    from balanced_search import feasible
    from track_coordinates import fit as optimize

    tick = time.process_time()
    (particle,) = beam["particles"]
    initial = np.asarray(beam["parameters"])
    history, coordinates = [], None
    for order in (8, 16, 32):
        model = make_model(event, background, particle, order)
        chart = SmoothChart(model, initial)
        if coordinates is not None:
            chart.initial = coordinates.copy()
        result, parameters = optimize(
            model,
            chart.full(chart.initial)[0],
            40,
            budget,
            chart=chart,
            full_likelihood=True,
            check_initial=True,
        )
        # Recover the accepted chart point exactly, without an inverse solve.
        coordinates = chart.retained_coordinates(parameters)
        fine = make_model(event, background, particle, 2 * order)
        state = fine(parameters, True)
        error = abs(state["ll"] - model(parameters, True)["ll"])
        valid = bool(
            feasible(fine, parameters)
            and fine.hardware_margin(parameters) > 1e-5
            and np.min(chart.full(coordinates)[2]) >= -1e-7
        )
        history.append(dict(order=order, evaluations=int(result.nfev), error=float(error), feasible=valid))
        if error < 0.01 and valid:
            fractions, support = counted(fine, state)
            record = dict(
                family="smooth_beam_core",
                particles=list(fine.particles),
                parameters=parameters,
                core_groups=[[0, 1]],
                dimension=fine.dimension,
                order=order,
                count=fractions.shape[1] - 1,
                supports=[support],
                refinement=history,
                cpu_seconds=time.process_time() - tick,
                split_rule="Half the fitted above-margin arclength",
                extra_free_coordinates=2,
                continuous_energy_and_tangent=True,
            )
            return fine, state, fractions, record
    return (
        None,
        None,
        None,
        dict(
            family="smooth_beam_core",
            supported=False,
            reason="No feasible quadrature-qualified smooth candidate",
            refinement=history,
            cpu_seconds=time.process_time() - tick,
        ),
    )
