"""A single charged track with two continuous arcs and one photon partition."""

import numpy as np

from balanced_search import feasible
from coupled_model import Model as Coupled
from electron import turning
from source import trajectory
from track_coordinates import Chart, fit


class TwoArcs(Coupled):
    @property
    def dimension(self):
        # Relative to one arc, the downstream curvature has two coordinates.
        return super().dimension - 5

    def all_constraints(self, x):
        margins, jacobian = super().all_constraints(x)
        # The chart enforces an identical material, tangent and remaining
        # range at the junction. A nuclear recoil constraint is redundant.
        return margins[:-1], jacobian[:-1]


class Continuation:
    def __init__(self, model, start, *, fraction=0.5, first=None):
        self.first = Chart(model, start) if first is None else first
        self.fraction = fraction
        self.initial = np.r_[self.first.initial[:-2], 0.0, 0.0, self.first.initial[-2:]]
        self.bounds = np.vstack([self.first.bounds[:-2], [(-0.5, 0.5)] * 2, self.first.bounds[-2:]])

    def full(self, y):
        base = np.r_[y[:-4], y[-2:]]
        p, base_map, margins, margin_jac = self.first.full(base)
        n = len(y)
        insertion = np.eye(n)[np.r_[np.arange(n - 4), n - 2, n - 1]]
        local = base_map @ insertion
        share = self.fraction
        _, tangents, _, tangent_jacobian = trajectory(p[:12], np.array([share, 1.0]))
        junction, end = tangents
        dj, de = tangent_jacobian @ local[:12]
        bend, dbj, dbe = turning(junction, end)

        # Project the added downstream bend onto the tangent plane at the
        # common junction. Zero added bend recovers the original circle.
        frame = self.first.axes[0][1]
        additional = frame @ y[-4:-2]
        along = junction @ additional
        project = np.eye(3) - np.outer(junction, junction)
        shift = additional - along * junction
        dshift = -along * dj - np.outer(junction, additional @ dj)
        dshift[:, -4:-2] += project @ frame
        derivative = dbj @ dj + dbe @ de + dshift

        first = p[:12].copy()
        first[6:9] *= share
        first[10] *= share
        first_map = local[:12].copy()
        first_map[6:9] *= share
        first_map[10] *= share
        fraction = first[10]
        remaining = first[9] * (1.0 - fraction)
        ratio = (1.0 - share) / share
        second = np.r_[junction, bend + shift, remaining, ratio * fraction / (1.0 - fraction), first[11]]
        second_map = np.vstack(
            [
                dj,
                derivative,
                (1.0 - fraction) * first_map[9] - first[9] * first_map[10],
                ratio * first_map[10] / (1.0 - fraction) ** 2,
                first_map[11],
            ]
        )
        return (
            np.r_[first, second, p[-2:]],
            np.vstack([first_map, second_map, local[-2:]]),
            margins,
            margin_jac @ insertion,
        )


def trial(model, state, iterations=24, maximum_evaluations=40, *, aperture=None, fraction=0.5):
    """Fit a continuous null to a detector-derived single circular track."""
    (particle,) = model.particles
    start = state["parameters"]
    first = None
    if aperture is not None:
        from entrance_track import EntranceChart

        first = EntranceChart(model, start, aperture)
    chart = Continuation(model, start, fraction=fraction, first=first)
    args = model.active[model.ix], model.q, model.bg, (particle, particle), 4
    short = TwoArcs(*args, 16)
    start = chart.full(chart.initial)[0]
    if not feasible(short, start) or short.hardware_margin(start) <= 1e-5:
        return None
    result, x = fit(short, start, iterations, maximum_evaluations, full_likelihood=True, chart=chart)
    if not feasible(short, x) or short.hardware_margin(x) <= 1e-5:
        return None
    if aperture is not None and x[:2] @ x[:2] > aperture.radius**2 + 1e-9:
        return None
    fine = TwoArcs(*args, 32)
    state = fine(x, True)
    error = abs(short(x, True)["ll"] - state["ll"])
    if error >= 0.01:
        return None
    light = state["posterior"][1:].sum(0)
    support = np.sum(1 - fine.obs.no_source(state["means_hit"], light))
    if support < 30 or not np.any(light > 0.5):
        return None
    fractions = np.column_stack([1.0 - light, light])
    return (
        fine,
        state,
        dict(
            family="continuous_track",
            particles=fine.particles,
            quadrature_ll_error=error,
            evaluations=int(result.nfev),
        ),
        fractions,
    )
