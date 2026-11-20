"""Fit several charged paths to one detector event with independent births."""

import numpy as np

from bare_model import Model as Bare
from geometry import hardware_margin
from recoil_competitor import render
from track_coordinates import Chart


class Independent(Bare):
    """Share absolute response while allowing each path to start separately."""

    def __init__(self, ids, q, bg, particles, order):
        super().__init__(ids, q, bg, particles, 4, order)
        bounds = self.bounds
        self.sections = [Bare(ids, q, bg, (mass,), 4, order) for mass in particles]
        self.bounds = [b for section in self.sections for b in section.bounds[:12]] + bounds[-2:]
        self.ng = 12 * len(particles)

    def primary_light(self, x):
        pieces, derivatives = [], []
        for p, particle in zip(x.reshape(-1, 12), self.particles):
            core, total, jacobian = render(self, p, particle, self.order)
            pieces.append([core, total - core])
            derivatives.append(jacobian)
        return np.asarray(pieces), np.hstack(derivatives)

    def equality(self, x):
        return np.concatenate(
            [m.equality(np.r_[p, x[-2:]]) for m, p in zip(self.sections, x[:-2].reshape(-1, 12))]
        )

    def all_constraints(self, x):
        old = getattr(self, "_independent_envelope", None)
        if old is not None and np.array_equal(x, old[0]):
            return old[1:]
        values, blocks = [], []
        for j, (p, mass) in enumerate(zip(x[:-2].reshape(-1, 12), self.particles)):
            margin, jac = self.primary_constraints(p, mass)
            block = np.zeros((len(margin), len(x)))
            block[:, 12 * j : 12 * j + 12] = jac
            values.append(margin)
            blocks.append(block)
        self._independent_envelope = (x.copy(), np.concatenate(values), np.vstack(blocks))
        return self._independent_envelope[1:]

    def hardware_margin(self, x):
        return min(hardware_margin(p) for p in x[:-2].reshape(-1, 12))

    def __call__(self, x, detail=False):
        # Use the same charge likelihood with the independent arc blocks.
        # The serial diagnostics assume every section has the same origin
        # coordinate and therefore cannot report these local ranges.
        if self.last is None or not np.array_equal(x, self.last["parameters"]):
            if self.primary_cache is None or not np.array_equal(x[:-2], self.primary_cache[0]):
                parts, jac = self.primary_light(x[:-2])
                self.primary_cache = x[:-2].copy(), parts.sum((0, 1)), parts[:, 0], jac
            _, light, cores, jac = self.primary_cache
            efficiency, background = np.exp(x[-2:])
            signal, diffuse = efficiency * light, background * self.bg
            mean = signal + diffuse
            ll, derivative = self.obs.score(mean, self.ix)
            hit_mean = mean[self.ix]
            share = efficiency * cores[:, self.ix] / hit_mean
            gradient = (
                np.r_[efficiency * (derivative @ jac), derivative @ signal, derivative @ diffuse] / self.total
            )
            self.last = dict(
                parameters=x.copy(),
                ll=ll,
                objective=-ll / self.total,
                gradient=gradient,
                posterior=np.vstack([1.0 - share.sum(0), share]),
                means_hit=hit_mean,
            )
        return self.last if detail else (self.last["objective"], self.last["gradient"])


class LocalCharts:
    """Apply the existing single-track coordinate chart to every path."""

    def __init__(self, model, x):
        self.charts = [
            Chart(single, np.r_[p, x[-2:]]) for single, p in zip(model.sections, x[:-2].reshape(-1, 12))
        ]
        self.initial = np.r_[np.concatenate([c.initial[:-2] for c in self.charts]), x[-2:]]
        self.bounds = np.vstack([*[c.bounds[:-2] for c in self.charts], model.bounds[-2:]])
        self.last = None

    def full(self, y):
        if self.last is not None and np.array_equal(y, self.last[0]):
            return self.last[1:]
        n = len(self.charts)
        x = np.zeros(12 * n + 2)
        x[-2:] = y[-2:]
        mapping = np.zeros((len(x), len(y)))
        mapping[-2:, -2:] = np.eye(2)
        margins = np.empty(2 * n)
        margin_jac = np.zeros((len(margins), len(y)))
        for j, chart in enumerate(self.charts):
            old = slice(12 * j, 12 * j + 12)
            new = slice(10 * j, 10 * j + 10)
            row = slice(2 * j, 2 * j + 2)
            local, d, g, dg = chart.full(np.r_[y[new], y[-2:]])
            x[old] = local[:-2]
            mapping[old, new] = d[:-2, :-2]
            margins[row], margin_jac[row, new] = g, dg[:, :-2]
        self.last = y.copy(), x, mapping, margins, margin_jac
        return self.last[1:]
