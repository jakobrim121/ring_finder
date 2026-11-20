"""Connected charged sections plus the existing full-energy electron source.

This research family adds physical interactions rather than approximating
several masses with one energy-loss curve. It shares the calibrated response,
charge-sharing likelihood, secondary scattering prior, integration checks,
and count-support convention with the existing conversion family. Every
additional independent coordinate is charged in the model comparison.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "conversion_detector"))
from detector_model import Electron
from material import material
from source import trajectory
from geometry import hardware_margin as path_margin
from electron_parts import source, prior, clearance as electron_margin
from chain_sources import paths, interactions, light


class Model(Electron):
    minimum_sections = 2

    def __init__(self, ids, q, background, particles, pieces=4, order=32):
        particles = tuple(particles)
        assert self.minimum_sections <= len(particles) <= 4
        super().__init__(ids, q, background, particles[0], pieces, order)
        original_bounds = self.bounds
        self.particles, self.connected = particles, True
        self.ng = 3 + 9 * len(particles)
        bounds = original_bounds[:3]
        for particle in particles:
            m = material(particle)
            section = original_bounds[3:12].copy()
            section[6] = (float(m["range_at"](m["energy"][0] + 1)) / 1000, m["maximum_range_mm"] / 1000)
            bounds += section
        self.bounds = bounds + original_bounds[12:]

    @property
    def dimension(self):
        return len(self.bounds) - 2 * len(self.particles) - self.axes_count - 1

    def primary_light(self, x):
        """Source hook for separately qualified trajectory hypotheses."""
        return light(self.pos, self.normals, x, self.particles, self.order)

    def __call__(self, x, detail=False):
        if self.last is not None and np.array_equal(x, self.last["parameters"]):
            return self.last if detail else (self.last["objective"], self.last["gradient"])
        if self.primary_cache is None or not np.array_equal(x[: self.ng], self.primary_cache[0]):
            parts, jac = self.primary_light(x[: self.ng])
            self.primary_cache = (x[: self.ng].copy(), parts.sum((0, 1)), parts[:, 0], jac)
        _, primary, primary_cores, primary_jac = self.primary_cache
        z = x[self.ng : -2]
        if self.electron_cache is None or not np.array_equal(z, self.electron_cache[0]):
            total = np.zeros(len(self.active))
            cores = []
            derivative = np.zeros((len(self.active), self.ns))
            lp = 0.0
            dp = np.zeros(self.ns)
            energies = []
            for path, transform in self.path_maps(z):
                parts, jac = source(self.pos, self.normals, path, self.order)
                total += parts.sum(0)
                cores.append(parts[0])
                derivative += jac.sum(0) @ transform
                value, gradient = prior(path)
                lp += value
                dp += gradient @ transform
                energies.append(float(material("electron")["energy_at"](1000 * np.exp(path[-1]))))
            self.electron_cache = (z.copy(), total, np.array(cores), derivative, lp, dp, energies)
        _, electron, electron_cores, electron_jac, lp, dp, energies = self.electron_cache
        efficiency, background = np.exp(x[-2:])
        signal = efficiency * (primary + electron)
        diffuse = background * self.bg
        means = signal + diffuse
        ll, residual = self.obs.score(means, self.ix)
        gradient = (
            np.r_[
                efficiency * (residual @ primary_jac),
                efficiency * (residual @ electron_jac) - dp,
                signal @ residual,
                diffuse @ residual,
            ]
            / self.total
        )
        cores = efficiency * np.vstack([primary_cores, electron_cores])[:, self.ix] / means[self.ix]
        self.last = dict(
            parameters=x.copy(),
            ll=ll,
            log_prior=lp,
            log_joint=ll + lp,
            objective=-(ll + lp) / self.total,
            gradient=gradient,
            posterior=np.vstack([1 - cores.sum(0), cores]),
            means_hit=means[self.ix],
            response=efficiency,
            primary_intensity=float(efficiency * primary.sum()),
            electron_intensity=float(efficiency * electron.sum()),
            electron_energy_mev=energies,
            primary_energy_mev=[
                float(material(p)["energy_at"](1000 * x[9 + 9 * j])) for j, p in enumerate(self.particles)
            ],
        )
        return self.last if detail else (self.last["objective"], self.last["gradient"])

    def equality(self, x):
        values = []
        for path, _ in paths(x[: self.ng]):
            values.extend([path[3:6] @ path[3:6] - 1, path[3:6] @ path[6:9]])
        values.extend(x[j : j + 3] @ x[j : j + 3] - 1 for j in self.axis_starts())
        return np.array(values)

    def equality_jacobian(self, x):
        jac = np.zeros((2 * len(self.particles) + self.axes_count, len(x)))
        for j in range(len(self.particles)):
            start = 3 + 9 * j
            jac[2 * j, start : start + 3] = 2 * x[start : start + 3]
            jac[2 * j + 1, start : start + 3] = x[start + 3 : start + 6]
            jac[2 * j + 1, start + 3 : start + 6] = x[start : start + 3]
        for j, start in enumerate(self.axis_starts()):
            jac[2 * len(self.particles) + j, start : start + 3] = 2 * x[start : start + 3]
        return jac

    def all_constraints(self, x):
        previous = getattr(self, "_envelope_cache", None)
        if previous is not None and np.array_equal(x, previous[0]):
            return previous[1:]
        values, rows = [], []
        mapped = paths(x[: self.ng])
        for (path, transform), particle in zip(mapped, self.particles):
            value, jac = self.primary_constraints(path, particle)
            row = np.zeros((len(value), len(x)))
            row[:, : self.ng] = jac @ transform
            values.append(value)
            rows.append(row)
        value, jac = interactions(mapped, self.particles)
        row = np.zeros((len(value), len(x)))
        row[:, : self.ng] = jac
        values.append(value)
        rows.append(row)
        for path, transform in self.path_maps(x[self.ng : -2]):
            value, jac = self.secondary_constraints(path)
            row = np.zeros((len(value), len(x)))
            row[:, self.ng : -2] = jac @ transform
            values.append(value)
            rows.append(row)
        value, jac = np.concatenate(values), np.vstack(rows)
        self._envelope_cache = (x.copy(), value, jac)
        return value, jac

    def hardware_margin(self, x):
        return min(
            *(path_margin(p) for p, _ in paths(x[: self.ng])),
            *(electron_margin(p) for p, _ in self.path_maps(x[self.ng : -2])),
        )


def counted(model, state):
    """Keep the existing support-first, 15-degree connected-lineage convention."""
    cores = state["posterior"][1:]
    support = [(1 - model.obs.no_source(state["means_hit"], core)).sum() for core in cores]
    keep = {j for j, value in enumerate(support) if value >= 30}
    groups = [[j] for j in range(len(cores)) if j in keep]
    mapped = paths(state["parameters"][: model.ng])
    for j, ((a, _), (b, _)) in enumerate(zip(mapped, mapped[1:])):
        incoming = trajectory(a, np.array([1.0]))[1][0]
        outgoing = b[3:6] / np.linalg.norm(b[3:6])
        if j in keep and j + 1 in keep and incoming @ outgoing >= np.cos(np.deg2rad(15.0)):
            left = next(group for group in groups if j in group)
            right = next(group for group in groups if j + 1 in group)
            if left is not right:
                left.extend(right)
                groups.remove(right)
    selected = [cores[group].sum(0) for group in groups]
    return (
        np.vstack([1 - np.sum(selected, axis=0), *selected]).T
        if selected
        else np.ones((len(state["means_hit"]), 1))
    )
