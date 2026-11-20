"""The resolved charged-track model without an extra electron trajectory.

The charged tracks still produce their own below-margin light. This nested
alternative removes only the independent electron, so a clean event need not
pay for unused electron directions and a birth point. It uses the current
coherent-core source, calibration and containment, and competes with all
previous models, including the older no-electron controls.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "charged_chain"))
from chain_model import Model as Chain
from chain_sources import paths
from geometry import hardware_margin as path_margin
from material import material


class Model(Chain):
    minimum_sections = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds = self.bounds[: self.ng] + self.bounds[-2:]
        self.ns = self.axes_count = self.groups = 0
        # The shared Fisher preconditioner expects both source blocks. An
        # absent block has zero light and zero parameter columns.
        n = len(self.active)
        self.electron_cache = (
            np.empty(0),
            np.zeros(n),
            np.empty((0, n)),
            np.empty((n, 0)),
            0.0,
            np.empty(0),
            [],
        )

    def axis_starts(self):
        return []

    def path_maps(self, unused):
        return []

    def all_constraints(self, x):
        if len(self.particles) > 1:
            return super().all_constraints(x)
        value, jac = self.primary_constraints(x[:12], self.particles[0])
        return value, np.pad(jac, ((0, 0), (0, 2)))

    def hardware_margin(self, x):
        return min(path_margin(path) for path, _ in paths(x[: self.ng]))

    def __call__(self, x, detail=False):
        if self.last is None or not np.array_equal(x, self.last["parameters"]):
            if self.primary_cache is None or not np.array_equal(x[: self.ng], self.primary_cache[0]):
                parts, jac = self.primary_light(x[: self.ng])
                self.primary_cache = (x[: self.ng].copy(), parts.sum((0, 1)), parts[:, 0], jac)
            _, total, cores, jac = self.primary_cache
            efficiency, background = np.exp(x[-2:])
            signal = efficiency * total
            diffuse = background * self.bg
            means = signal + diffuse
            ll, residual = self.obs.score(means, self.ix)
            posterior = efficiency * cores[:, self.ix] / means[self.ix]
            gradient = (
                np.r_[efficiency * (residual @ jac), signal @ residual, diffuse @ residual] / self.total
            )
            self.last = dict(
                parameters=x.copy(),
                ll=ll,
                log_prior=0.0,
                log_joint=ll,
                objective=-ll / self.total,
                gradient=gradient,
                posterior=np.vstack([1 - posterior.sum(0), posterior]),
                means_hit=means[self.ix],
                response=efficiency,
                primary_intensity=float(signal.sum()),
                electron_intensity=0.0,
                electron_energy_mev=[],
                primary_energy_mev=[
                    float(material(p)["energy_at"](1000 * x[9 + 9 * j])) for j, p in enumerate(self.particles)
                ],
            )
        return self.last if detail else (self.last["objective"], self.last["gradient"])
