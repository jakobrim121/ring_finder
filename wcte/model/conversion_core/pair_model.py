"""A charged primary plus an independent electron or conversion pair.

The old below-margin electron source is an exact limit of the one-electron
family. The two-electron family adds five independent fitted coordinates:
two later tangents and one range. Full parameter counting is retained.

This charge-only approximation permits neutral flight to a distinct vertex.
It is not a complete electromagnetic shower or an energy-conserving event
generator. Every high-energy portion contributes a possible core, which is
counted only if the existing observed-PMT support requirement is satisfied.
"""

import numpy as np
from scipy.optimize import nnls
from single_background import primary_inequality_jacobian
from readout import observation, parameters
from wctering.likelihood.geometry import active_geometry, BOX
from source import raw
from material import material
from electron_parts import source, core_turn, paths, prior, containment, containment_jacobian, clearance
from geometry import equalities, inequalities, hardware_margin, MAX_TAIL_TURN
import curved_legs


class Model:
    source_function = staticmethod(source)
    prior_function = staticmethod(prior)
    secondary_material = staticmethod(material)

    def __init__(self, ids, q, background, particle, pieces=3, order=32, pair=False, groups=1):
        self.active, self.pos, self.normals = active_geometry()
        self.ix = np.searchsorted(self.active, ids)
        np.testing.assert_array_equal(self.active[self.ix], ids)
        self.q = np.asarray(q)
        self.obs = observation(q)
        self.total = max(float(self.q.sum() / self.obs.gain), 1.0)
        self.bg = np.asarray(background)
        self.pieces, self.order = pieces, order
        self.last = self.primary_cache = self.electron_cache = None

        # A two-section parent has one shared vertex, two axes and two
        # range/fraction pairs. A single parent uses the 12-coordinate arc.
        self.connected = isinstance(particle, (tuple, list))
        bend_bounds = [(-MAX_TAIL_TURN, MAX_TAIL_TURN)] * 3
        axis_bounds = [(-1.0, 1.0)] * 3
        # The calibrated angular band accounts for residual optical smearing.
        width_bounds = tuple(np.log(np.deg2rad([0.5, 2.0])))

        def range_bounds(species):
            water = material(species)
            return [
                (float(water["range_at"](water["energy"][0] + 1)) / 1000, water["maximum_range_mm"] / 1000),
                (0.0001, 1.0),
            ]

        if self.connected:
            assert len(np.unique(self.ix)) == len(self.ix)
            self.particles = tuple(particle)
            self.particle = particle[0]
            parent_bounds = [
                *BOX,
                *axis_bounds,
                *axis_bounds,
                *(bound for species in self.particles for bound in range_bounds(species)),
                width_bounds,
                width_bounds,
                *bend_bounds,
                *bend_bounds,
            ]
        else:
            self.particle = particle
            parent_bounds = [*BOX, *axis_bounds, *bend_bounds, *range_bounds(particle), width_bounds]
        self.ng = len(parent_bounds)
        config = parameters()
        self.response = config["light_coefficient_photons"]
        low, high = self.response * np.asarray(config["light_coefficient_relative_bounds"])
        scale_bounds = [(np.log(low), np.log(high)), (-18.0, np.log(1e6))]

        self.pair, self.groups = pair, groups
        self.axes_per_group = 2 * pieces - 1 if pair else pieces
        self.axes_count = groups * self.axes_per_group
        self.ranges_count = 2 if pair else 1
        self.group_size = 3 + 3 * self.axes_per_group + self.ranges_count
        self.ns = groups * self.group_size
        # Each additional electron group has its own vertex and range.
        electron = material("electron")
        table = self.secondary_material("electron")
        range_bound = (
            np.log(float(electron["range_at"](electron["energy"][0] + 1)) / 1000),
            np.log(table["maximum_range_mm"] / 1000) - 1e-12,
        )
        group_bounds = [*BOX, *axis_bounds * self.axes_per_group, *[range_bound] * self.ranges_count]
        self.bounds = parent_bounds + groups * group_bounds + scale_bounds

    def path_maps(self, z):
        """Individual electron paths and their maps into all secondary coordinates."""
        output = []
        for group in range(self.groups):
            start = group * self.group_size
            stop = start + self.group_size
            for path, local in paths(z[start:stop], self.pieces, self.pair):
                transform = np.zeros((len(path), self.ns))
                transform[:, start:stop] = local
                output.append((path, transform))
        return output

    def axis_starts(self):
        return [
            self.ng + group * self.group_size + 3 + 3 * j
            for group in range(self.groups)
            for j in range(self.axes_per_group)
        ]

    @property
    def dimension(self):
        # Remove unit-axis/transverse constraints and the common diffuse scale.
        return len(self.bounds) - ((4 if self.connected else 2) + self.axes_count) - 1

    def __call__(self, x, detail=False):
        if self.last is not None and np.array_equal(x, self.last["parameters"]):
            return self.last if detail else (self.last["objective"], self.last["gradient"])
        if self.primary_cache is None or not np.array_equal(x[: self.ng], self.primary_cache[0]):
            if self.connected:
                parts, derivative = curved_legs.source(
                    self.pos, self.normals, x[: self.ng], self.particles, self.order, False
                )
                cores, light = parts[:, 0], parts.sum((0, 1))
            else:
                parts, jac = raw(self.pos, self.normals, x[:12], self.order, False, self.particle)
                length = 1000 * x[9] * x[10]
                derivative = length * jac.sum(0)
                derivative[:, 9] += 1000 * x[10] * parts.sum(0)
                derivative[:, 10] += 1000 * x[9] * parts.sum(0)
                cores, light = length * parts[:1], length * parts.sum(0)
            self.primary_cache = (x[: self.ng].copy(), light, cores, derivative)
        _, primary, primary_cores, primary_jac = self.primary_cache
        z = x[self.ng : -2]
        if self.electron_cache is None or not np.array_equal(z, self.electron_cache[0]):
            light = np.zeros(len(self.active))
            cores = []
            derivative = np.zeros((len(self.active), self.ns))
            log_prior = 0.0
            prior_jac = np.zeros(self.ns)
            energies = []
            for path, transform in self.path_maps(z):
                parts, jac = self.source_function(self.pos, self.normals, path, self.order)
                light += parts.sum(0)
                cores.append(parts[0])
                derivative += jac.sum(0) @ transform
                value, gradient = self.prior_function(path)
                log_prior += value
                prior_jac += gradient @ transform
                energies.append(
                    float(self.secondary_material("electron")["energy_at"](1000 * np.exp(path[-1])))
                )
            self.electron_cache = (
                z.copy(),
                light,
                np.array(cores),
                derivative,
                log_prior,
                prior_jac,
                energies,
            )
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
        energies_main = (
            [float(material(p)["energy_at"](1000 * x[9 + 2 * j])) for j, p in enumerate(self.particles)]
            if self.connected
            else float(material(self.particle)["energy_at"](1000 * x[9]))
        )
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
            primary_energy_mev=energies_main,
        )
        return self.last if detail else (self.last["objective"], self.last["gradient"])

    def equality(self, x):
        axes = np.array([x[j : j + 3] for j in self.axis_starts()])
        main = curved_legs.equality(x[:21]) if self.connected else equalities(x[:12])
        return np.r_[main, (axes * axes).sum(1) - 1]

    def equality_jacobian(self, x):
        main_size = 4 if self.connected else 2
        jac = np.zeros((main_size + self.axes_count, len(x)))
        if self.connected:
            jac[:4, :21] = curved_legs.equality_jacobian(x[:21])
        else:
            jac[0, 3:6] = 2 * x[3:6]
            jac[1, 3:6] = x[6:9]
            jac[1, 6:9] = x[3:6]
        for j, start in enumerate(self.axis_starts()):
            jac[main_size + j, start : start + 3] = 2 * x[start : start + 3]
        return jac

    def inequality(self, x):
        values = [
            curved_legs.inequality(x[:21], self.particles)
            if self.connected
            else inequalities(x[:12], self.particle)
        ]
        for path, _ in self.path_maps(x[self.ng : -2]):
            values.extend([containment(path), np.atleast_1d(core_turn(path)[0])])
        return np.concatenate(values)

    def inequality_jacobian(self, x):
        main = (
            curved_legs.inequality_jacobian(x[:21], self.particles)
            if self.connected
            else primary_inequality_jacobian(x[:12], self.particle)
        )
        rows = []
        for path, transform in self.path_maps(x[self.ng : -2]):
            rows.extend([containment_jacobian(path) @ transform, (core_turn(path)[1] @ transform)[None, :]])
        secondary = np.vstack(rows)
        jac = np.zeros((len(main) + len(secondary), len(x)))
        jac[: len(main), : self.ng] = main
        jac[len(main) :, self.ng : -2] = secondary
        return jac

    def hardware_margin(self, x):
        main = curved_legs.hardware_margin(x[:21]) if self.connected else hardware_margin(x[:12])
        return min(main, *(clearance(path) for path, _ in self.path_maps(x[self.ng : -2])))

    def constraints(self):
        return [
            dict(type="eq", fun=self.equality, jac=self.equality_jacobian),
            dict(type="ineq", fun=self.inequality, jac=self.inequality_jacobian),
        ]

    def stationarity(self, x, gradient):
        """Measure the first-order residual with nonnegative forces on active constraints."""
        basis = np.linalg.qr(self.equality_jacobian(x).T, mode="reduced")[0]
        projected = gradient - basis @ (basis.T @ gradient)
        margins = self.inequality(x)
        jac = self.inequality_jacobian(x)
        normals = [-jac[j] for j in np.flatnonzero(margins <= 1e-6)]
        eye = np.eye(len(x))
        for j, (low, high) in enumerate(self.bounds):
            if x[j] - low <= 1e-6:
                normals.append(-eye[j])
            if high - x[j] <= 1e-6:
                normals.append(eye[j])
        if normals:
            matrix = np.asarray(normals).T
            matrix -= basis @ (basis.T @ matrix)
            projected += matrix @ nnls(matrix, -projected, maxiter=5000)[0]
        return float(abs(projected).max())


def counted(model, state):
    """Apply the unchanged support and connected-lineage counting rules.

    Conversion siblings are separate charged sources. Only the two connected
    primary legs may merge under the existing 15-degree lineage convention.
    The low-energy parts remain background even when their parent is counted.
    """
    from curved_legs import core_axes

    cores = state["posterior"][1:]
    support = [float((1 - model.obs.no_source(state["means_hit"], core)).sum()) for core in cores]
    keep = [j for j, value in enumerate(support) if value >= 30]
    merge = False
    if model.connected and 0 in keep and 1 in keep:
        a, b = core_axes(state["parameters"][:21])
        merge = a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) >= np.cos(np.deg2rad(15.0))
    selected = []
    for j in keep:
        if merge and j == 1:
            continue
        selected.append(cores[0] + cores[1] if merge and j == 0 else cores[j])
    if not selected:
        return np.ones((len(state["means_hit"]), 1))
    return np.vstack([1 - np.sum(selected, axis=0), *selected]).T
