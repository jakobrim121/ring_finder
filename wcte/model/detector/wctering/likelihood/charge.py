"""Shared-charge likelihood and constrained point/finite ring geometry.

Each source is a normalized light pattern over all 1,843 active PMTs. Row
zero of the internal shapes/posterior is the fixed diffuse background. The
public N by (K+1) result transposes this convention. Mixture weights sum to one.
The charge likelihood conditions on the training PMTs; held-out modules do not
leak into a fold's fit. Geometry uses metres internally, while detector positions
and the public geometry record use millimetres.

This class replaces the old Capped -> Uncapped -> Physical -> Spherical -> ChargeModel
inheritance chain. Its final objective, bounds, derivatives and constraints are
unchanged. Occupancy and water-domain extensions are separate subclasses.
"""

from pathlib import Path
import json
import time
import numpy as np
from wctering.likelihood.constraints import last_input
from wctering.likelihood import constraints as constraint_kernels
from wctering.likelihood.mixture import charge_vectors
from wctering.likelihood.mixture import full_mass, full_jacobian_mass
from scipy.optimize import nnls
from wctering.logmath import logsumexp
from wctering.likelihood.geometry import (
    finite_start,
    NORMALS,
    UPPER,
    BOX,
    active_geometry,
)
from wctering.receiver.point_source import point_source
from wctering.receiver.finite_source import scalar_source as source

ROOT = Path(__file__).resolve().parent.parent
OPTICS_PATH = ROOT / "assets/optics/wcsim_water_optics.json"
WATER = json.loads(OPTICS_PATH.read_text())["water"]
MAX_INDEX = float(np.max(WATER["phase_refractive_index"]))
MAX_OPENING_RAD = float(np.arccos(1 / MAX_INDEX))


def direction(z):
    """Unit axis and its derivative, preserving the original length-three dot."""
    return constraint_kernels.direction(np.asarray(z, float))


class ChargeModel:
    """Charge likelihood with one geometry block per ring and K relative logits.

    Point block: [center_xyz_m, axis_xyz, opening_rad, log(width_rad)].
    Finite block: the same, with length_m inserted before opening_rad.
    The final K values are log(ring_weight / background_weight).
    Axis vectors have an explicit unit-norm equality constraint.
    """

    def __init__(self, ids, charges, fit, background, kind, order=16):
        self.ids, self.pos, self.normals = active_geometry()
        self.ix = np.searchsorted(self.ids, ids)
        np.testing.assert_array_equal(self.ids[self.ix], ids)
        self.w = np.asarray(charges, float).copy()
        self.total = float(self.w.sum())
        self.k = len(fit["vertices"])
        self.kind = kind
        # A point source has no quadrature: every requested order is identical.
        self.quadrature_independent = kind == "point"
        self.order = order
        self.size = 9 if kind == "finite" else 8
        if kind not in ("point", "finite"):
            raise ValueError(kind)
        self.background = np.asarray(background, float)
        assert self.background.shape == (len(self.ids),) and np.all(self.background > 0)
        np.testing.assert_allclose(self.background.sum(), 1, rtol=0, atol=1e-14)
        x = []
        bounds = []
        for v, a, op, wd in zip(fit["vertices"], fit["axes"], fit["openings_deg"], fit["widths_deg"]):
            a = np.asarray(a, float)
            # Preserve the frozen fit order: normalization updates a supplied
            # floating-point seed axis in place. These are internal working
            # geometries, never the caller's raw detector arrays.
            a /= np.linalg.norm(a)
            if kind == "finite":
                ep = finite_start(v, a, op, wd)
                length = np.linalg.norm(ep[3:6] - ep[:3])
                x.extend([*(0.5 * (ep[:3] + ep[3:6])), *a, length, *ep[6:]])
            else:
                x.extend(
                    [
                        *(np.asarray(v) / 1000),
                        *a,
                        np.deg2rad(op),
                        np.log(np.deg2rad(wd)),
                    ]
                )
            bounds.extend([*BOX, *[(-1.0, 1.0)] * 3])
            if kind == "finite":
                bounds.append((0.0, 4.8))
            bounds.extend(
                [
                    (np.deg2rad(12), np.deg2rad(55)),
                    (np.log(np.deg2rad(2.5)), np.log(np.deg2rad(20))),
                ]
            )
        pi = np.asarray(fit["weights"])
        x.extend(np.clip(np.log(pi[1:] / pi[0]), -8.0, 8.0))
        bounds.extend([(-8.0, 8.0)] * self.k)
        self.x = np.asarray(x, float)
        self.bounds = bounds
        self.water_upper = np.tile(UPPER, 2 * self.k)
        # SLSQP expects nonnegative inequality margins and zero equality
        # residuals. Supply them directly instead of converting a general
        # NonlinearConstraint at each solve. Negate AFTER subtraction, as in
        # SciPy's adapter, to preserve even the sign of a zero water margin.
        # WaterDomain extends self.water_upper before these closures are used.
        self.constraints = [
            dict(
                type="ineq",
                fun=lambda x: -(self.water_values(x) - self.water_upper),
                jac=lambda x: -self.water_jacobian(x),
            ),
            dict(
                type="eq",
                fun=lambda x: self.axis_norms(x) - 1.0,
                jac=self.axis_jacobian,
            ),
        ]
        self.last = None
        self.calls = 0
        # The source opening cannot exceed the largest Cherenkov angle in water.
        # Apply this after seed construction, in the same order as frozen V101.
        assert np.all(np.isfinite(self.w)) and np.all(self.w > 0)
        for j in range(self.k):
            i = self.size * (j + 1) - 2
            self.bounds[i] = (0.0, MAX_OPENING_RAD)
            self.x[i] = np.clip(self.x[i], 0.0, MAX_OPENING_RAD)
        mask = np.asarray(fit.get("training_mask", np.ones(len(self.ids), bool)))
        if mask.dtype != bool or mask.shape != self.ids.shape or not mask.any() or not mask[self.ix].all():
            raise ValueError("Invalid training mask")
        self.all_training = bool(mask.all())
        self.training_mask = mask.copy()
        self.training_mask.flags.writeable = False

    def component(self, parameters):
        """Evaluate one ring's normalized light and geometry derivatives."""
        return (
            source(self.pos, self.normals, parameters, self.order)
            if self.kind == "finite"
            else point_source(self.pos, self.normals, parameters)
        )

    def __call__(self, x, detail=False):
        x = np.asarray(x, float)
        if self.last is not None and np.array_equal(x, self.last["x"]):
            return self._response(detail)
        tick = time.monotonic()
        # Evaluate each source on ALL active PMTs before selecting observed hits.
        # The normalization therefore includes the detector's unhit regions.
        shapes = [self.background]
        jac = []
        for j in range(self.k):
            p = x[j * self.size : (j + 1) * self.size]
            s, d = self.component(p)
            shapes.append(s)
            jac.append(d)
        shapes = np.asarray(shapes)
        logits = np.concatenate(([0.0], x[self.k * self.size :]))
        pi = np.exp(logits - logsumexp(logits))
        # A module-held-out fit conditions on the retained training region.
        # Subtracting total*log(prob) gives the conditional log likelihood.
        mass = full_mass(shapes) if self.all_training else shapes[:, self.training_mask].sum(1)
        prob = float(pi @ mass)
        if not 0.0 < prob <= 1.0 + 1e-12:
            raise FloatingPointError("Invalid training probability")
        density, post, weighted_inverse = charge_vectors(shapes, pi, self.ix, self.w)
        # Responsibilities allocate measured charge; components need not be
        # exclusive at a PMT. Internal row zero remains diffuse background.
        raw_ll = float(self.w @ np.log(density))
        ll = raw_ll - self.total * np.log(prob)
        g = np.zeros_like(x)
        for j, d in enumerate(jac):
            g[j * self.size : (j + 1) * self.size] = pi[j + 1] * (
                (full_jacobian_mass(d) if self.all_training else d[self.training_mask].sum(0)) / prob
                - weighted_inverse @ d[self.ix] / self.total
            )
        g[self.k * self.size :] = pi[1:] * mass[1:] / prob - (post[1:] @ self.w) / self.total
        self.calls += 1
        self.last = dict(
            x=x.copy(),
            ll=ll,
            unconditional_ll=raw_ll,
            objective=-ll / self.total,
            gradient=g,
            posterior=post,
            shapes=shapes,
            shape_jacobians=jac,
            weights=pi,
            unconditional_density=density,
            training_probability=prob,
            training_component_probability=mass,
            seconds=time.monotonic() - tick,
            call=self.calls,
        )
        return self._response(detail)

    def _response(self, detail):
        if detail:
            return self._with_details()
        return self.last["objective"], self.last["gradient"].copy()

    def _with_details(self):
        """Complete diagnostic light arrays only when a caller requests them."""
        state = self.last
        if "expected_charge_all" not in state:
            probability = state["training_probability"]
            state["density"] = state["unconditional_density"] / probability
            state["expected_charge_all"] = self.total / probability * (state["weights"] @ state["shapes"])
        return state

    @last_input
    def axis_norms(self, x):
        return constraint_kernels.axis_norms(x, self.size, self.k)

    @last_input
    def axis_jacobian(self, x):
        return constraint_kernels.axis_jacobian(x, self.size, self.k)

    def lengths(self, x):
        return np.array([x[self.size * j + 6] if self.kind == "finite" else 0.0 for j in range(self.k)])

    @last_input
    def water_values(self, x):
        return constraint_kernels.water_values(x, self.size, self.k, NORMALS)

    @last_input
    def water_jacobian(self, x):
        return constraint_kernels.water_jacobian(x, self.size, self.k, NORMALS)

    def extra_inequalities(self, x):
        """Optional constraints as values <= 0, Jacobian rows, and audit labels."""
        return np.empty(0), np.empty((0, len(x))), []

    def kkt(self, x, g):
        """Remove allowed constraint forces from the objective gradient.

        The remaining infinity norm tests stationarity on the feasible surface.
        An optimizer's success flag alone does not establish this condition.
        """
        normals = []
        labels = []
        water = self.water_upper - self.water_values(x)
        wj = self.water_jacobian(x)
        for i in np.flatnonzero(water <= 1e-6):
            normals.append(wj[i])
            labels.append(f"water{i}")
        for i, (lo, hi) in enumerate(self.bounds):
            upper_active = hi - x[i] <= 1e-6
            lower_active = x[i] - lo <= 1e-6
            if upper_active or lower_active:
                # Most bounds are inactive. Construct a constraint normal
                # only when it will participate in the stationarity check.
                v = np.zeros_like(x)
                v[i] = 1.0
                if upper_active:
                    normals.append(v)
                    labels.append(f"upper{i}")
                if lower_active:
                    normals.append(-v)
                    labels.append(f"lower{i}")
        extra_values, extra_jacobian, extra_labels = self.extra_inequalities(x)
        for index in np.flatnonzero(extra_values >= -1e-6):
            normals.append(extra_jacobian[index])
            labels.append(extra_labels[index])
        # Eliminate the unrestricted equality multipliers before the NNLS
        # inequality solve. The axis rows are disjoint and mutually orthogonal.
        eq = self.axis_jacobian(x)
        q = eq.T / np.linalg.norm(eq, axis=1)
        pg = g - q @ (q.T @ g)
        if normals:
            matrix = np.asarray(normals).T
            matrix -= q @ (q.T @ matrix)
            mu, _ = nnls(matrix, -pg, maxiter=2000)
            residual = pg + matrix @ mu
        else:
            mu = np.empty(0)
            residual = pg
        return dict(
            residual_inf=float(abs(residual).max()),
            extra_constraint_violation=float(np.maximum(extra_values, 0).max(initial=0)),
            active=labels,
            multipliers=mu,
            water_margin_m=float(water.min()),
            minimum_length_m=float(self.lengths(x).min()),
            axis_unit_error=float(abs(self.axis_norms(x) - 1).max()),
            box_margin=float(min(min(x[i] - lo, hi - x[i]) for i, (lo, hi) in enumerate(self.bounds))),
        )

    def geometry(self, x):
        p = x[: self.size * self.k].reshape(self.k, self.size)
        axes = np.array([direction(z[3:6])[0] for z in p])
        result = dict(
            vertices=p[:, :3] * 1000,
            axes=axes,
            openings_deg=np.rad2deg(p[:, -2]),
            widths_deg=np.rad2deg(np.exp(p[:, -1])),
        )
        if self.kind == "finite":
            result.update(
                starts_mm=(p[:, :3] - 0.5 * p[:, 6, None] * axes) * 1000,
                ends_mm=(p[:, :3] + 0.5 * p[:, 6, None] * axes) * 1000,
                lengths_mm=p[:, 6] * 1000,
            )
        return result
