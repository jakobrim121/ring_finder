"""Use the optical shapes to predict which active PMTs recorded a prompt hit.

For shape s and total effective intensity A, lambda=A*s and P(hit)=1-exp(-lambda).
Every active PMT contributes once, including unhit tubes. A is an occupancy
scale, not calibrated analogue charge. The returned posterior still allocates
each observed hit's measured charge among background and ring components.
"""

import time
import numpy as np
from scipy.special import softmax
from wctering.likelihood.charge import ChargeModel
from wctering.likelihood.mixture import occupancy_vectors
from wctering.receiver.angular_source import source
from wctering.receiver.finite_source import scalar_source
from wctering.likelihood.charge import point_source


class OccupancyModel(ChargeModel):
    """ChargeModel parameters plus one log-intensity; geometry and weights are inherited."""

    reference_arithmetic = False

    def __init__(self, ids, q, fit, bg, kind, order=16):
        super().__init__(ids, q, fit, bg, kind, order)
        self.bounds = list(self.bounds) + [(np.log(1e-6), np.log(1e6))]
        self.x = np.concatenate((self.x, [np.log(self.total)]))
        self.total = float(len(self.ix))
        # This likelihood requires one retained digitized record per hit PMT.
        assert len(np.unique(self.ix)) == len(self.ix)
        self.observation = np.zeros(len(self.ids))
        self.observation[self.ix] = 1.0
        self.last = None

    def component(self, parameters):
        """Occupancy may use the validated angular integral; charge does not."""
        light = scalar_source if self.reference_arithmetic else source
        return (
            light(self.pos, self.normals, parameters, self.order)
            if self.kind == "finite"
            else point_source(self.pos, self.normals, parameters)
        )

    def __call__(self, x, detail=False):
        x = np.asarray(x, float)
        if self.last is not None and np.array_equal(x, self.last["x"]):
            return self._response(detail)
        tick = time.monotonic()
        shapes = [self.background]
        jac = []
        offset = self.k * self.size
        for j in range(self.k):
            p = x[j * self.size : (j + 1) * self.size]
            s, d = self.component(p)
            shapes.append(s)
            jac.append(d)
        shapes = np.asarray(shapes)
        pi = softmax(np.concatenate(([0.0], x[offset:-1])))
        amplitude = np.exp(x[-1])
        density = pi @ shapes
        lam = amplitude * density
        hit = -np.expm1(-lam)
        mask = self.training_mask
        y = self.observation
        kept_y = y if self.all_training else y[mask]
        kept_hit = hit if self.all_training else hit[mask]
        kept_lam = lam if self.all_training else lam[mask]
        # d(-log L)/d(lambda) = 1 - observation/P(hit). Chain this through
        # geometry, softmax mixture weights, and log(total intensity).
        if self.all_training:
            terms, dmean, residual = occupancy_vectors(y, hit, np.log(hit), lam)
            ll = float(terms.sum())
        else:
            ll = float((kept_y * np.log(kept_hit) - (1 - kept_y) * kept_lam).sum())
            dmean = np.zeros(len(self.ids))
            dmean[mask] = 1.0 - y[mask] / hit[mask]
            residual = dmean * lam
        residual_total = residual.sum()
        gradient = np.zeros_like(x)
        for j, d in enumerate(jac):
            gradient[j * self.size : (j + 1) * self.size] = amplitude * pi[j + 1] * (dmean @ d) / self.total
        gradient[offset:-1] = pi[1:] * (amplitude * (shapes[1:] @ dmean) - residual_total) / self.total
        gradient[-1] = residual_total / self.total
        # Component responsibilities sum to one for each observed PMT.
        self.calls += 1
        self.last = dict(
            x=x.copy(),
            ll=ll,
            objective=-ll / self.total,
            gradient=gradient,
            shapes=shapes,
            weights=pi,
            optical_density=density,
            expected_charge_all=lam,
            expected_light_mean_kind="effective_occupancy_intensity_not_calibrated_analog_charge",
            hit_probability=hit,
            seconds=time.monotonic() - tick,
            call=self.calls,
        )
        return self._response(detail)

    def _response(self, detail):
        # Optimizer callbacks need ll and gradient, not a charge partition.
        return self.last if detail == "optimizer" else super()._response(detail)

    def _with_details(self):
        """Construct the same posterior and totals, once per evaluated state."""
        state = self.last
        if "posterior" not in state:
            pi, shapes = state["weights"], state["shapes"]
            density = state["optical_density"]
            lam = state["expected_charge_all"]
            state["posterior"] = pi[:, None] * shapes[:, self.ix] / density[self.ix]
            state["density"] = lam[self.ix]
            state["expected_total_charge"] = float(lam.sum())
            state["expected_training_charge"] = float(
                (lam if self.all_training else lam[self.training_mask]).sum()
            )
        return state
