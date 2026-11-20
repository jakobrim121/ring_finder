"""Convolve the first detected PE with the measured WCSim digit time jitter."""

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm


class Selection:
    """Reuse the calibrated detection law at each common digit clock shift."""

    def __init__(self, base, *, sigma=0.65, quadrature=9):
        from selected_pulse import likelihood

        self.time = base.time
        self.gate = base.gate
        self.hit = base.hit
        self.model = base.model
        self.obs = base.obs
        # Preserve the declared marginal single-photon width. The digit's
        # jitter acts after taking the minimum of all PE arrival times.
        self.width = np.sqrt(np.maximum(base.width**2 - sigma**2, 0.25**2))
        nodes, weights = roots_hermitenorm(quadrature)
        self.shifts = nodes * sigma
        self.log_weight = np.log(weights / weights.sum())
        self.likelihood = likelihood

    def conditional(self, rates, centers):
        return [
            self.likelihood(
                self.hit,
                self.time - shift,
                *(self.gate - shift),
                centers,
                rates,
                self.width,
                self.obs.offsets,
                self.obs.coefficients,
                self.obs.r,
                self.obs.p2,
            )
            for shift in self.shifts
        ]

    def integrated_weights(self, terms):
        joint = self.log_weight[:, None] + np.stack([t[0] for t in terms])
        total = logsumexp(joint, axis=0)
        return total, np.exp(joint - total[None])

    def score(self, rates, centers):
        terms = self.conditional(rates, centers)
        total, weight = self.integrated_weights(terms)
        grad_rate = sum(w[:, None] * term[1] for w, term in zip(weight, terms, strict=True))
        grad_time = sum(w[:, None] * term[2] for w, term in zip(weight, terms, strict=True))
        return total, grad_rate, grad_time

    def ownership(self, rates, centers, core_masks, *, prompt):
        return self.ownership_with_score(rates, centers, core_masks, prompt=prompt)[:3]

    def ownership_with_score(self, rates, centers, core_masks, *, prompt):
        """Reuse the integrated likelihood needed for the ownership posterior."""
        from physical_sharing import ownership as first_arrival_shares
        from prompt_ownership import ownership as prompt_shares

        total, weight = self.integrated_weights(self.conditional(rates, centers))
        ix = self.model.ix
        hit_rates, hit_times, width = rates[ix], centers[ix], self.width[ix]
        terms = (hit_rates, hit_times, width, self.obs.offsets, self.obs.coefficients)
        timed = np.zeros_like(hit_rates)
        in_gate = np.zeros_like(hit_rates) if prompt else None
        support = np.zeros((len(hit_rates), len(core_masks))) if prompt else None
        for i, shift in enumerate(self.shifts):
            posterior = weight[i, ix, None]
            digit_time, selected_gate = self.time - shift, self.gate - shift
            timed += posterior * first_arrival_shares(digit_time, selected_gate, *terms)
            if prompt:
                share, present = prompt_shares(digit_time, selected_gate, *terms, core_masks, self.gate)
                in_gate += posterior * share
                support += posterior * present
        if prompt:
            # The weighted probabilities can round just beyond the unit interval.
            np.clip(support, 0.0, 1.0, out=support)
        return timed, in_gate, support, total
