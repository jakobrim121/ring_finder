"""Evaluate the selected first pulse with calibrated high-charge censoring."""

from functools import lru_cache

from occupancy_pulse import build_likelihood


def detected_tail(cutoff):
    from censored_charge import coefficients

    return coefficients(cutoff, 384)[1]


@lru_cache(maxsize=1)
def kernel():
    return build_likelihood(censored=True)


def create(event, model, cutoff):
    from selected_pulse import Selection

    class Censored(Selection):
        def score(self, rates, times):
            return kernel()(
                self.hit,
                self.time,
                *self.gate,
                times,
                rates,
                self.width,
                self.obs.r,
                self.obs.p2,
                self.obs.q,
                cutoff,
                self.obs.offsets,
                self.obs.coefficients,
                detected_tail(cutoff),
            )

    return Censored(event, model)
