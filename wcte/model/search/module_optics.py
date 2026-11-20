"""Check the shape of the light within each multi-PMT module."""

import numpy as np


def conditional(active, indices, charges, means):
    """Charge-weighted log score after profiling one intensity per module.

    The Poisson conditional model removes the unknown module brightness.
    All active receivers in each module remain in the normalization, so
    lighting a nominally unhit PMT still changes the score. The cap is a
    robust weight for analogue pulses and makes this a composite score.
    """
    modules = active // 100
    grouping = np.unique(modules, return_inverse=True)[1]
    mass = np.bincount(grouping, weights=means)
    group = grouping[indices]
    fraction = means[indices] / np.maximum(mass[group], 1e-300)
    w = np.minimum(charges, 2.0)
    score = w * np.log(np.maximum(fraction, 1e-300))
    return np.bincount(group, weights=score, minlength=len(mass))


class ConditionalSearch:
    """Guide an optical start using the pattern inside each detector module."""

    def __init__(self, observation, active, indices, fraction):
        self.observation = observation
        self.fraction = fraction
        self.modules = np.unique(active // 100, return_inverse=True)[1]
        self.weight = np.minimum(observation.q, 2.0)
        self.weight_by_module = np.bincount(
            self.modules[indices], weights=self.weight, minlength=int(self.modules.max()) + 1
        )

    def __getattr__(self, name):
        return getattr(self.observation, name)

    def score(self, mean, indices):
        value, gradient = self.observation.score(mean, indices)
        mass = np.maximum(np.bincount(self.modules, weights=mean), 1e-30)
        hit = np.maximum(mean[indices], 1e-30)
        group = self.modules[indices]
        contrast = self.weight @ (np.log(hit) - np.log(mass[group]))
        slope = self.weight_by_module[self.modules] / mass[self.modules]
        slope[indices] -= self.weight / hit
        return value + self.fraction * contrast, gradient + self.fraction * slope


def shapes(model, state):
    """Recover the exact complete PMT intensity used in the selected fit."""
    x = state["parameters"]
    model(x, True)
    return np.exp(x[-2]) * model.primary_cache[1] + np.exp(x[-1]) * model.bg


def compare(current, proposed, event, fraction):
    """Measure improvement on the modules that see the proposed second leg."""
    model, state = current
    rival, rival_state = proposed
    old = shapes(model, state)
    new = shapes(rival, rival_state)
    active, indices = model.active, model.ix
    q = event.charges_pe
    score = conditional(active, indices, q, new) - conditional(active, indices, q, old)
    grouping = np.unique(active // 100, return_inverse=True)[1]
    added = np.bincount(grouping[indices], weights=np.minimum(q, 2.0) * fraction, minlength=len(score))
    enough = added >= 2.0
    most = int(np.argmax(added))
    return dict(
        delta=float(score.sum()),
        extra_modules=int(np.count_nonzero(enough)),
        extra_module_delta=float(score[enough].sum()),
        extra_module_positive=int(np.count_nonzero(enough & (score > 0.0))),
        extra_module_delta_without_brightest=float(score[enough].sum() - score[most])
        if enough[most]
        else 0.0,
    )
