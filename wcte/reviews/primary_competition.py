"""Compete the measured beam-entry core against independently born light."""

from itertools import combinations

import numpy as np

from beam_prefix import fit_prefix
from light_competition import Template


def candidates(competition, particle, optics, incumbent_count, *, second_start=False, reference_starts=1):
    """Fit a qualified primary beside alternatives that may contain fewer rings."""
    from physical_pulse import components

    event = competition.event
    # Fit the primary against all of the strongest available optical light.
    # A second start can leave out the optical lobe whose projected axis is
    # most beam-like, so an already occupied light ridge can move ownership.
    choices = [(g, t) for _, g, t in optics if len(t.core_groups) == incumbent_count]
    if not choices:
        choices = [(g, t) for _, g, t in optics if len(t.core_groups) == 2]
    if not choices:
        return [], []
    means = []
    for geometry, ref in choices[:reference_starts]:
        masks = [np.ones(len(ref.shapes), dtype=bool)]
        if second_start:
            closest = int(np.argmax(np.asarray(geometry["axes"])[:, 2])) + 1
            exclude = masks[0].copy()
            exclude[closest] = False
            masks.append(exclude)
        means.extend(ref.amplitudes[mask] @ ref.shapes[mask] for mask in masks)
    trials, checked, fingerprints = [], [], []
    for i, local in enumerate(means):
        found, history = fit_prefix(event, particle, local)
        checked.append(history)
        if found is None:
            continue
        beam, state = found
        x = state["parameters"]
        if any(np.linalg.norm(x[:-2] - p) < 0.045 for p in fingerprints):
            continue
        fingerprints.append(x[:-2])
        rates, centers = components(event, beam, state, separate_knockon=True, all_receivers=True)
        charge = rates[:, 1:]
        if charge[:, 0].sum() < 6.0:
            continue
        terms = charge.sum(0)
        shapes = charge.T / np.maximum(terms[:, None], 1e-300)
        # The modeled scattering/stopping intensity is uncounted, even if
        # charge-only fitting prefers that light to a direct source.
        optical = [t for _, _, t in optics if len(t.core_groups) in {1, 2, 3, 4}]
        proposed = list(templates(optical, incumbent_count, i, beam.dimension - 1, shapes, terms, centers))
        for trial in proposed:
            if (r := competition.fit(trial, iterations=24)) is not None:
                r["beam_mode"] = x.tolist()
                r["beam_core"] = r["count"] - 1
                trials.append(r)
    return trials, checked


def templates(optics, incumbent_count, mode, dimension, beam_shapes, beam_terms, beam_times):
    for template in optics:
        n = len(template.core_groups)
        if n < max(1, incumbent_count - 1) or n > incumbent_count:
            continue
        # Same-count replacements and a genuine extra ring share the same
        # measured likelihood. The measured data decide if the candidate
        # contributes distinct visible light.
        for dropped in [None] + list(combinations(range(1, n + 1), 1)):
            if dropped is None:
                if n > 3:
                    continue
                indices = np.arange(n + 1)
                tag = "add"
            else:
                indices = np.delete(np.arange(n + 1), dropped[0])
                tag = f"replace{dropped[0]}"
            retained = len(indices) - 1
            if retained == 0 and incumbent_count > 2:
                continue
            basis = np.vstack([template.shapes[indices], beam_shapes])
            times = np.column_stack([template.center[:, indices[1:] - 1], beam_times])
            a = np.r_[template.amplitudes[indices], beam_terms]
            groups = np.r_[np.arange(retained), retained, retained]
            intensity = np.r_[np.arange(retained + 1), retained + 1, retained + 1]
            dims = 1 + round((template.dimensions - 1) * retained / n) + dimension
            cap = 1.22 * beam_terms.sum()
            yield Template(
                f"beam_{mode}_{n}_{tag}",
                basis,
                times,
                a,
                tuple((k,) for k in range(1, retained + 2)),
                groups,
                dims,
                {retained + 1: cap},
                intensity,
            )
