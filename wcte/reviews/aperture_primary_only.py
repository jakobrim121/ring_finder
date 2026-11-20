"""Test replacing a single fitted partition with independently supported entry light."""

from functools import lru_cache

import numpy as np

from finite_chains import factory
from light_competition import Template


def holds_entry(original, prepared):
    """Keep a current ring whose fitted direction can account for the aperture light."""
    from staged_physics.primary import Aperture

    selected = original.get("selected") or {}
    parameters = selected.get("parameters")
    if parameters is not None and len(parameters) >= 12:
        point, axis = np.asarray(parameters[:3]), np.asarray(parameters[3:6])
    else:
        index = prepared.get("selected_fit_index")
        fits = prepared.get("fits", [])
        geometry = fits[index].get("geometry") if index is not None and 0 <= index < len(fits) else None
        if not geometry or len(geometry["axes"]) != 1:
            return False
        point = np.asarray(geometry["vertices"][0]) * 0.001
        axis = np.asarray(geometry["axes"][0])
    port = Aperture()
    if axis[2] < 0.65:
        return False
    depth = port.z - point[2]
    crossing = point[:2] + depth * axis[:2] / axis[2]
    # This is an incumbent preservation gate. Reserve room for curvature
    # over the unobserved extrapolation to the surveyed entry plane.
    return np.linalg.norm(crossing) <= port.radius + 0.085 + 0.08 * abs(depth)


@lru_cache(maxsize=2)
def absolute_entry_fields(particle):
    from beam_endpoint_grid import library
    from readout import parameters

    active, plans = library(particle)
    fields = (
        parameters()["light_coefficient_photons"] * np.asarray([row[1] for row in plans])
        if plans
        else np.empty((0, len(active)))
    )
    return active, fields, fields.sum(1)


def entry_evidence(ids, charges, shares, particle, *, residual_support=True):
    """Measure surveyed-entry overlap and independent unclaimed PMT support."""
    from readout import observation

    active, field, absolute = absolute_entry_fields(particle)
    if not len(field):
        return 0.0, 0.0
    ix = np.searchsorted(active, ids)
    np.testing.assert_array_equal(active[ix], ids)
    photons = field[:, ix]
    q = np.minimum(charges, 2.0)
    soft = photons / (photons + 0.42)
    amount = soft @ q
    total = charges.sum() / observation(charges).gain
    eligible = (amount > 22.0) & (absolute <= 1.45 * total)
    overlap = 2.0 * (np.minimum(soft, shares) @ q) / np.maximum(amount + q @ shares, 1.0e-12)
    if not eligible.any():
        return 0.0, 0.0
    covered = float(overlap[eligible].max(initial=0.0))
    if not residual_support:
        return covered, 0.0
    residual = soft[eligible] * (1.0 - shares)
    modules = np.unique(ids // 100, return_inverse=True)[1]
    by_module = np.array([np.bincount(modules, weights=r) for r in residual])
    distributed = residual.sum(1) - by_module.max(1)
    return covered, float(distributed.max(initial=0.0))


def measured_entry_overlap(ids, charges, shares, particle):
    return entry_evidence(ids, charges, shares, particle, residual_support=False)[0]


def compare(competition, trials, incumbent, original, prepared, *, independent_support=False):
    from physical_pulse import components

    if not trials or holds_entry(original, prepared):
        return []
    event, receiver = competition.event, competition.model
    q = np.minimum(incumbent["charges"], 2.0)
    prior = incumbent["fractions"][:, 1]
    if (
        independent_support
        and measured_entry_overlap(event.pmt_ids, event.charges_pe, prior, trials[0]["particles"][0]) >= 0.70
    ):
        return []
    options = []
    for t in trials:
        if t["count"] != 2 or t["neighborhoods"][0]["gate_pmt_support"] < 30.0:
            continue
        if independent_support and t["neighborhoods"][0]["gate_leave_one_module"] < 30.0:
            continue
        incoming = t["fractions"]["prompt"][:, 1]
        overlap = 2.0 * (q @ np.minimum(prior, incoming)) / max(q @ (prior + incoming), 1e-15)
        if overlap > 0.54:
            continue
        model = factory(event, receiver.bg, t["particles"], 2 * t["order"])
        state = model(t["parameters"], True)
        field, time = components(event, model, state, separate_knockon=True, all_receivers=True)
        amplitude = np.asarray(t["ring_amplitudes"])
        if field.shape[1] != len(amplitude) or len(amplitude) != 5:
            continue
        keep = [0, 2, 3, 4]
        shape = field[:, keep].T
        shape /= np.maximum(shape.sum(1)[:, None], 1e-300)
        template = Template(
            "aperture_omitted_entry",
            shape,
            time[:, [1, 2, 3]],
            amplitude[keep],
            ((2,),),
            np.zeros(3, int),
            t["dimension"] - 4,
            {},
        )
        control = competition.fit(template, iterations=23)
        if control is None:
            continue
        change = 2.0 * (0.65 * (t["charge"] - control["charge"]) + 0.35 * (t["joint"] - control["joint"]))
        change -= 4.0 * np.log(len(receiver.active))
        if change <= 6.6349:
            continue
        chosen = dict(t)
        chosen["label"] = "aperture_primary_only"
        chosen["count"] = 1
        chosen["neighborhoods"] = t["neighborhoods"][:1]
        chosen["fractions"] = {
            name: np.column_stack([1.0 - f[:, 1], f[:, 1]]) for name, f in t["fractions"].items()
        }
        chosen.update(
            entry_core_evidence=float(change),
            overlap_with_prior=float(overlap),
            daughter_prompt_support=t["neighborhoods"][1]["gate_pmt_support"],
        )
        options.append(chosen)
    return options
