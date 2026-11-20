"""Qualify an existing two-track fit when both arcs illuminate the same PMTs."""

import numpy as np

from bare_model import Model as Bare
from chain_comparison import MARGIN, counted
from chain_sources import paths
from coupled_model import Model as Coupled
from physical_pulse import profile
from ring_supported import preserves_prompt
from source import trajectory
from terminal_track_seeds import pulse_support


def restore(event, background, record):
    """Re-evaluate one already fitted hypothesis at its checked optical order."""
    particles = record["particles"]
    factory = Bare if all(p == "electron" for p in particles) else Coupled
    model = factory(event.pmt_ids, event.charges_pe, background, particles, 4, 2 * record["order"])
    state = model(np.asarray(record["parameters"]), True)
    return model, state, counted(model, state)


def charge_gap(event, model, state, records):
    """Compare against the best detector-only one-ring endpoints."""
    pulse = profile(event, model, state)
    if pulse is None:
        return None
    tested = []
    for record in records:
        rival, previous, _ = restore(event, model.bg, record)
        time = profile(event, rival, previous)
        if time is not None:
            tested.append(previous["ll"] + time["timing_ll"])
    if not tested:
        return None
    return 2.0 * (state["ll"] + pulse["timing_ll"] - max(tested))


def accept(event, single, alternatives, baseline_gain, *, extra_controls=()):
    """Require a resolved shared-clock pair and reject competing sub-count light."""
    one, old = single
    prior = one, old, counted(one, old)
    if prior[2].shape[1] != 2:
        return None
    ordered = sorted(
        (r for r in alternatives if r.get("particles") and "order" in r),
        key=lambda r: r["gain"],
        reverse=True,
    )
    controls = sorted(
        [r for r in ordered if r["count"] == 1][:3] + list(extra_controls),
        key=lambda r: r["gain"],
        reverse=True,
    )[:3]
    candidates = [
        r
        for r in ordered
        if r["count"] == 2 and len(r["particles"]) == 2 and r["gain"] > baseline_gain + MARGIN
    ][:2]
    for record in candidates:
        model, state, fractions = restore(event, one.bg, record)
        if fractions.shape[1] != 3:
            continue
        # This test targets a resolved bend between two finite charged
        # tracks. It does not promote a misfit from a short diffuse lobe.
        a, b = [p for p, _ in paths(state["parameters"][:-2])]
        if min(a[9] * a[10], b[9] * b[10]) < 0.09 or "electron" in model.particles:
            continue
        tangent = trajectory(a, np.ones(1))[1][0]
        turn = np.arccos(np.clip(tangent @ b[3:6], -1.0, 1.0))
        widths = np.hypot(np.exp(a[-1]), np.exp(b[-1]))
        if turn <= np.deg2rad(15.0) + 2.0 * widths:
            continue
        support = pulse_support(event, model, state, fractions)
        if support is None or min(s["lower_statistical"] for s in support) < 30.0:
            continue
        retained, lost = preserves_prompt(event, prior, (model, state, fractions))
        if not retained:
            continue
        joint = charge_gap(event, model, state, controls)
        if joint is None or joint <= 2.0 * MARGIN:
            continue
        from branched_recoil import fit as branch
        from lineage_competitor import fit as lineage
        from visibility_profile import fit as below

        weak = int(np.argmin([r["pmt_support"] for r in support]))
        alternate = lineage(event, one, model, state, 44)
        if alternate is None or 2.0 * (state["ll"] - alternate["ll"]) <= MARGIN:
            continue
        visibility = recoil = None
        if support[weak]["pmt_support"] < 65.0:
            visibility = below(model, state, weak, record["order"], 32)
            if visibility is not None and 2.0 * (state["ll"] - visibility["ll"]) <= MARGIN:
                continue
            recoil = branch(event, one, old, model, state, fractions, weak)
            if recoil is not None and 2.0 * (state["ll"] - recoil["ll"]) <= MARGIN:
                continue
        return (
            record["gain"],
            fractions,
            dict(
                record,
                family="timed_two_track",
                common_clock_score_gap=joint,
                conditional_support=support,
                lost_prompt_hits=lost,
                turn_deg=float(np.rad2deg(turn)),
                same_lineage=alternate,
                below_support=visibility,
                subcount_recoil=recoil,
            ),
        )
    return None


def grouped_pulse(event, model, state, groups, *, spread=False):
    """Condition PMT occupancy on the first arrival and all physical light."""
    from module_supported_ring import lower_bound
    from physical_pulse import components
    from resolved_ring_evidence import first_pulse_support

    light = components(event, model, state, separate_knockon=True, spread=spread)
    rates, arrivals, *extra = light
    variance = extra[0] if extra else None
    timing = profile(event, model, state, precomputed=light)
    if timing is None:
        return None
    center = np.median(event.times_ns)
    time = event.times_ns - center
    gate = np.asarray(event.timing_selection["prompt_window_ns"]) - center
    sigma = np.sqrt(0.8**2 + (1.5**2 + 2.1**2 / np.maximum(event.charges_pe, 0.25)) / 2.355**2)
    arrivals = arrivals + timing["clock_ns"]
    # Every physical section supplies one above-margin core and one
    # slowing tail. Leave the tails and knock-on light in the comparison.
    support = []
    for group in groups:
        mask = sum(1 << (2 * j) for j in group)
        probability = first_pulse_support(
            time,
            gate,
            rates,
            arrivals,
            sigma,
            -1,
            model.obs.offsets,
            model.obs.coefficients,
            excluded_mask=mask,
            variance=variance,
        )
        support.append(lower_bound(event.pmt_ids, probability))
    return timing, support
