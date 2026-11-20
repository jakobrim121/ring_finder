"""Use the saved winning charged history as the comparator for new light."""

from dataclasses import replace

import numpy as np

from light_competition import Template, optical_template, physical_template, short_lepton_templates


def restore_record(competition, history, prediction):
    from coupled_model import Model
    from timed_two_track import restore as repeat_fit

    if history.get("final_comparison") == "fine_chord":
        history = history.get("fine_chord", history)
    selected = (history.get("selected") or {}).get("backtrack")
    if not selected:
        return None
    count = prediction["fractions"].shape[1] - 1
    pool = [
        r
        for r in history.get("candidates", ())
        if r.get("family") == selected
        and r.get("eligible")
        and r.get("count") == count
        and r.get("detector_ll") is not None
    ]
    if not pool:
        return None
    previous = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        for record in sorted(pool, key=lambda r: -r["gain"]):
            _, state, fractions = repeat_fit(competition.event, competition.model.bg, record)
            if abs(state["ll"] - record["detector_ll"]) > 1.0e-5:
                continue
            if fractions.shape != prediction["fractions"].shape or not np.allclose(
                fractions, prediction["fractions"], rtol=0.0, atol=1.0e-10
            ):
                continue
            t = physical_template(competition.event, competition.model, record)
            if t is not None:
                return replace(t, label="restored_current", dimensions=record["dimension"])
    finally:
        Model.delta_scale = previous
    return None


def restore_original_physical(competition, original, prediction):
    """Use the selected charged fit when it explains both retained footprints."""
    from coupled_model import Model
    from shared_scoring import match_charge_shares
    from timed_two_track import restore as repeat_fit

    record = original.get("selected") or {}
    if (
        record.get("count") != 2
        or prediction["fractions"].shape[1] != 3
        or not all(record.get(k) for k in ("particles", "parameters", "order"))
    ):
        return None
    q = np.minimum(prediction["charges"], 2.0)
    settings = Model.delta_scale
    try:
        best = None
        for scale in dict.fromkeys((settings, 0.0)):
            Model.delta_scale = scale
            _, _, replay = repeat_fit(competition.event, competition.model.bg, record)
            if replay.shape[1] != 3:
                continue
            _, _, matched = match_charge_shares(prediction["fractions"], replay[:, 1:].T, q)
            weakest = float(np.min(matched, initial=1.0))
            if weakest >= 0.74 and (best is None or weakest > best[0]):
                source = physical_template(competition.event, competition.model, record)
                if source is not None:
                    best = weakest, source
        return replace(best[1], label="restored_original_physical") if best else None
    finally:
        Model.delta_scale = settings


def restore_energy_fit(competition, history):
    """Render the accepted connected energy refit with its own normalization."""
    from coupled_model import Model

    settings = Model.delta_scale
    Model.delta_scale = 0.0
    try:
        source = physical_template(
            competition.event, competition.model, dict(history, family="retained_energy_fit")
        )
    finally:
        Model.delta_scale = settings
    return replace(source, dimensions=history["dimension"]) if source is not None else None


def combine(source, rival, amplitude, *, counted):
    direct = rival.shapes[1:]
    seed = np.r_[source.amplitudes, amplitude]
    groups = source.core_groups
    start = len(source.amplitudes)
    if counted:
        groups = (*groups, (start,))
    clocks = np.r_[source.time_groups, source.time_groups.max() + 1]
    gains = np.r_[0, np.ones(start - 1, dtype=int), 2]
    limits = {2: rival.limits[1]} if 1 in rival.limits else {}
    return Template(
        "residual_restored_counted" if counted else "restored_current_submargin",
        np.vstack([source.shapes, direct]),
        np.column_stack([source.center, rival.center]),
        seed,
        groups,
        clocks,
        source.dimensions + (rival.dimensions - 1),
        limits,
        gains,
    )


def compare_on_history(competition, measured, base):
    if base is None:
        return []
    reference = competition.fit(base, iterations=26, fixed_ratios=True, physical_coverage=True)
    if reference is None:
        return []
    output = [reference]
    base = replace(base, amplitudes=reference["ring_amplitudes"])
    for trial in [t for t in measured if t.get("residual_path") is not None]:
        p = trial["residual_path"]
        amplitude = trial["ring_amplitudes"][-1]
        shape = dict(
            vertices=[1000.0 * p[:3]],
            axes=[p[3:6]],
            lengths_mm=[1000.0 * p[6]],
            openings_deg=[np.degrees(p[7])],
            widths_deg=[np.degrees(np.exp(p[8]))],
        )
        optic = optical_template(competition.model, shape, "restored_extra")
        if optic is None:
            continue
        test = combine(base, optic, amplitude, counted=True)
        extra = competition.fit(test, iterations=26, physical_coverage=True)
        if extra is not None:
            output.append(extra)
        optical = replace(optic, amplitudes=np.r_[base.amplitudes[0], amplitude])
        for soft in short_lepton_templates(competition.model, shape, optical, 1, flex_core=False):
            control = combine(base, soft, soft.amplitudes[1], counted=False)
            result = competition.fit(control, iterations=26, physical_coverage=True)
            if result is not None:
                output.append(result)
    return output
