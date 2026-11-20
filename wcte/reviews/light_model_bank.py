"""Profile the compact measured hypotheses used by the multiplicity review."""

from dataclasses import replace

import numpy as np

from light_competition import optical_template, physical_template, short_lepton_templates


def measured_bank(
    choice,
    prepared,
    bank,
    original,
    count,
    *,
    deep=False,
    compare_count=None,
    physical_coverage=False,
    optical_starts=1,
):
    from optical_chain_search import geometries

    event, model = choice.event, choice.model
    trials, optics, results, messages = [], [], {}, []
    for k in range(1, min(4, max(3, count)) + 1):
        starts = geometries(prepared, bank, k)
        if len(starts) > 2 and not deep:
            starts = [starts[0], starts[-1]]
        qualified = []
        for i, geometry in enumerate(starts):
            template = optical_template(model, geometry, f"optical_{k}_{i}")
            if template is None:
                continue
            # All optical counts still seed the beam history. Their fit
            # scores and amplitudes suffice when only comparing a count.
            fitted = choice.fit(
                template,
                score_only=compare_count is not None,
                physical_coverage=physical_coverage,
            )
            if fitted is None:
                continue
            fitted["geometry"] = {
                n: geometry[n]
                for n in ("vertices", "axes", "openings_deg", "widths_deg", "lengths_mm")
                if n in geometry
            }
            trials.append(fitted)
            qualified.append((fitted, geometry, template))
        if qualified:
            qualified.sort(key=lambda q: q[0]["joint"], reverse=True)
            results[k] = qualified[0][0]
            optics.extend(
                (r["joint"], g, replace(t, amplitudes=r["ring_amplitudes"]))
                for r, g, t in qualified[:optical_starts]
            )
    records = [
        r
        for r in original.get("candidates", ())
        if r.get("supported") and r.get("particles") and r.get("parameters") and r.get("order")
    ]
    if (original.get("selected") or {}).get("particles"):
        records.append(original["selected"])
    tested, by_count = set(), {}
    for record in sorted(records, key=lambda r: -r.get("gain", -np.inf)):
        k = record["count"]
        if compare_count is not None and k != compare_count:
            continue
        signature = tuple(record["particles"]), tuple(record["parameters"])
        if signature in tested or by_count.get(k, 0) >= (2 if deep else 1):
            continue
        tested.add(signature)
        try:
            template = physical_template(event, model, record)
            if template is None:
                continue
            fitted = choice.fit(
                template,
                fixed_ratios=True,
                score_only=compare_count is not None,
                physical_coverage=physical_coverage,
            )
            if fitted is not None:
                fitted["legacy_gain"] = record["gain"]
                trials.append(fitted)
                by_count[k] = by_count.get(k, 0) + 1
        except (ArithmeticError, ValueError, KeyError) as error:
            messages.append(dict(family=record["family"], reason=str(error)))
    for _, geometry, template in optics:
        k = len(template.core_groups)
        if k < 2 or (compare_count is not None and k not in (compare_count, compare_count + 1)):
            continue
        for ring in range(1, k + 1):
            flex_core = deep and (compare_count is None or k == compare_count)
            for below in short_lepton_templates(model, geometry, template, ring, flex_core=flex_core):
                if compare_count is not None and len(below.core_groups) != compare_count:
                    continue
                fitted = choice.fit(
                    below,
                    score_only=compare_count is not None,
                    physical_coverage=physical_coverage,
                )
                if fitted is not None:
                    trials.append(fitted)
    return trials, optics, results, messages
