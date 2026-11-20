"""Test visible residual wavefronts beyond the original fitter's count limit."""

from dataclasses import replace
from itertools import islice

import numpy as np

from light_competition import optical_template, short_lepton_templates


def geometry_with_path(geometry, path, amplitudes):
    """Add one finite Cherenkov path without changing the existing sources."""
    k = len(geometry["vertices"])
    return dict(
        vertices=np.vstack([geometry["vertices"], 1000.0 * path[:3]]),
        axes=np.vstack([geometry["axes"], path[3:6]]),
        openings_deg=np.r_[geometry["openings_deg"], np.degrees(path[7])],
        widths_deg=np.r_[geometry["widths_deg"], np.degrees(np.exp(path[8]))],
        lengths_mm=[*geometry.get("lengths_mm", [0.0] * k), 1000.0 * path[6]],
        weights=amplitudes / amplitudes.sum(),
    )


def candidates(event, model, bank, geometry, template, *, thorough=False):
    from analog_optical import contain_finite
    from charge_starts import allowed_chord
    from sparse_paths import screen

    if thorough:
        from timed_small_circles import starts

        for path, pe in starts(
            event, model, bank, geometry, template.shapes, template.amplitudes, marked_residual=True
        ):
            yield path, pe, "small_circle"
    mean = template.amplitudes @ template.shapes
    ll, slope = model.obs.score(mean, model.ix)
    for _, origin, axis, opening, pe, lo, hi, _, _ in screen(
        event, bank, geometry, mean, slope, template.shapes, model.ix, residual_vertices=True
    ):
        chord = allowed_chord(origin, axis)
        if chord is None:
            continue
        lo, hi = max(chord[0], lo - 0.03), min(chord[1], hi + 0.03)
        if hi - lo <= 0.015:
            continue
        p = np.r_[origin + 0.5 * (hi + lo) * axis, axis, hi - lo, opening, np.log(np.deg2rad(3.0))]
        try:
            contain_finite(p)
        except ValueError:
            continue
        yield p, pe, "axis_grid"


def propose(competition, bank, geometry, template, control, *, thorough=False, steps=2):
    """Attach coherent residual sources in increments, reprising old amplitudes."""
    from chain_comparison import MARGIN
    from finite_residual_fit import local_fit, optical_shape

    event, model = competition.event, competition.model
    trials = []
    for _ in range(steps):
        mean = template.amplitudes @ template.shapes
        proposals = []
        k = len(template.core_groups) + 1
        for j, (path, pe, family) in enumerate(
            islice(
                candidates(event, model, bank, geometry, template, thorough=thorough), 5 if thorough else 3
            )
        ):
            strength = np.clip(pe, 3.0, 0.45 * template.amplitudes.sum())
            result = local_fit(model, path, strength, mean, 26)
            if result is None:
                continue
            p, strength, _ = result
            low, high = (optical_shape(model, p, order)[0] for order in (16, 32))
            norm_error = np.sum(abs(low - high))
            if not np.isfinite(norm_error) or norm_error > 0.012:
                continue
            g = geometry_with_path(geometry, p, np.r_[template.amplitudes, strength])
            alternative = optical_template(model, g, f"residual_{k}_{j}_{family}")
            if alternative is None:
                continue
            alternatives = alternative.shapes.copy()
            alternatives[-1] = high
            alternative = replace(
                alternative, shapes=alternatives, amplitudes=np.r_[template.amplitudes, strength]
            )
            summary = competition.fit(alternative, iterations=26)
            if summary is None:
                continue
            delta = summary["joint_by_module"] - control["joint_by_module"]
            penalty = (alternative.dimensions - template.dimensions) * np.log(len(model.active))
            robust_gain = 2.0 * (delta.sum() - delta.max(initial=0.0)) - penalty
            q = competition.q
            current = control["fractions"]["timed"][:, 1:]
            added = summary["fractions"]["timed"][:, 1:]
            retention = (q @ np.minimum(current, added[:, :-1])) / np.maximum(q @ current, 1e-12)
            surplus = q * np.maximum(added[:, -1] - current.max(1), 0.0)
            module = np.bincount(competition.modules[model.ix], weights=surplus)
            active_evidence = surplus.sum() - module.max(initial=0.0)
            extra = summary["neighborhoods"][-1]
            summary.update(
                residual_path=p,
                robust_gain=float(robust_gain),
                retained=retention,
                independent_pe=float(active_evidence),
                quadrature_shape_error=float(norm_error),
                selected_step=False,
            )
            trials.append(summary)
            novel = (
                extra["pmt_support"] >= 30.0
                and extra["lower_statistical"] >= 22.0
                and extra["leave_one_module"] >= 22.0
                and active_evidence >= 12.0
                and np.min(retention, initial=1.0) >= 0.65
            )
            if robust_gain <= MARGIN or not novel:
                continue
            uncounted = next(
                short_lepton_templates(
                    model, g, replace(alternative, amplitudes=summary["ring_amplitudes"]), k, flex_core=False
                ),
                None,
            )
            if uncounted is not None:
                rival = competition.fit(uncounted, iterations=20)
                if rival is not None:
                    trials.append(rival)
                    cost = (alternative.dimensions - uncounted.dimensions) * np.log(len(model.active))
                    if 2.0 * (summary["joint"] - rival["joint"]) <= cost + MARGIN:
                        continue
            proposals.append((robust_gain, summary, g, alternative))
        if not proposals:
            break
        _, control, geometry, added = max(proposals, key=lambda r: r[0])
        control["selected_step"] = True
        template = replace(added, amplitudes=control["ring_amplitudes"])
    return trials
