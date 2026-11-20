"""Multiplicity evidence built only from detector-fitted hypotheses.

No source, event number, primary PID, energy, truth, old classifier score,
old decision, absolute coordinate, or runtime is accepted as a feature.
Component summaries are invariant under permutations of ring labels.
"""

from collections import defaultdict
import numpy as np

_EXCLUDE = (
    "runtime",
    "threshold",
    "minimum_required",
    "decision",
    "prediction",
    "model",
    "schema",
    "truth",
    "particle",
    "source",
    "event_id",
    "pdg",
    "track_id",
    "metadata",
    "display",
    "vertex_containment",
    "partition_proposal_diagnostics",
    "partition_labels",
    "direction",
    "vertex_mm",
    "vertex",
    "weak_axis",
    "weak_axes",
    "matrix",
    "matrices",
    "covariance_axis_pairwise",
)
_LEAF_EXCLUDE = {
    "valid",
    "t0",
    "t0_ns",
    "ring_label",
    "expected_ring_count",
    "returned_n_rings",
    "group_index",
    "core_band_deg",
    "broad_band_deg",
    "module_neighbor_distance_mm",
    "module_time_residual_tolerance_ns",
    "n_partitions",
}


def _scalar_summary(value, prefix, out):
    buckets = defaultdict(list)

    def walk(v, path):
        if isinstance(v, dict):
            for k, x in v.items():
                if k in _LEAF_EXCLUDE or any(w in k for w in _EXCLUDE):
                    continue
                walk(x, path + "." + ("component" if str(k).isdigit() else k))
        elif isinstance(v, (list, tuple, np.ndarray)):
            if len(v) > 12:
                return
            for i, x in enumerate(v):
                # Eigenvalue index is physically ordered; ring component index is not.
                walk(x, path + (f".eig{i}" if "eigenvalues" in path else ".component"))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
            if np.isfinite(v):
                buckets[path].append(float(v))

    walk(value, prefix)
    for path, vals in buckets.items():
        if ".component" in path:
            for name, f in [("min", np.min), ("mean", np.mean), ("max", np.max)]:
                out[path + "." + name] = float(f(vals))
        else:
            out[path] = float(vals[0])


def _unit(v):
    x = np.asarray(v, float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def geometry_features(full, prefix, out):
    rings = full.get("partition_rings", [])
    if not rings:
        return
    for key in ["theta_c", "sigma_t"]:
        vals = [float(r[key]) for r in rings if key in r]
        if vals:
            for name, fn in [("min", min), ("max", max), ("mean", np.mean)]:
                out[f"{prefix}.cone_{key}.{name}"] = float(fn(vals))
    pairs = defaultdict(list)
    fits = full.get("h2_fits", full.get("h3_fits", []))
    for i in range(len(rings)):
        for j in range(i):
            d1 = _unit(rings[i]["direction"])
            d2 = _unit(rings[j]["direction"])
            pairs["axis_angle_deg"].append(float(np.degrees(np.arccos(np.clip(d1 @ d2, -1, 1)))))
            pairs["opening_difference_deg"].append(
                abs(float(rings[i]["theta_c"]) - float(rings[j]["theta_c"]))
            )
            if len(fits) != len(rings):
                continue
            dx = np.asarray(fits[i]["vertex_mm"]) - np.asarray(fits[j]["vertex_mm"])
            length = float(np.linalg.norm(dx))
            chord = _unit(dx)
            cos1 = float(d1 @ chord)
            cos2 = float(d2 @ chord)
            dt = float(fits[i]["t0_ns"]) - float(fits[j]["t0_ns"])
            pairs["vertex_separation_mm"].append(length)
            pairs["chord_axis_product"].append(cos1 * cos2)
            pairs["chord_axis_abs_min"].append(min(abs(cos1), abs(cos2)))
            pairs["clock_abs_difference_ns"].append(abs(dt))
            pairs["causal_clock_abs_residual_ns"].append(abs(abs(dt) - length / 299.792458))
            pairs["opening_order_product"].append(
                (float(rings[i]["theta_c"]) - float(rings[j]["theta_c"])) * (cos1 + cos2) / 2
            )
    for k, vals in pairs.items():
        for name, fn in [("min", min), ("max", max), ("mean", np.mean)]:
            out[f"{prefix}.{k}.{name}"] = float(fn(vals))
    h1 = full.get("h1_partition_ring")
    if h1:
        out[prefix + ".h1_opening_deg"] = float(h1["theta_c"])
        out[prefix + ".h1_time_width_ns"] = float(h1["sigma_t"])
    one = full.get("h1_fit", {})
    h2 = full.get("h2_fits", [])
    if h2 and "chi2" in one:
        out[prefix + ".point_timing_chi2_gain"] = float(one["chi2"]) - sum(float(f["chi2"]) for f in h2)
        out[prefix + ".point_timing_chi2_gain_per_hit"] = out[prefix + ".point_timing_chi2_gain"] / max(
            1, float(one.get("n_hits", 1))
        )


def evidence_features(diagnostics, extra_h3=None, extra_physics=None):
    d = diagnostics
    out = {}
    for k in [
        "usable_hits",
        "usable_unique_pmts",
        "usable_charge_pe",
        "r_full",
        "r_robust_q25",
        "decision_score_s_new",
    ]:
        if isinstance(d.get(k), (int, float)) and np.isfinite(d[k]):
            out[k] = float(d[k])
    full = d.get("full_event_hypotheses") or {}
    _scalar_summary(full, "h12", out)
    geometry_features(full, "h12.geometry", out)
    three = (
        extra_h3
        if extra_h3 is not None
        else (d.get("three_ring_hypothesis_test") or {}).get("full_event_hypotheses")
    )
    if three:
        _scalar_summary(three, "h3", out)
        geometry_features(three, "h3.geometry", out)
    fold_values = defaultdict(list)
    for fold in d.get("module_fold_refits", []):
        h = fold.get("hypotheses", {})
        f = {}
        _scalar_summary(h, "h12", f)
        geometry_features(h, "h12.geometry", f)
        for k, v in f.items():
            fold_values[k].append(v)
    for k, vals in fold_values.items():
        if k not in out:
            continue
        if any(
            x in k
            for x in [
                "fraction",
                "incompatibility",
                "angle",
                "reduced_chi2",
                "opening",
                "time_width",
            ]
        ):
            out["fold_spread." + k] = float(np.std(vals))
            out["fold_shift." + k] = float(np.mean(vals)) - out[k]
    charge = out.get("usable_charge_pe", 1)
    n = out.get("usable_unique_pmts", 1)
    out["mean_charge_per_pmt"] = charge / max(1, n)
    # Relative light and support provide scale-independent significance alongside counts.
    for k, v in list(out.items()):
        if k.startswith("h12.") or k.startswith("h3."):
            if k.endswith(("charge_pe", "charge_pe.min", "charge_pe.max", "charge_pe.mean")):
                out["relative." + k] = v / max(charge, 1)
            if k.endswith(
                (
                    "unique_pmts",
                    "unique_pmts.min",
                    "unique_pmts.max",
                    "unique_pmts.mean",
                )
            ):
                out["relative." + k] = v / max(n, 1)
    if extra_physics:
        out.update(extra_physics)
    return out
