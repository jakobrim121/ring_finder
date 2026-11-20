"""Measured likelihood and morphology features for the existing count head."""

import numpy as np


def light_features(row, record, bundle):
    """Use comparable likelihood gains and the geometry of each count."""
    output = {"incoming_pion": float(row["particle"] == "pion")}
    if record is None:
        return output, {}
    nh = max(record["n_hits"], 1)
    output.update(hit_count=nh, charge_per_hit=record["raw_charge"] / nh)
    output["seed_class"] = record["initial_class"]
    logn = np.log(record["n_active"])
    fits = list(enumerate(record["trials"]))

    def charge(entry):
        t = entry[1]
        return 2.0 * t["charge"] - t["dimension"] * logn

    def joint(entry):
        t = entry[1]
        return 2.0 * t["joint"] - t["dimension"] * logn

    brightest = {}
    for count in (1, 2, 3):
        same_count = [r for r in fits if min(r[1]["count"], 3) == count]
        optical = [r for r in same_count if "geometry" in r[1]]
        physical = [r for r in same_count if "geometry" not in r[1] and "_soft" not in r[1]["label"]]
        if not optical and not physical:
            continue
        ic, selected = best_charge = max(optical + physical, key=charge)
        ij, _ = best_joint = max(optical + physical, key=joint)
        brightest[count] = (ic, ij)
        p = f"k{count}_"
        output[p + "q"] = charge(best_charge) / nh
        output[p + "qt"] = joint(best_joint) / nh
        output[p + "time"] = 2.0 * (selected["joint"] - selected["charge"]) / nh
        gain = [charge(t) for t in physical] if physical else []
        output[p + "physical"] = max(gain, default=np.nan) / nh
        for side, idx in (("charge", ic), ("joint", ij)):
            local = bundle[f"{idx}_joint_by_module"]
            output[p + side + "_module_q10"] = np.quantile(local, 0.1)
            output[p + side + "_module_max"] = local.max()
            output[p + side + "_module_shape"] = bundle[f"{idx}_module_shape"].sum() / nh
        rings = selected["neighborhoods"]
        for name in ("pmt_support", "lower_statistical", "leave_one_module", "weighted_charge", "amplitude"):
            v = sorted(n[name] for n in rings)
            if v:
                output[p + name] = v[0]
                output[p + name + "_relative"] = v[0] / max(sum(v), 1e-8)
        output[p + "background_response"] = selected["ring_amplitudes"][0] / max(
            sum(selected["ring_amplitudes"]), 1.0
        )
        if not optical:
            continue
        _, best_optical = max(optical, key=charge)
        geometry = best_optical["geometry"]
        axes, vertices = np.asarray(geometry["axes"]), np.asarray(geometry["vertices"])
        openings, widths = np.asarray(geometry["openings_deg"]), np.asarray(geometry["widths_deg"])
        weak = int(np.argmin([n["pmt_support"] for n in best_optical["neighborhoods"]]))
        output.update(
            {
                p + "forward_axis": max(axes[:, 2]),
                p + "weak_opening": openings[weak],
                p + "weak_width": widths[weak],
                p + "maximum_width": max(widths),
                p + "weak_vertex_z": vertices[weak, 2],
                p + "weak_vertex_r": np.linalg.norm(vertices[weak, :2]),
                p + "relative_clocks": np.ptp(best_optical["clocks"]),
                p + "angle_close": np.max((axes @ axes.T)[np.triu_indices(len(axes), 1)], initial=-1.0),
                p + "vertex_spread": np.linalg.norm(vertices - vertices.mean(0), axis=1).max(),
            }
        )
        onbeam = axes[:, 2] > 0.25
        if onbeam.any():
            flight = (-1348.763 - vertices[onbeam, 2]) / axes[onbeam, 2]
            aperture = vertices[onbeam, :2] + flight[:, None] * axes[onbeam, :2]
            output[p + "entry_distance"] = min(np.linalg.norm(aperture, axis=1))
        softer = [t for t in fits if t[1]["label"].startswith(best_optical["label"] + "_soft")]
        if softer:
            output[p + "soft_delta_q"] = max(map(charge, softer)) - charge((0, best_optical))
            output[p + "soft_delta_qt"] = max(map(joint, softer)) - joint((0, best_optical))
    for count in (2, 3):
        for name in ("q", "qt", "time", "physical", "joint_module_shape"):
            p, old = f"k{count}_{name}", f"k{count - 1}_{name}"
            output[f"step{count}_{name}"] = output.get(p, np.nan) - output.get(old, np.nan)
    return output, brightest
