"""Test whether separate fitted ribbons follow one measured optical trajectory."""

import numpy as np


def ray_coordinates(positions, vertex, axis, opening, length):
    from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE

    radius = positions - vertex
    parallel = radius @ axis
    normal = radius - parallel[:, None] * axis
    distance = np.linalg.norm(normal, axis=1)
    theta = np.deg2rad(np.clip(opening, 10.0, 42.0))
    emission = parallel - distance / np.tan(theta)
    actual_length = max(1.0, length or 0.0)
    u = np.clip(emission, -0.5 * actual_length, 0.5 * actual_length)
    travel = np.hypot(parallel - u, distance)
    beta_inverse = N_PHASE * np.cos(theta)
    delay = (u * beta_inverse + N_GROUP * travel) / C
    angle_error = np.abs(np.arctan2(distance, parallel - u) - theta)
    return emission, delay, normal, angle_error


def group_moments(mass, location, radial, delay, times, angular_error, assigned, scale):
    from charge_starts import weighted_quantile

    total = float(mass.sum())
    if total < 5.0:
        return None
    local = mass * np.exp(-0.5 * (angular_error / scale) ** 2)
    weight = local.sum()
    if weight < 3.0:
        return None
    emit = weighted_quantile(location, local, [0.10, 0.50, 0.90])
    time = weighted_quantile(times - delay, local, [0.10, 0.50, 0.90])
    origin = (local @ radial) / max(weight, 1.0e-12)
    return emit, time, origin, float(weight / total), float(mass @ assigned / total)


def describe_times(observed, receipt, book, prediction):
    from wctering.likelihood.geometry import active_geometry

    current = prediction["fractions"][:, 1:]
    if current.shape[1] not in (2, 3) or receipt is None:
        return {}
    geometry = [
        (i, t)
        for i, t in enumerate(receipt["trials"])
        if "geometry" in t and t["count"] == current.shape[1] - 1
    ]
    if not geometry:
        return {}
    i, trial = max(geometry, key=lambda r: 0.65 * r[1]["charge"] + 0.35 * r[1]["joint"])
    lower = book[f"{i}_timed"][:, 1:]
    q = np.minimum(prediction["charges"], 2.0)
    ids, position, _ = active_geometry()
    use = np.searchsorted(ids, prediction["pmt_ids"])
    np.testing.assert_array_equal(ids[use], prediction["pmt_ids"])
    hit_time = observed - np.median(observed)
    pos = position[use]
    g = trial["geometry"]
    proposals = []
    lengths = g.get("lengths_mm", [0.0] * len(lower.T))
    for index, (vertex, axis, theta, width, length) in enumerate(
        zip(g["vertices"], g["axes"], g["openings_deg"], g["widths_deg"], lengths, strict=True),
    ):
        loc, flight, radial, tolerance = ray_coordinates(
            pos, np.asarray(vertex), np.asarray(axis), theta, length
        )
        d = np.linalg.norm(radial, axis=1)
        radial = radial / np.maximum(d[:, None], 1.0)
        profiles = [
            group_moments(
                q * f,
                loc,
                radial,
                flight,
                hit_time,
                tolerance,
                lower[:, index],
                np.deg2rad(np.clip(width, 1.8, 7.0)),
            )
            for f in current.T
        ]
        for b, right in enumerate(profiles):
            if right is None:
                continue
            for left in profiles[:b]:
                if left is None:
                    continue
                lz, lt, lr, lnear, lcover = left
                rz, rt, rr, rnear, rcover = right
                shared = min(lcover, rcover)
                proposals.append(
                    (
                        shared,
                        min(lnear, rnear),
                        (length or 0.0) / 1000.0,
                        abs(lz[1] - rz[1]) / max(0.5 * (np.ptp(lz) + np.ptp(rz)), 60.0),
                        abs(lt[1] - rt[1]) / max(0.5 * (np.ptp(lt) + np.ptp(rt)), 1.0),
                        np.linalg.norm(lr - rr),
                        axis[2],
                    )
                )
    if not proposals:
        return {}
    result = max(proposals, key=lambda v: v[0] + 0.5 * v[1])
    fields = (
        "same_path_charge",
        "same_path_optical_support",
        "same_path_length_m",
        "emission_depth_separation",
        "emission_time_separation",
        "azimuth_separation",
        "same_path_axis_z",
    )
    return dict(zip(fields, result, strict=True))
