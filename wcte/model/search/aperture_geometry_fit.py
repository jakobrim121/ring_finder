"""Fit the incoming beam wavefront to an observed downstream light interval."""

from itertools import product

import numpy as np

from analog_optical import contain_finite
from aperture_pair_review import pairs
from charge_starts import weighted_quantile
from chord_profile import profile
from entrance_track import Aperture, axes as entrance_axes
from finite_residual_fit import optical_shape
from low_angle_continuation import coordinates
from residual_joint_optics import path_from_geometry
from wctering.likelihood.geometry import UPPER, NORMALS
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


def ray_times(positions, point, axis, angle):
    """Project each PMT to its emission point and compute its flight time."""
    r = positions - point
    z = r @ axis
    rho = np.linalg.norm(r - z[:, None] * axis, axis=1)
    emission = z - rho / np.tan(angle)
    width = np.maximum(0.020, np.deg2rad(2.2) * rho / np.sin(angle) ** 2)
    flight = 1000.0 * (N_PHASE * np.cos(angle) * emission + N_GROUP * rho / np.sin(angle)) / C
    return emission, width, flight


def ray_vote(rays, t, weights, clock, stop):
    """Weight each pulse by emission position and causally predicted time."""
    emission, width, flight = rays
    error = emission - np.clip(emission, 0.0, stop)
    membership = weights * np.exp(-0.5 * (error / width) ** 2)
    if clock is None:
        if membership.sum() <= 1e-8:
            return 0.0, 0.0, membership
        clock = weighted_quantile(t - flight, membership, [0.5])[0]
    membership *= np.exp(-0.5 * ((t - flight - clock) / 1.7) ** 2)
    return float(membership.sum()), clock, membership


def starts(event, model, geometry, amplitude, optical, limit=2):
    """Search the junction using downstream direction and the survey plane."""
    if event.timing_selection["prompt_window_ns"] is None:
        return []
    port = Aperture()
    origin = np.array([0.0, 0.0, port.z])
    q = np.minimum(event.charges_pe, 2.0)
    pos = event.pmt_positions_mm / 1000.0
    t = event.times_ns - np.median(event.times_ns)
    projections = {}
    rough = []
    for g, amp in pairs(geometry, amplitude, optical):
        try:
            measured = [path_from_geometry(g, j) for j in range(2)]
        except (ValueError, ArithmeticError):
            continue
        for h in (0, 1):
            target = measured[h]
            a = target[3:6]
            for direction in entrance_axes(g, port)[:3]:
                if direction @ a > np.cos(np.deg2rad(16.0)):
                    continue
                # Closest points of rays from the aperture and the fitted
                # downstream center. The birth itself can move along the
                # downstream ray within the resolution of its length.
                x = np.column_stack([direction, -a])
                if np.linalg.norm(np.cross(direction, a)) <= 1e-5:
                    continue
                at, err = np.linalg.lstsq(x, target[:3] - origin, rcond=None)[:2]
                if len(err) and err[0] > 0.45**2:
                    continue
                guessed = at[0]
                for length, incoming_angle in product(
                    np.clip(guessed + np.array([-0.20, -0.08, 0.0, 0.08, 0.20]), 0.035, 1.85),
                    np.deg2rad([29.0, 34.0, 38.0, 41.0]),
                ):
                    join = origin + length * direction
                    extent = np.clip(0.5 * target[6] + (target[:3] - join) @ a + 0.05, 0.045, 1.4)
                    if (UPPER - NORMALS @ join).min() <= 0.0:
                        continue
                    projection = (*direction, incoming_angle)
                    if projection not in projections:
                        projections[projection] = ray_times(pos, origin, direction, incoming_angle)
                    score, clock, membership = ray_vote(projections[projection], t, q, None, length)
                    if score < 28.0:
                        continue
                    delay = 1000.0 * N_PHASE * np.cos(incoming_angle) * length / C
                    opening = np.clip(target[7], np.deg2rad(12.0), np.deg2rad(41.8))
                    rays = ray_times(pos, join, a, opening)
                    residual = q - 0.8 * membership
                    done, _, _ = ray_vote(rays, t, residual, clock + delay, extent)
                    if done < 21.0:
                        continue
                    parent = np.r_[
                        origin + 0.5 * length * direction,
                        direction,
                        length,
                        incoming_angle,
                        np.log(np.deg2rad(2.0)),
                    ]
                    daughter = np.r_[
                        join + 0.5 * extent * a,
                        a,
                        extent,
                        opening,
                        np.clip(target[8], *np.log(np.deg2rad([1.4, 5.0]))),
                    ]
                    rough.append((min(score, done) + 0.4 * max(score, done), parent, daughter, amp))
    unique, refined = [], []
    for _, parent, daughter, amp in sorted(rough, key=lambda row: -row[0]):
        if any(
            np.linalg.norm(daughter[:3] - b[:3]) < 0.09
            and daughter[3:6] @ b[3:6] > 0.97
            and abs(daughter[7] - b[7]) < 0.05
            and np.linalg.norm(parent[:3] - a[:3]) < 0.07
            for a, b in unique
        ):
            continue
        unique.append((parent, daughter))
        try:
            for path in (parent, daughter):
                contain_finite(path)
        except (ArithmeticError, ValueError):
            continue
        light = np.vstack([model.bg, *[optical_shape(model, path, 8)[0] for path in (parent, daughter)]])
        if np.any(light.sum(1) <= 0.0):
            continue
        ll, a, error = profile(light, model.obs, model.ix, amp, budget=12)
        if error <= 1e-5:
            refined.append((ll, coordinates(np.vstack([parent, daughter]), a), "aperture_two_wavefronts"))
        if len(refined) == 2 * limit:
            break
    return sorted(refined, key=lambda row: -row[0])[:limit]
