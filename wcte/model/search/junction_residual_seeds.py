"""Look for short emitting tracks at points along the measured tracks."""

from functools import lru_cache

import numpy as np
from wcte.counting import track_counting
from numba import njit

from analog_optical import contain_finite
from charge_starts import allowed_chord
from chord_profile import profile
from finite_residual_fit import optical_shape
from residual_search import axis_grid
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


OPENINGS = np.deg2rad([23.0, 30.0, 35.0, 39.0, 42.0])
SINES = np.sin(OPENINGS)
COSINES = np.cos(OPENINGS)


@lru_cache(maxsize=1)
def directions():
    return axis_grid(480)


@njit(cache=True)
def ridge_scores(delta, distance, axes, t, q, excess, clock, lengths):
    """Test a finite emission interval on either side of a candidate junction.

    Intersect each PMT ray with the cone to infer its emission coordinate.
    Use a broad angular resolution here so the grid can seed narrower rings.
    The full finite-track likelihood later evaluates hit and unhit PMTs.
    """
    score = np.zeros((len(lengths), len(axes), len(OPENINGS), 2))
    extent = max(lengths)
    for i, axis in enumerate(axes):
        for j in range(len(distance)):
            z = delta[j, 0] * axis[0] + delta[j, 1] * axis[1] + delta[j, 2] * axis[2]
            r = np.sqrt(max(distance[j] ** 2 - z * z, 0.0))
            for k, (sine, cosine) in enumerate(zip(SINES, COSINES)):
                s = z - r * cosine / sine
                # A PMT's opening-angle error grows into a longer interval
                # of possible emission coordinates at greater distance.
                sigma = max(30.0, 0.079 * r / sine**2)
                if abs(s) > extent + 2.5 * sigma:
                    continue
                flight = (N_PHASE * cosine * s + N_GROUP * r / sine) / C
                mismatch = (t[j] - clock - flight) / 2.0
                if abs(mismatch) > 3.5:
                    continue
                timed = q[j] * excess[j] * np.exp(-0.5 * mismatch * mismatch)
                for h, length in enumerate(lengths):
                    if abs(s) > length + 2.5 * sigma:
                        continue
                    for sign in range(2):
                        along = s if sign else -s
                        edge = (along - min(max(along, 0.0), length)) / sigma
                        score[h, i, k, sign] += timed * np.exp(-0.5 * edge * edge)
    return score


def ridge_score(delta, distance, axes, time, charge, excess, clock):
    return ridge_scores(delta, distance, axes, time, charge, excess, clock, (240.0,))[0]


def scan(event, paths, shares, excess, scale):
    """Walk each measured path and keep distinct residual directions."""
    from charge_starts import weighted_quantile
    from residual_joint_optics import prompt
    from joint_prompt_fit import wavefront

    t = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    old_time = np.column_stack([wavefront(event.pmt_positions_mm, p)[0] for p in paths])
    _, clocks = prompt(event, old_time, shares)
    grid = directions()
    peaks = []
    for j, p in enumerate(paths):
        # Prefer locations where the existing optical fit allows a nearby
        # interaction. Sample the interior as a wide ring can bridge a kink.
        extent = np.clip(p[6] * 0.5, 0.045, 0.65)
        for offset in np.unique(np.r_[np.linspace(-extent, extent, 5), -0.055, 0.055]):
            point = p[:3] + offset * p[3:6]
            delta = event.pmt_positions_mm - 1000.0 * point
            distance = np.linalg.norm(delta, axis=1)
            clock = clocks[j] + 1000.0 * offset * N_PHASE * np.cos(p[7]) / C
            score = ridge_score(delta, distance, grid, t, q, excess, clock)
            ordered = np.argsort(score.ravel())[::-1]
            chosen = []
            for key in ordered:
                a, b, side = np.unravel_index(key, score.shape)
                strength, axis, opening = score[a, b, side], grid[a], OPENINGS[b]
                if strength < 9.0:
                    break
                if axis @ p[3:6] > np.cos(np.deg2rad(track_counting.get().turn_offset(5.0))):
                    continue
                if any(axis @ prior > 0.97 and abs(opening - angle) < 0.11 for prior, angle in chosen):
                    continue
                chosen.append((axis, opening))
                along = delta @ axis
                radial = np.linalg.norm(delta - along[:, None] * axis, axis=1)
                sine, cosine = np.sin(opening), np.cos(opening)
                where = along - radial * cosine / sine
                flight = (N_PHASE * cosine * where + N_GROUP * radial / sine) / C
                sign = 1.0 if side else -1.0
                signed = where * sign
                weight = q * np.maximum(excess, 0.0) * np.exp(-0.5 * ((t - clock - flight) / 2.0) ** 2)
                weight *= (signed >= -90.0) & (signed <= 300.0)
                water = allowed_chord(point, sign * axis)
                if weight.sum() >= 11.0 and water is not None and water[0] <= 0.002 and water[1] >= 0.025:
                    length = np.clip(
                        weighted_quantile(signed / 1000.0, weight, [0.78])[0] + 0.028, 0.035, 0.27
                    )
                    length = min(length, water[1])
                    cone = np.r_[
                        point + 0.5 * sign * length * axis, axis, length, opening, np.log(np.deg2rad(2.8))
                    ]
                    try:
                        contain_finite(cone)
                    except (ValueError, ArithmeticError):
                        pass
                    else:
                        # A start on the measured trajectory can share an
                        # endpoint, even if its independent optical fit moves.
                        peaks.append((strength, cone, j, offset, sign))
                if len(chosen) == 2:
                    break
    qualified = []
    for strength, p, section, offset, sign in sorted(peaks, key=lambda row: row[0], reverse=True):
        if any(
            np.linalg.norm(p[:3] - a[:3]) < 0.09 and p[3:6] @ a[3:6] > 0.972 and abs(p[7] - a[7]) < 0.09
            for a, *_ in qualified
        ):
            continue
        qualified.append((p, section, offset, sign, np.clip(strength, 12.0, 0.35 * scale)))
        if len(qualified) == 8:
            break
    return qualified


def start(event, model, paths, shares, shapes, amp):
    """Profile both the original and shortened parent with one new segment."""
    mean = amp @ shapes
    _, gradient = model.obs.score(mean, model.ix)
    # Keep negative charge contrast. Otherwise every bright edge near an
    # existing ring scores as residual light even when it is already fit.
    excess = np.clip(-gradient[model.ix] * mean[model.ix], -0.7, 2.0)
    if np.count_nonzero(excess > 0.25) < 18:
        return None
    best = None
    for cone, j, offset, sign, brightness in scan(event, paths, shares, excess, amp.sum()):
        # A true scattered leg may replace a poorly placed optical tail.
        # Test moving the main interval's endpoint before the joint solve.
        old = paths[j]
        edge = -0.5 * sign * old[6]
        shortened = old.copy()
        shortened[:3] += 0.5 * (edge + offset) * old[3:6]
        shortened[6] = sign * (offset - edge)
        for parent in (old.copy(), shortened):
            if parent[6] < 0.02:
                continue
            try:
                contain_finite(parent)
            except (ValueError, ArithmeticError):
                continue
            trial = np.vstack([paths, cone])
            trial[j] = parent
            basis = np.vstack([shapes[0], *[optical_shape(model, p, 4)[0] for p in trial]])
            if np.any(basis[1:].sum(1) <= 0.0):
                continue
            ll, photons, error = profile(basis, model.obs, model.ix, np.r_[amp, brightness], budget=18)
            if error <= 1e-5 and photons[-1] >= 12.0 and (best is None or ll > best[0]):
                best = ll, trial, photons, "junction_residual"
    return best
