"""Search along a measured entrance path for unmodelled prompt light."""

import numpy as np
from numba import njit

from gated_photon_time import series, wavefront as pulse
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


@njit(cache=True)
def pulse_adjoint(times, gate, rates, centers, sigma, offsets, coefficients):
    """Factors of the time-score derivative for a source of zero intensity."""
    weights = np.empty((len(times), 4))
    width = gate[1] - gate[0]
    for i, time in enumerate(times):
        sky = rates[i, 0]
        first = sky / width
        after = first * (gate[1] - time)
        low, high = sky, 0.0
        for mean, rate in zip(centers[i], rates[i, 1:]):
            density, surv, _ = pulse(time - mean, sigma[i])
            first += rate * density
            after += rate * surv
            low += rate * pulse(gate[0] - mean, sigma[i])[1]
            high += rate * pulse(gate[1] - mean, sigma[i])[1]
        a, b = offsets[i : i + 2]
        _, d_after = series(after, coefficients, a, b, 1)
        left, d_left = series(low, coefficients, a, b, 0)
        right, d_right = series(high, coefficients, a, b, 0)
        delta = min(right - left, -1e-15)
        accept = -np.expm1(delta)
        weights[i, 0] = 1.0 / max(first, 1e-300)
        weights[i, 1] = d_after
        weights[i, 2] = -d_left / accept
        weights[i, 3] = d_right * np.exp(delta) / accept
    return weights


@njit(cache=True)
def cone_scores(
    delta,
    normals,
    axes,
    openings,
    spans,
    time,
    sigma,
    used,
    gate,
    clock,
    charge_slope,
    adjoint,
    mean,
    timing=True,
    time_only=False,
):
    """Score finite cones on every PMT using charge and first arrival."""
    score = np.zeros((len(axes), len(openings), len(spans)))
    for i, axis in enumerate(axes):
        for j, opening in enumerate(openings):
            sine, cosine = np.sin(opening), np.cos(opening)
            gradient, info = np.zeros(len(spans)), np.zeros(len(spans))
            for k in range(len(delta)):
                vx, vy, vz = delta[k]
                px, py, pz = axis
                z = vx * px + vy * py + vz * pz
                r2 = max(vx * vx + vy * vy + vz * vz - z * z, 0.0)
                radial = np.sqrt(r2)
                s = z - radial * cosine / sine
                width = max(25.0, np.deg2rad(2.8) * radial / sine**2)
                if s < -3.0 * width or s > spans[-1] + 3.0 * width:
                    continue
                previous_u = -1.0
                geometric, derivative, denominator = 0.0, 0.0, 1.0
                for l, length in enumerate(spans):
                    u = min(max(s, 0.0), length)
                    edge = (s - u) / width
                    if abs(edge) >= 3.0:
                        continue
                    if u != previous_u:
                        dx, dy, dz = vx - u * px, vy - u * py, vz - u * pz
                        d2 = dx * dx + dy * dy + dz * dz
                        d = np.sqrt(max(d2, 1e-8))
                        nx, ny, nz = normals[k]
                        facing = max(-(nx * dx + ny * dy + nz * dz) / d, 0.025)
                        geometric = 1e6 * facing
                        denominator = d2 + 10000.0
                        derivative = 0.0 if time_only else charge_slope[k]
                        if timing and used[k]:
                            arrival = clock + (N_PHASE * cosine * u + N_GROUP * d) / C
                            f, g, _ = pulse(time[k] - arrival, sigma[k])
                            a, b, c, e = adjoint[k]
                            shift = f * a + g * b + c * pulse(gate[0] - arrival, sigma[k])[1]
                            shift += e * pulse(gate[1] - arrival, sigma[k])[1]
                            # Bound the leverage of isolated pulses at
                            # vanishing intensity in the linear expansion.
                            clip = 3.0 / max(mean[k], 0.15)
                            derivative += min(max(shift, -clip), clip)
                        # Several span limits can contain the same emission
                        # point. Reuse the exact same ray and clock terms.
                        previous_u = u
                    light = geometric * np.exp(-0.5 * edge * edge) / denominator
                    gradient[l] += light * derivative
                    info[l] += light**2 / max(mean[k], 0.15)
            for l in range(len(spans)):
                if gradient[l] > 0.0 and info[l] > 1e-10:
                    score[i, j, l] = gradient[l] ** 2 / info[l]
    return score


def measurements(compare, rates, centers):
    """Arrange measured first-PE derivatives in detector receiver order."""
    model = compare.model
    ix = model.ix
    hits = np.zeros(len(model.pos), dtype=bool)
    hits[ix] = True
    time, sigma = np.zeros(len(hits)), np.zeros(len(hits))
    weight = np.zeros((len(hits), 4))
    time[ix], sigma[ix] = compare.time, compare.sigma
    weight[ix] = pulse_adjoint(
        compare.time, compare.gate, rates, centers, compare.sigma, model.obs.offsets, model.obs.coefficients
    )
    return time, sigma, hits, weight
