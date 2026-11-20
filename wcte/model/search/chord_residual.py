"""Require absence of a coherent residual before accepting one track."""

import numpy as np

from scipy.optimize import minimize

from readout import observation
from wctering.likelihood.geometry import active_geometry, common_background
from wctering.receiver.point_source import point_source

from throughgoing import spectrum_model


def residuals(event, bank, chord, *, track=None, background=None, profiled=None):
    """Profile independent positive light at fixed fitted cone geometries.

    The alternative cone must improve the detector likelihood beyond the
    main track and diffuse intensity, and own enough independently supported
    PMTs. Search candidates come from detector fits on the same event.
    """
    ids, pos, normals = active_geometry()
    ix = np.searchsorted(ids, event.pmt_ids)
    q = event.charges_pe
    obs = observation(q)
    bg = background if background is not None else common_background(np.asarray(bank[1]["vertices"][0]))
    origin = np.asarray(chord["anchor_m"])
    axis = np.asarray(chord["axis"])
    theta, width = np.deg2rad([chord["opening_deg"], chord["width_deg"]])
    if profiled is not None:
        base, amps, old_ll = profiled
    elif track is None:
        shape, _ = spectrum_model(pos, normals, origin, axis, theta, width, 8)
        amps = np.asarray(chord["amplitudes"])
        old_ll = chord["ll"]
    else:
        from coupled_model import Model

        x = np.asarray(track["parameters"])
        model = Model(event.pmt_ids, q, bg, (track["particle"],), 4, 2 * track["order"])
        old_ll = model(x, True)["ll"]
        signal = model.primary_cache[1]
        shape = signal / signal.sum()
        amps = np.exp(x[-2:][::-1]) * [1.0, signal.sum()]
    if profiled is None:
        base = np.vstack([bg, shape])
    candidates = []
    for key in (2, 3):
        geometry = bank.get(key)
        if geometry is None:
            continue
        for vertex, direction, opening, angular_width in zip(
            geometry["vertices"], geometry["axes"], geometry["openings_deg"], geometry["widths_deg"]
        ):
            v = np.asarray(vertex) / 1000
            a = np.asarray(direction)
            theta = np.deg2rad(opening)
            # The seed fit's width can describe the entire optical halo.
            # Search both that width and a narrower resolvable cone.
            for width_deg in sorted({float(angular_width), min(float(angular_width), 4.0)}):
                x = np.r_[v, a, theta, np.log(np.deg2rad(width_deg))]
                light, _ = point_source(pos, normals, x)
                sources = np.vstack([base, light])
                total = max(q.sum() / obs.gain, 1.0)

                start = np.r_[amps, min(10.0, total / 100)]
                if profiled is None:

                    def objective(a):
                        ll, g = obs.score(a @ sources, ix)
                        return -ll / total, (sources @ g) / total

                    fitted = minimize(
                        objective,
                        start,
                        method="L-BFGS-B",
                        jac=True,
                        bounds=[(1e-8, 1e6)] * 3,
                        options=dict(maxiter=40, ftol=1e-10, gtol=1e-8),
                    )
                    ll, fitted_amps = -fitted.fun * total, fitted.x
                else:
                    from chord_profile import profile

                    ll, fitted_amps, _ = profile(sources, obs, ix, start)
                means = fitted_amps @ sources
                fraction = fitted_amps[-1] * light[ix] / means[ix]
                support = np.sum(1 - obs.no_source(means[ix], fraction))
                own = np.count_nonzero(fraction > 0.5)
                gain = 2 * (ll - old_ll) - 8 * np.log(len(ids))
                angle = np.rad2deg(np.arccos(np.clip(axis @ a, -1.0, 1.0)))
                distance = np.linalg.norm(event.pmt_positions_mm - v * 1000, axis=1)
                from wctering.seeding.coherent_timing import C, N_GROUP

                residual_time = event.times_ns - N_GROUP * distance / C
                from charge_starts import weighted_quantile

                timing = weighted_quantile(residual_time, fraction * np.minimum(q, 2.0), [0.16, 0.5, 0.84])
                residual_charge = fraction * np.minimum(q, 2.0)
                record = dict(
                    gain=float(gain),
                    photon_support=float(support),
                    hard_support=int(own),
                    axis_separation_deg=float(angle),
                    opening_deg=opening,
                    width_deg=width_deg,
                    axis=a.tolist(),
                    vertex_m=v.tolist(),
                    residual_charge_fraction=float(residual_charge.sum() / np.minimum(q, 2.0).sum()),
                    timing_half_width_ns=float(0.5 * (timing[2] - timing[0])),
                )
                if profiled is not None:
                    from chord_profile import timing_evidence

                    record["relative_timing_gain"] = timing_evidence(event, chord, v, a, theta, fraction)
                candidates.append(record)
    return sorted(candidates, key=lambda r: r["gain"], reverse=True)
