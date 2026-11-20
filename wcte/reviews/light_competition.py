"""Compare counted cones and bounded short-lepton light under the same detector law."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Template:
    label: str
    shapes: np.ndarray
    center: np.ndarray
    amplitudes: np.ndarray
    core_groups: tuple
    time_groups: np.ndarray
    dimensions: int
    limits: dict
    intensity_groups: np.ndarray | None = None


def optical_template(model, geometry, label):
    from timed_optical import arrival
    from wctering.receiver.angular_source import source as finite_source
    from wctering.receiver.point_source import point_source

    shapes = [model.bg]
    lengths = geometry.get("lengths_mm", [0.0] * len(geometry["axes"]))
    if geometry.get("emission_model") == "full_water_line":
        return None
    for point, axis, opening, width, length in zip(
        geometry["vertices"], geometry["axes"], geometry["openings_deg"], geometry["widths_deg"], lengths
    ):
        x = np.r_[np.asarray(point) / 1000.0, axis]
        optics = np.radians(opening), np.log(np.radians(width))
        if length and length > 0.1:
            shape = finite_source(model.pos, model.normals, np.r_[x, length / 1000.0, optics], 16)[0]
        else:
            shape = point_source(model.pos, model.normals, np.r_[x, optics])[0]
        shapes.append(shape)
    shapes = np.asarray(shapes)
    if not np.all(shapes.sum(1) > 0.0):
        return None
    k = len(shapes) - 1
    centers = arrival(None, geometry, positions=model.pos)
    amps = np.asarray(geometry.get("weights", np.ones(len(shapes))), dtype=float)
    amps = model.total * amps / max(amps.sum(), 1e-10)
    dims = 1 + 8 * k + sum(bool(v and v > 0.1) for v in lengths)
    return Template(
        label,
        shapes / shapes.sum(1)[:, None],
        centers,
        amps,
        tuple((i,) for i in range(1, k + 1)),
        np.arange(k),
        dims,
        {},
    )


def physical_template(event, model, record):
    from physical_pulse import components
    from timed_two_track import restore
    from wavefront_support import section_mapping

    count = record["count"]
    particles = record["particles"]
    if len(record["parameters"]) != 5 + 9 * len(particles) or not record.get("order"):
        return None
    fit_model, state, fractions = restore(event, model.bg, record)
    if count != fractions.shape[1] - 1:
        return None
    groups = section_mapping(fit_model, state, fractions)
    if groups is None or len(groups) != count:
        return None
    rates, centers = components(event, fit_model, state, separate_knockon=True, all_receivers=True)
    amplitudes = rates.sum(0)
    core = tuple(tuple(1 + 2 * j for j in g) for g in groups)
    angles = np.zeros(len(amplitudes) - 1, dtype=int)
    # Keep each range's core/tail light ratio and exact propagation time.
    return Template(
        record["family"],
        rates.T / np.maximum(amplitudes, 1e-300)[:, None],
        centers,
        amplitudes,
        core,
        angles,
        fit_model.dimension,
        {},
    )


def short_lepton_templates(model, geometry, template, ring, *, flex_core):
    """Nominate a few physically budgeted alternatives around the same light ridge."""
    from directional_shower import geometry_modes, shower
    from material import material
    from residual_joint_optics import path_from_geometry

    try:
        path = path_from_geometry(geometry, ring - 1)
    except ValueError:
        return
    water = material("electron")
    edge = water["energy_at"](water["margin_mm"])
    energy = 0.98
    length = float(water["range_at"](energy * edge)) / 1000.0
    all_tubes = type("Receiver", (), dict(pos=model.pos, normals=model.normals, ix=slice(None)))
    control = template.amplitudes @ template.shapes
    _, slope = model.obs.score(control, model.ix)
    own = template.amplitudes[ring] * template.shapes[ring]
    candidates = []
    for birth, axis in geometry_modes(path, length):
        shape, arrival = shower(all_tubes, birth, axis, energy, 16)
        mass = shape.sum()
        if not np.isfinite(mass) or mass <= 1e-15:
            continue
        shape = shape / mass
        # The calibrated absolute response limits the total yield of one
        # sub-margin electron at its fitted location and distance.
        yield_cap = mass * model.response
        if yield_cap <= 2.0:
            continue
        start = min(template.amplitudes[ring], yield_cap)
        nominated = model.obs.score(control - own + start * shape, model.ix)[0]
        candidates.append((nominated, shape, arrival, start, yield_cap))
    for _, shape, arrival, start, limit in sorted(candidates, key=lambda r: -r[0])[:1]:
        shapes, center = template.shapes.copy(), template.center.copy()
        amplitudes = template.amplitudes.copy()
        shapes[ring], center[:, ring - 1], amplitudes[ring] = shape, arrival, start
        lower = tuple(g for g in template.core_groups if g != (ring,))
        yield Template(
            template.label + f"_soft{ring}",
            shapes,
            center,
            amplitudes,
            lower,
            template.time_groups,
            template.dimensions - 1,
            {ring: limit},
        )
        if flex_core:
            consistent = np.column_stack([template.center, arrival])
            yield Template(
                template.label + f"_core_soft{ring}",
                np.vstack([template.shapes, shape]),
                consistent,
                np.r_[amplitudes, min(0.5 * start, limit)],
                template.core_groups,
                np.r_[template.time_groups, ring - 1],
                template.dimensions + 1,
                {len(amplitudes): limit},
            )


class Competition:
    def __init__(
        self,
        event,
        model,
        *,
        prompt=False,
        after_first_jitter=False,
        primary_models=None,
        receiver_weights=None,
        module_charge=False,
    ):
        from selected_pulse import Selection
        from module_optics import conditional

        self.event = event
        self.model = model
        self.selection = Selection(event, model)
        self.after_first_jitter = after_first_jitter
        if after_first_jitter:
            from after_first_jitter import Selection as JitteredSelection

            self.selection = JitteredSelection(self.selection)
        self.modules = np.unique(model.active // 100, return_inverse=True)[1]
        self.conditional = conditional
        self.q = np.minimum(model.q, 2.0)
        self.prompt = prompt
        self.primary_models = primary_models
        self.receiver_weights = receiver_weights
        self.module_charge = module_charge

    def finite_primary(self, particle):
        """Reuse an identical finite beam fit across comparisons of one event."""
        from coupled_model import Model
        from value_seed_beam_candidate import fit

        background = self.model.bg
        if self.primary_models is None:
            return fit(self.event, background, particle, budget=56)
        key = id(self.event), particle, Model.delta_scale, background.tobytes()
        if key not in self.primary_models:
            self.primary_models[key] = fit(self.event, background, particle, budget=56)
        return self.primary_models[key]

    def fit(
        self,
        template,
        *,
        iterations=32,
        fixed_ratios=False,
        physical_coverage=False,
        visibility_probability=False,
        score_only=False,
    ):
        from charge_starts import weighted_quantile
        from module_supported_ring import lower_bound
        from physical_sharing import ownership as first_arrival_shares
        from resolved_ring_evidence import first_pulse_support

        model, selection = self.model, self.selection
        ix, obs = model.ix, model.obs
        receiver_weights = self.receiver_weights
        if receiver_weights is None:
            charge_score = obs.score
        else:
            from readout import positive

            def charge_score(means, hit_indices):
                logp, derivative = obs.no_hit(means)
                hit_rates = np.maximum(means[hit_indices], 1e-300)
                observed, photons, _, tail = positive(
                    hit_rates, obs.offsets, obs.coefficients, obs.zero_source
                )
                if tail.max(initial=0.0) > 1e-11:
                    raise ArithmeticError("Photon-count summation did not converge")
                logp[hit_indices] = observed
                derivative[hit_indices] = 1.0 - photons / hit_rates
                return float(receiver_weights @ logp), receiver_weights * derivative

        shapes = template.shapes
        n = len(shapes)
        g = template.time_groups
        amplitude_groups = np.arange(n)
        if template.intensity_groups is not None:
            amplitude_groups = template.intensity_groups
        elif fixed_ratios:
            amplitude_groups[1:] = 1
        n_amp = amplitude_groups.max() + 1
        v = np.asarray(template.amplitudes, float)
        starts = np.bincount(amplitude_groups, weights=v)
        relative_amp = np.divide(
            v, starts[amplitude_groups], out=np.zeros_like(v), where=starts[amplitude_groups] > 0.0
        )
        amplitude = np.clip(starts / model.total, 1e-7, 1e6 / model.total)
        weights = v[1:, None] * shapes[1:, ix] * self.q
        center = np.asarray(template.center)
        time_weight = np.bincount(g, weights=weights.sum(1))
        clocks = []
        for group in range(len(time_weight)):
            part = g == group
            weights_at_group = weights[part].ravel()
            observed = (selection.time - center[ix][:, part].T).ravel()
            clocks.append(
                weighted_quantile(observed, weights_at_group, [0.5])[0]
                if weights_at_group.sum() > 0.0
                else -5.0
            )
        amplitudes_bounds = [(1e-8, 1e6 / model.total)] * len(amplitude)
        for j, upper in template.limits.items():
            lo, hi = (0.0, upper) if np.isscalar(upper) else upper
            amplitudes_bounds[j] = (max(1e-8, lo / model.total), hi / model.total)
            amplitude[j] = np.clip(amplitude[j], *amplitudes_bounds[j])
        shapes = shapes.T * (relative_amp * model.total)
        initial = np.r_[amplitude, clocks]
        bounds = amplitudes_bounds + [(t - 4.0, t + 4.0) for t in clocks]

        def objective(x):
            rates = shapes * x[:n_amp][amplitude_groups]
            times = center + x[n_amp:][g]
            scores, amp_slope, clock_slope = selection.score(rates, times)
            if receiver_weights is not None:
                scores = receiver_weights * scores
                amp_slope = receiver_weights[:, None] * amp_slope
                clock_slope = receiver_weights[:, None] * clock_slope
            grad = np.r_[
                np.bincount(amplitude_groups, weights=(shapes * amp_slope).sum(0)),
                np.bincount(g, weights=clock_slope.sum(0)),
            ]
            return -scores.sum() / model.total, -grad / model.total

        seed_evaluation = objective(initial)
        start = seed_evaluation[0]

        def evaluate(x):
            return seed_evaluation if np.array_equal(x, initial) else objective(x)

        opt = minimize(
            evaluate,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=iterations, ftol=2e-9, gtol=2e-6),
        )
        if not np.isfinite(opt.fun) or opt.fun > start + 1e-10:
            return None
        x = opt.x
        rates = shapes * x[:n_amp][amplitude_groups]
        centers = center + x[n_amp:][g]
        if score_only:
            if self.after_first_jitter:
                joint, _ = selection.integrated_weights(selection.conditional(rates, centers))
            else:
                joint, _, _ = selection.score(rates, centers)
            if receiver_weights is not None:
                joint = joint * receiver_weights
            return dict(
                label=template.label,
                count=len(template.core_groups),
                dimension=template.dimensions,
                joint=float(joint.sum()),
                charge=float(charge_score(rates.sum(1), ix)[0]),
                ring_amplitudes=rates.sum(0),
            )
        hit_rates, hit_times = rates[ix], centers[ix]
        charge = hit_rates / np.maximum(hit_rates.sum(1), 1e-300)[:, None]
        in_gate, gate_presence, joint = None, None, None
        masks = np.asarray([sum(1 << j for j in group) for group in template.core_groups], dtype=np.int64)
        if self.after_first_jitter:
            timed, in_gate, gate_presence, joint = selection.ownership_with_score(
                rates, centers, masks, prompt=self.prompt
            )
        else:
            timed = first_arrival_shares(
                selection.time,
                selection.gate,
                hit_rates,
                hit_times,
                selection.width[ix],
                obs.offsets,
                obs.coefficients,
            )
            if self.prompt:
                from prompt_ownership import ownership

                in_gate, gate_presence = ownership(
                    selection.time,
                    selection.gate,
                    hit_rates,
                    hit_times,
                    selection.width[ix],
                    obs.offsets,
                    obs.coefficients,
                    masks,
                )
        realizations = [("charge", charge), ("timed", timed)]
        if in_gate is not None:
            realizations.append(("prompt", in_gate))
        ownership = {}
        for name, values in realizations:
            parts = np.column_stack([values[:, list(group)].sum(1) for group in template.core_groups])
            ownership[name] = np.column_stack([1.0 - parts.sum(1), parts])
        measured = []
        for j, (group, f) in enumerate(zip(template.core_groups, ownership["charge"].T[1:])):
            mask = sum(1 << (j - 1) for j in group)
            pmt = 1.0 - obs.no_source(hit_rates.sum(1), np.clip(f, 0.0, 1.0))
            prompt = first_pulse_support(
                selection.time,
                selection.gate,
                hit_rates,
                hit_times,
                selection.width[ix],
                -1,
                obs.offsets,
                obs.coefficients,
                excluded_mask=mask,
            )
            region = lower_bound(self.event.pmt_ids, prompt)
            region.update(
                charge_support=float(pmt.sum()),
                weighted_charge=float(self.q @ f),
                amplitude=float(rates[:, list(group)].sum()),
            )
            if gate_presence is not None:
                gated = lower_bound(self.event.pmt_ids, gate_presence[:, j])
                region.update({f"gate_{name}": value for name, value in gated.items()})
                if visibility_probability:
                    from visible_probability import visibility

                    probability, mean = visibility(self.event.pmt_ids, gate_presence[:, j])
                    region.update(probability_of_30=probability, distinct_support_mean=mean)
            measured.append(region)
        if joint is None:
            joint, _, _ = selection.score(rates, centers)
        if receiver_weights is not None:
            joint = joint * receiver_weights
        total = rates.sum(1)
        scores = dict(
            label=template.label,
            count=len(measured),
            dimension=template.dimensions,
            joint=float(joint.sum()),
            charge=float(charge_score(total, ix)[0]),
            time_fit=True,
            evals=opt.nfev,
            neighborhoods=measured,
            fractions=ownership,
            joint_by_module=np.bincount(self.modules, weights=joint),
            module_shape=self.conditional(model.active, ix, model.q, total),
            ring_amplitudes=rates.sum(0),
            clocks=x[n_amp:],
        )
        if self.module_charge:
            from readout import positive

            charge_per_pmt, _ = obs.no_hit(total)
            observed, _, _, _ = positive(
                np.maximum(total[ix], 1e-300), obs.offsets, obs.coefficients, obs.zero_source
            )
            charge_per_pmt[ix] = observed
            if receiver_weights is not None:
                charge_per_pmt *= receiver_weights
            scores["charge_by_module"] = np.bincount(self.modules, weights=charge_per_pmt)
        if physical_coverage:
            last = in_gate if gate_presence is not None else timed
            scores["modeled_charge_share"] = 1.0 - last[:, 0]
            counted = {i for group in template.core_groups for i in group}
            uncounted = sorted(set(range(1, len(template.amplitudes))) - counted)
            scores["uncounted_source_share"] = last[:, uncounted].sum(1)
        return scores
