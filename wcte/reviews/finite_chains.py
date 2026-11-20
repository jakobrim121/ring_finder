"""Shortlist finite, energy-conserving charged sequences from the beam aperture."""

import numpy as np

from light_competition import Template
from wcte.counting import track_counting


def factory(event, background, particles, order):
    from coupled_model import Model
    from staged_physics.primary import Aperture

    aperture = Aperture()

    class BeamChain(Model):
        delta_scale = 0.0

        def __init__(self):
            super().__init__(event.pmt_ids, event.charges_pe, background, particles, 4, order)
            r = aperture.radius
            self.bounds[:3] = [(-r, r), (-r, r), (aperture.z, aperture.z)]

        @property
        def dimension(self):
            return super().dimension - 1

        def inequality(self, x):
            r = aperture.radius
            return np.r_[super().inequality(x), (r**2 - x[:2] @ x[:2]) / r]

        def inequality_jacobian(self, x):
            gradient = np.zeros(len(x))
            gradient[:2] = -2.0 * x[:2] / aperture.radius
            return np.vstack([super().inequality_jacobian(x), gradient])

    return BeamChain()


def measured_axes(geometries):
    result = []
    for geometry in geometries:
        lengths = geometry.get("lengths_mm", [0.0] * len(geometry["axes"]))
        for center, axis, opening, span in zip(
            geometry["vertices"], geometry["axes"], geometry["openings_deg"], lengths
        ):
            origin = np.asarray(center) / 1000.0
            direction = np.asarray(axis, float)
            direction /= np.linalg.norm(direction)
            if any(direction @ row[1] > 0.992 and np.linalg.norm(origin - row[0]) < 0.15 for row in result):
                continue
            result.append((origin, direction, opening, float(span or 0.0) / 1000.0))
    return result


def refine(
    competition,
    candidate,
    *,
    fine=False,
    physical_coverage=False,
    visibility_probability=False,
    lineage_union=False,
):
    """Fully qualify one physical basin with independent optical quadrature."""
    from balanced_search import feasible
    from chain_comparison import counted
    from physical_pulse import components
    from source import trajectory
    from track_coordinates import fit
    from wavefront_support import section_mapping

    event, model = competition.event, competition.model
    _, particles, x, _ = candidate
    for order in (16, 32, 64) if fine else (8, 16, 32):
        coarse = factory(event, model.bg, particles, order)
        opt, x = fit(coarse, x, 36, 54, full_likelihood=True, check_initial=True)
        high = factory(event, model.bg, particles, 2 * order)
        state = high(x, True)
        error = abs(state["ll"] - coarse(x, True)["ll"])
        if not (error < 0.01 and feasible(high, x) and high.hardware_margin(x) > 1e-5):
            continue
        fractions = counted(high, state)
        count = fractions.shape[1] - 1
        if lineage_union:
            from lineage_support import compare

            result = compare(competition, high, state)
            if result is not None:
                result.update(
                    parameters=x,
                    particles=particles,
                    quadrature_error=float(error),
                    order=order,
                    finite_evaluations=int(opt.nfev),
                    physical_count=count,
                )
            return result
        if visibility_probability:
            from chain_sources import paths

            sections = [p for p, _ in paths(x[:-2])]
            groups = [(0,)]
            threshold = track_counting.get().merge_cosine
            for j, (a, b) in enumerate(zip(sections, sections[1:]), 1):
                tangent = trajectory(a, np.ones(1))[1][0]
                if tangent @ b[3:6] >= threshold:
                    groups[-1] += (j,)
                else:
                    groups.append((j,))
        else:
            if count == 0:
                return None
            groups = section_mapping(high, state, fractions)
            if groups is None or len(groups) != count:
                return None
        light, times = components(event, high, state, separate_knockon=True, all_receivers=True)
        a = light.sum(0)
        masks = tuple(tuple(1 + 2 * j for j in group) for group in groups)
        template = Template(
            "finite_" + "_".join(particles),
            light.T / np.maximum(a[:, None], 1e-300),
            times,
            a,
            masks,
            np.zeros(len(a) - 1, dtype=int),
            high.dimension,
            {},
        )
        result = competition.fit(
            template,
            iterations=20,
            fixed_ratios=True,
            physical_coverage=physical_coverage,
            visibility_probability=visibility_probability,
        )
        if result is None:
            return None
        result.update(
            parameters=x,
            particles=particles,
            quadrature_error=float(error),
            order=order,
            finite_evaluations=int(opt.nfev),
            physical_count=count,
        )
        return result
    return None
