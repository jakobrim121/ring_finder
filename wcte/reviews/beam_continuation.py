"""Common detector preparation and finite-track comparison."""

from dataclasses import dataclass

import numpy as np
from wcte.counting import track_counting


@dataclass
class Reference:
    background: np.ndarray
    basis: np.ndarray
    means: np.ndarray
    ll: float
    dimension: int
    geometry: dict
    kind: str


def reference_model(captured, detail):
    """Recover the one-ring comparator in the calibrated charge likelihood."""
    from chord_profile import profile_chord
    from readout import profile
    from replay import frozen_shapes
    from timed_two_track import restore
    from wctering.likelihood.geometry import active_geometry, common_background

    event, old, bank = (captured[name] for name in ("event", "detail", "bank"))
    active, _, _ = active_geometry()
    ix = np.searchsorted(active, event.pmt_ids)
    bg = common_background(np.asarray(bank["1"]["vertices"][0]))
    if old.get("chord") is not None:
        chord = old["chord"]["chord"]
        if (fitted := profile_chord(event, chord, bg)) is None:
            return None
        basis, amplitudes, ll = fitted[1]
        geometry = dict(vertices=[np.asarray(chord["anchor_m"]) * 1000])
        # The chord shape has six geometric coordinates and profiled
        # brightnesses, including each longitudinal section and the halo.
        return Reference(bg, basis, amplitudes @ basis, ll, 6 + len(basis) - 1, geometry, "coherent_chord")

    selected = detail["selected"]
    particles = selected.get("particles")
    if particles and len(particles) == 1 and selected.get("order") and len(selected["parameters"]) == 14:
        model, state, _ = restore(event, bg, selected)
        if abs(state["ll"] - state["log_joint"]) > 1e-5:
            return None
        light = model.primary_cache[1]
        response, background = np.exp(state["parameters"][-2:])
        means = response * light + background * bg
        ll, _ = model.obs.score(means, ix)
        if abs(ll - state["ll"]) > 1e-6:
            return None
        basis = np.stack((bg, light / max(light.sum(), 1e-300)))
        geometry = dict(vertices=[state["parameters"][:3] * 1000])
        return Reference(bg, basis, means, ll, model.dimension, geometry, "finite")

    pred = captured["prediction"]
    if selected["family"] != "released" or pred["fractions"].shape[1] != 2:
        return None
    shapes, saved = frozen_shapes(event.pmt_ids, old, bank, pred["fractions"])
    weights = np.asarray(saved["weights"]) * saved["amplitude"]
    _, fit = profile(shapes, ix, event.charges_pe, weights)
    if not fit["qualified"]:
        return None
    geom = saved["geometry"]
    source = old["fits"][old["selected_fit_index"]]
    dimension = (
        7 if geom.get("emission_model") == "full_water_line" else (9 if source["kind"] == "finite" else 8)
    )
    return Reference(
        bg, shapes, np.asarray(fit["amplitudes"]) @ shapes, fit["ll"], dimension, geom, "optical"
    )


def refine_continuation(
    event, background, particle, parameters, *, orders=(8, 16, 32), daughter=None, daughter_max_range=None
):
    """Refit the linked trajectory and check finer optical quadrature."""
    from balanced_search import feasible
    from chain_comparison import counted
    from coupled_model import Model
    from staged_physics.primary import Aperture
    from track_coordinates import fit

    port = Aperture()

    class Incoming(Model):
        def __init__(self, order):
            super().__init__(
                event.pmt_ids, event.charges_pe, background, (particle, daughter or particle), 4, order
            )
            self.bounds[:3] = [(-port.radius, port.radius), (-port.radius, port.radius), (port.z, port.z)]
            if daughter_max_range is not None:
                self.bounds[18] = self.bounds[18][0], daughter_max_range

        @property
        def dimension(self):
            return super().dimension - 1

        def inequality(self, z):
            return np.r_[super().inequality(z), (port.radius**2 - z[:2] @ z[:2]) / port.radius]

        def inequality_jacobian(self, z):
            gradient = np.zeros(len(z))
            gradient[:2] = -2 * z[:2] / port.radius
            return np.vstack((super().inequality_jacobian(z), gradient))

    z = parameters
    history = []
    for order in orders:
        coarse = Incoming(order)
        solution, z = fit(coarse, z, 40, 72, full_likelihood=True, check_initial=True)
        model = Incoming(order * 2)
        state = model(z, True)
        delta = abs(state["ll"] - coarse(z, True)["ll"])
        valid = bool(feasible(model, z) and model.hardware_margin(z) > 1e-5)
        history.append(dict(error=delta, evaluations=int(solution.nfev), order=order, feasible=valid))
        if valid and delta < 0.01:
            return (
                model,
                state,
                counted(model, state),
                dict(
                    family="beam_vertex_continuation",
                    parameters=z,
                    particles=model.particles,
                    order=order,
                    refinement=history,
                ),
            ), history
    return None, history


def score_candidate(event, trial, reference, family):
    from ring_supported import evidence

    model, state, fractions, details = trial
    count = fractions.shape[1] - 1
    occupied = np.bincount(fractions.argmax(1), minlength=count + 1)[1:]
    prompt = evidence(event, model, state, fractions) if count == 2 else []
    eligible = bool(
        count == 1
        and occupied.all()
        or count == 2
        and occupied.all()
        and prompt
        and all(row["prompt_pmts"] >= track_counting.get().scaled_support(18) and row["prompt_pmts"] >= 0.35 * row["hard_pmts"] for row in prompt)
    )
    gain = 2 * (state["log_joint"] - reference.ll) + (reference.dimension - model.dimension) * np.log(
        len(model.active)
    )
    return dict(
        details,
        family=family,
        count=count,
        eligible=eligible,
        gain=gain,
        detector_ll=state["ll"],
        prompt=prompt,
        dimension=model.dimension,
    ), fractions
