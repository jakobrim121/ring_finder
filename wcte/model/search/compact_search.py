"""Refine and compare finite-track hypotheses using calibrated charge and time.

Optical cone fits provide starting geometry. The charged search compares
latent masses and continuous trajectories, then tests residual light and
causal timing before changing the ring count or sharing charge across rings.
"""

from dataclasses import dataclass
from functools import partial
from itertools import chain, product
import time

import numpy as np

import kernel_setup  # noqa: F401  Configure the charged-track integral.
import analytic_knockon
import bare_model
import coupled_model
import pair_domes
import readout
from balanced_search import feasible
from bare_residual_search import proposals, repair_energy, serial_coordinates
from chain_comparison import MARGIN, counted
from chain_soft_search import canonical
from optical_chain_search import geometries
from readout import profile
from replay import frozen_shapes
from residual_search import starting_paths
from wctering.receiver.cache import event_cache
from wctering.seeding.search import seeded_event
from threadpoolctl import threadpool_limits

from fast_knockon import install
from fast_domes import certificate
from exact_series import positive_exact, positive_without_source

PARTICLES = ("electron", "muon", "pion", "proton")
install(analytic_knockon)
readout.positive = positive_exact
pair_domes.certificate = certificate


@dataclass
class Proposal:
    particles: tuple[str, ...]
    coordinates: np.ndarray
    preliminary_score: float
    family: str


def prepare(raw, event_id, *, use_water_chord):
    """Build the seed bank and an optical partition from the measured hits."""
    hits, diagnostic, bank, _ = seeded_event(raw, event_id)
    chord_detail = None
    # A full water chord has calibrated direct light and a wide diffuse halo.
    if use_water_chord and diagnostic["predicted_ring_class"] > 1:
        from orthogonal_chord import accepts_profile, test

        trial = test(hits, bank, span=0.74, relative="hybrid")
        if trial is not None:
            fractions, chord_detail = trial
            allowed = accepts_profile(chord_detail, 0.74, relative="hybrid")
            if allowed:
                allowed = chord_detail["chord"]["direct_yield_fraction"] >= 0.65
                if not allowed:
                    chord_detail["failed_direct_yield_fraction"] = 0.65
            if allowed:
                assert np.any(fractions.argmax(1) == 1)
                prediction = dict(
                    pmt_ids=hits.pmt_ids,
                    charges=hits.charges_pe,
                    input_indices=hits.input_indices,
                    fractions=fractions,
                )
                return (
                    prediction,
                    dict(fits=[], selected_fit_index=None, chord=chord_detail),
                    {str(count): seed for count, seed in bank.items()},
                    hits,
                )

    from charge_reference import reference

    result = reference(hits, diagnostic, bank)
    result.validate()
    if chord_detail is not None:
        result.diagnostics["rejected_chord"] = chord_detail
    prediction = dict(
        pmt_ids=hits.pmt_ids,
        charges=hits.charges_pe,
        input_indices=hits.input_indices,
        fractions=result.fractions,
    )
    return (
        prediction,
        dict(result.diagnostics, counter=diagnostic.get("unified_counter")),
        {str(count): seed for count, seed in bank.items()},
        hits,
    )


def candidate(model, x, family):
    log_joint = model(x, True)["log_joint"]
    score = 2 * log_joint - model.dimension * np.log(len(model.active))
    return Proposal(model.particles, x, float(score), family)


def mass_choices(count):
    if count < 3:
        return product(PARTICLES, repeat=count)
    return ((first,) + (second,) * (count - 1) for first, second in product(PARTICLES, repeat=2))


def charge_proposals(ids, q, background, old, bank, count, normalization):
    """Search a small set of physically bright initial geometries."""
    from scipy.optimize import lsq_linear
    from charge_starts import initializations, nominal_sections, recoil_start

    prototype = bare_model.Model(ids, q, background, ("muon",), 4, 2)
    models = {}
    result = []
    optical = geometries(old, bank, count)
    if not optical:
        return []
    starts = [optical[0]]
    if len(optical) > 1:
        starts.append(optical[-1])

    def model_for(masses):
        if masses not in models:
            models[masses] = bare_model.Model(ids, q, background, masses, 4, 2)
        return models[masses]

    for geometry in starts:
        section = nominal_sections(geometry, prototype)
        if section is None:
            continue
        # Optical extents can repeat across latent masses. Reuse that
        # geometry fit while keeping the solver and its inputs identical.
        geometry_fits = {}

        def fit_geometry(matrix, target, **options):
            key = matrix.tobytes(), target.tobytes()
            if key not in geometry_fits:
                geometry_fits[key] = lsq_linear(matrix, target, **options)
            return geometry_fits[key]

        for particles in mass_choices(count):
            starts = initializations(section, particles, normalization, fit_geometry=fit_geometry)
            for masses, start in starts:
                model = model_for(masses)
                x = recoil_start(model, start)
                if x is not None and feasible(model, x) and model.hardware_margin(x) > 1e-5:
                    result.append(candidate(model, x, "absolute_yield"))
    chosen = []
    for row in sorted(result, key=lambda v: v.preliminary_score, reverse=True):
        if any(
            row.particles == other.particles and np.linalg.norm(row.coordinates - other.coordinates) < 0.06
            for other in chosen
        ):
            continue
        chosen.append(row)
        if len(chosen) == 3:
            break
    fine = [
        candidate(bare_model.Model(ids, q, background, row.particles, 4, 8), row.coordinates, row.family)
        for row in chosen
    ]
    return sorted(fine, key=lambda row: row.preliminary_score, reverse=True)


def minimize_scaled(model, x, iterations, chart, maximum_evaluations):
    """Use junction coordinates when the charged sections share a vertex."""
    if chart == "junction":
        from junction_coordinates import minimize as fit_junction

        return fit_junction(model, x, iterations, maximum_evaluations)
    from track_coordinates import fit

    return fit(model, x, iterations, maximum_evaluations, full_likelihood=chart == "chart_full")


def residual_proposals(model, x):
    """Search for a second charged source after fitting the main track."""
    peaks = proposals(model, x)
    starts = sorted(starting_paths(x, model.particles[0], peaks), key=lambda row: row[2], reverse=True)
    result = []
    for masses, coordinates, _ in starts[:16]:
        trial = coupled_model.Model(model.active[model.ix], model.q, model.bg, masses, 4, 4)
        z = canonical(trial, serial_coordinates(coordinates))
        if z is None:
            continue
        z = repair_energy(trial, z)
        if z is not None and feasible(trial, z) and trial.hardware_margin(z) > 1e-5:
            result.append(candidate(trial, z, "residual"))
    return sorted(result, key=lambda row: row.preliminary_score, reverse=True)


def refine(ids, q, background, proposal, *, chart, maximum_evaluations):
    """Fit the nominated mode, then check its feasibility and light integral."""
    factory = bare_model.Model if all((p == "electron" for p in proposal.particles)) else coupled_model.Model
    x = proposal.coordinates
    evaluations = 0
    for order in (16, 32):
        model = factory(ids, q, background, proposal.particles, 4, order)
        initial = model(x, True)
        fine = factory(ids, q, background, proposal.particles, 4, 2 * order)
        initial_error = abs(initial["ll"] - fine(x, True)["ll"])
        if initial_error >= 0.01 and order == 16:
            continue
        fit, z = minimize_scaled(model, x, 40, chart, maximum_evaluations)
        evaluations += fit.nfev
        if not feasible(model, z) or model.hardware_margin(z) <= 1e-05:
            z = x
        coarse = model(z, True)
        if coarse["log_joint"] < initial["log_joint"]:
            z = x
            coarse = model(x, True)
        state = fine(z, True)
        error = abs(coarse["ll"] - state["ll"])
        kkt = fine.stationarity(z, state["gradient"])
        if error < 0.01:
            return (
                fine,
                state,
                dict(
                    family=proposal.family,
                    particles=proposal.particles,
                    order=order,
                    quadrature_ll_error=error,
                    stationarity=kkt,
                    optimizer_converged=bool(fit.success),
                    evaluations=evaluations,
                ),
            )
        x = z
    return None


def search(pred, old, bank, *, event=None):
    """Compare feasible, quadrature-resolved tracks against the optical prediction."""
    # Establish the calibrated optical reference before proposing physical tracks.
    baseline = pred["fractions"]
    base_count = baseline.shape[1] - 1
    if base_count == 0 or "1" not in bank:
        return (baseline, dict(reason="no reconstructable optical source", candidates=[]))
    if base_count == 4 and old.get("kept_released_fourth"):
        return (baseline, dict(selected=dict(family="released_fourth", count=4), candidates=[]))
    replay = frozen_shapes(pred["pmt_ids"], old, bank, baseline)
    if replay is None:
        return (baseline, dict(reason="no reconstructable optical source", candidates=[]))
    shapes, saved = replay
    ids, q, bg = (pred["pmt_ids"], pred["charges"], shapes[0])
    from wctering.likelihood.geometry import active_geometry

    active, _, _ = active_geometry()
    _, reference = profile(
        shapes, np.searchsorted(active, ids), q, np.asarray(saved["weights"]) * saved["amplitude"]
    )
    if not reference["qualified"]:
        return (baseline, dict(reason="unqualified optical reference", candidates=[]))
    dimension = (9 if old["fits"][old["selected_fit_index"]]["kind"] == "finite" else 8) * base_count
    if saved["geometry"].get("emission_model") == "full_water_line":
        dimension = 7
    prototype = coupled_model.Model(ids, q, bg, ("muon",), 4, 4)
    response = prototype.response
    normalization = np.log([response, max(float(q.sum()) * saved["weights"][0], 1e-08)])
    alternatives = [(0.0, baseline, dict(family="released", count=base_count))]
    audit = []
    # Try to localize a sparse fourth arc while the three optical shapes are fixed.
    if base_count == 3 and event is not None:
        from sparse_paths import fit

        args = (event, bank, saved["geometry"], shapes, np.asarray(reference["amplitudes"]), reference["ll"])
        trial = fit(
            *args,
            prior_fractions=baseline,
        )
        if trial is not None:
            ll, fractions, record = trial
            gain = float(2 * (ll - reference["ll"]) - 9 * np.log(len(active)))
            record.update(family="residual_finite_track", count=base_count + 1, gain=gain, supported=True)
            audit.append(record)
            alternatives.append((gain, fractions, record))
    # Generate and refine starts in count order. Several plausible modes can survive screening.
    proposals_by_count = {}
    qualified_tracks = []
    best_single = None
    fitted_starts = []
    single_track_modes = []
    top = max(base_count, 3)
    for count in range(1, top + 1):
        proposals_by_count[count] = charge_proposals(ids, q, bg, old, bank, count, normalization)
    for count, proposals in proposals_by_count.items():
        shortlist = proposals[:1]
        if count in (2, 3):
            from multi_start import allocate

            def coarse_fit(proposal):
                model = bare_model.Model(ids, q, bg, proposal.particles, 4, 16)
                x = proposal.coordinates
                _, z = minimize_scaled(model, x, 12, "chart", 18)
                if not feasible(model, z) or model.hardware_margin(z) <= 1e-05:
                    return None
                state = model(z, True)
                if state["log_joint"] < model(x, True)["log_joint"]:
                    z, state = (x, model(x, True))
                score = 2 * state["log_joint"] - model.dimension * np.log(len(model.active))
                return (score, counted(model, state).shape[1] - 1, z)

            endpoints = allocate(proposals, coarse_fit, max_modes=3)
            shortlist = [
                Proposal(p.particles, x, p.preliminary_score, f"{p.family}_screened") for p, x in endpoints
            ]
            if not shortlist:
                shortlist = proposals[:1]
        for proposal in shortlist:
            if count == base_count:
                fitted_starts.append(proposal)
            result = refine(ids, q, bg, proposal, chart="chart", maximum_evaluations=72)
            if result is None:
                continue
            model, state, record = result
            if count == 1 and (best_single is None or state["log_joint"] > best_single[1]["log_joint"]):
                best_single = (model, state)
            if len(model.particles) <= 4:
                qualified_tracks.append((model, state))
            add(alternatives, audit, model, state, record, reference["ll"], dimension)
            if base_count > 1 and record["count"] == 1 and record["supported"]:
                single_track_modes.append((*alternatives[-1], model, state))
    if best_single is not None:
        model, state = best_single
        for proposal in residual_proposals(model, state["parameters"])[:1]:
            result = refine(ids, q, bg, proposal, chart="chart", maximum_evaluations=72)
            if result is not None:
                add(alternatives, audit, *result, reference["ll"], dimension)
                if base_count > 1 and result[2]["count"] == 1 and result[2]["supported"]:
                    single_track_modes.append((*alternatives[-1], result[0], result[1]))
    if len(proposals_by_count.get(base_count, ())) > 1:
        same_gain = max((row[0] for row in alternatives if row[2]["count"] == base_count))
        if same_gain <= MARGIN:
            from multi_start import separate

            nominees = list(separate(proposals_by_count[base_count], 3))
            p = next(
                (
                    p
                    for p in nominees
                    if all(
                        (
                            p.particles != done.particles
                            or np.linalg.norm(p.coordinates - done.coordinates) > 0.2
                            for done in fitted_starts
                        )
                    )
                ),
                None,
            )
            if p is not None:
                retry_budget = 72
                coordinates = "chart_full"
                result = refine(ids, q, bg, p, chart=coordinates, maximum_evaluations=retry_budget)
                if result is not None:
                    m, s, r = result
                    r["family"] = "alternate_optical_start"
                    if counted(m, s).shape[1] == base_count + 1:
                        add(alternatives, audit, m, s, r, reference["ll"], dimension)
    # Extend the best physical parents at a localized interaction. Keep the
    # measured prompt support of the incumbent rings when testing a new one.
    if event is not None and qualified_tracks:
        from serial_refinement import prepend

        ranked_tracks = sorted(
            qualified_tracks,
            key=lambda r: 2 * r[1]["log_joint"] - r[0].dimension * np.log(len(active)),
            reverse=True,
        )

        def extensions(model, state):
            yield from prepend(event, model, state, movable=True)

        parents = [(m, s) for m, s in ranked_tracks if len(m.particles) < 4][:2]
        choices = sorted(
            (start for model, state in parents for start in extensions(model, state)),
            key=lambda r: r[0],
            reverse=True,
        )
        for score, particles, x in choices[:1]:
            proposal = Proposal(particles, x, score, "localized_interaction")
            result = refine(ids, q, bg, proposal, chart="chart", maximum_evaluations=72)
            if result is not None:
                model, state, record = result
                fractions = counted(model, state)
                supported = True
                from ring_supported import preserves_prompt

                incumbents = [(m, s) for m, s in ranked_tracks if counted(m, s).shape[1] == base_count + 1]
                if incumbents:
                    base_model, base_state = incumbents[0]
                    supported, lost = preserves_prompt(
                        event,
                        (base_model, base_state, counted(base_model, base_state)),
                        (model, state, fractions),
                    )
                else:
                    supported, lost = (fractions.shape[1] >= base_count + 1, [])
                record["lost_direct_hits_by_ring"] = lost
                if supported:
                    add(alternatives, audit, model, state, record, reference["ll"], dimension)
                    if base_count > 1 and record["count"] == 1 and record["supported"]:
                        single_track_modes.append((*alternatives[-1], model, state))
    # Protect the incumbent count unless the competing fit clears the evidence margin.
    incumbent = max((row for row in alternatives if row[2]["count"] == base_count), key=lambda row: row[0])
    if base_count > 1 and event is not None:
        from single_track_closure import test

        test(event, alternatives, single_track_modes, MARGIN)
    eligible = [
        row
        for row in alternatives
        if base_count == 1
        or row[2]["count"] > 1
        or row[2]["family"] == "finite_track"
        or row[2].get("closure_supported")
        or (row[2]["family"] == "timed_optical" and row[2]["prompt_supported"])
    ]
    if base_count == 1:
        eligible = [row for row in eligible if row[2]["count"] == 1 or row[2].get("prompt_supported")]
    winner = max(eligible, key=lambda row: row[0])
    if winner[2]["count"] != base_count and winner[0] <= incumbent[0] + MARGIN:
        winner = incumbent
    if winner[0] <= MARGIN:
        winner = alternatives[0]
    # Fit a residual arc jointly with the current sources. A same-count
    # improvement may change PE ownership through the soft light model.
    if event is not None and 1 <= base_count <= 2 and (winner[2]["count"] == base_count) and (len(q) >= 60):
        from residual_joint_optics import fit as fit_joint_residual

        corroboration = None
        if base_count == 1:
            from track_promotion import candidate_modes

            corroboration = partial(candidate_modes, alternatives, base_count + 1, winner[0], ids, q, bg)
        residual_incumbent = winner
        residual_alternatives = []
        for ll, fractions, record in fit_joint_residual(
            event,
            prototype,
            bank,
            saved["geometry"],
            shapes,
            np.asarray(reference["amplitudes"]),
            baseline,
            44,
            current=residual_incumbent[1],
            competing_modes=corroboration,
        ):
            count = fractions.shape[1] - 1
            gain = float(
                2.0 * (ll - reference["ll"]) + (dimension - 9 * (base_count + 1)) * np.log(len(active))
            )
            excess = 2.0 * (ll - record["null_ll"]) - record["penalty"]
            family = "joint_residual_optics" if count > base_count else "soft_light_partition"
            record.update(family=family, count=count, gain=gain, supported=True)
            audit.append(record)
            if gain > residual_incumbent[0] + MARGIN and excess > MARGIN:
                residual_alternatives.append((gain, fractions, record))
        for candidate in residual_alternatives:
            if candidate[2]["count"] > base_count and candidate[0] > winner[0] + MARGIN:
                winner = candidate
        if winner[2]["count"] == base_count:
            for candidate in residual_alternatives:
                if candidate[2]["count"] == base_count and candidate[0] > winner[0] + MARGIN:
                    winner = candidate
    # Test terminal and in-flight turns against physically continuous parents.
    if event is not None and winner[2]["count"] == 1 and (best_single is not None) and (len(q) >= 60):
        from terminal_track_seeds import evidence, starts as terminal_starts

        span_mode = "resolved_continuation"
        starts = terminal_starts(event, *best_single, variable_span=span_mode)
        for score, masses, x in starts[:2]:
            proposal = Proposal(masses, x, score, "terminal_charged_section")
            result = refine(ids, q, bg, proposal, chart="junction", maximum_evaluations=44)
            if result is None:
                continue
            model, state, record = result
            fractions = counted(model, state)
            count = fractions.shape[1] - 1
            gain = float(
                2.0 * (state["ll"] - reference["ll"]) + (dimension - model.dimension) * np.log(len(active))
            )
            measured = None
            diagnostic = {}
            if count == 2 and gain > winner[0] + MARGIN:
                measured = evidence(
                    event,
                    best_single,
                    model,
                    state,
                    fractions,
                    compare_soft=span_mode,
                    diagnostics=diagnostic,
                )
            record.update(
                count=count,
                gain=gain,
                supported=bool(measured),
                parameters=state["parameters"].tolist(),
                terminal_evidence=measured,
                terminal_diagnostic=diagnostic,
            )
            audit.append(record)
            if measured:
                winner = (gain, fractions, record)
    overlap_modes = len(audit) if winner[2]["count"] == 1 else None
    if event is not None and winner[2]["count"] == 1 and (best_single is not None):
        from long_track_split import evidence, starts as long_track_starts

        for score, masses, x in long_track_starts(event, *best_single):
            proposal = Proposal(masses, x, score, "long_track_split")
            result = refine(ids, q, bg, proposal, chart="junction", maximum_evaluations=56)
            if result is None:
                continue
            model, state, record = result
            fractions = counted(model, state)
            count = fractions.shape[1] - 1
            gain = float(
                2.0 * (state["ll"] - reference["ll"]) + (dimension - model.dimension) * np.log(len(active))
            )
            details = {}
            accepted = (
                count == 2
                and gain > winner[0] + MARGIN
                and evidence(
                    event,
                    best_single,
                    model,
                    state,
                    fractions,
                    details,
                    64,
                    comparison=audit,
                    timed=True,
                    separate_knockon=True,
                    resolved_turn=True,
                )
            )
            record.update(
                count=count,
                gain=gain,
                supported=accepted,
                parameters=state["parameters"].tolist(),
                split_evidence=details,
            )
            audit.append(record)
            if accepted:
                winner = (gain, fractions, record)
    # Use the same hits to compare clocks for overlapping one- and two-track fits.
    if (
        overlap_modes is not None
        and event is not None
        and (winner[2]["count"] == 1)
        and (best_single is not None)
    ):
        from timed_two_track import accept

        controls = [r for r in audit[overlap_modes:] if r["count"] == 1 and r.get("particles")]
        trial = accept(event, best_single, audit[:overlap_modes], winner[0], extra_controls=controls)
        if trial is not None:
            winner = trial
            audit.append(winner[2])
    # The last daughter search uses first-pulse timing to verify the residual
    # source. A longer extension needs independent evidence from the fit bank.
    if event is not None and winner[2]["count"] == 1 and (best_single is not None):
        from pulse_residual import extension_starts, qualify, starts

        forward = (
            (score, masses, x, "short_timed_daughter")
            for score, masses, x in starts(event, *best_single, batch_shapes=True)
        )

        def extension():
            witnessed = any(
                (
                    row["count"] == 2
                    and len(row.get("particles", ())) == 2
                    and ("electron" not in row["particles"])
                    and (row["gain"] > winner[0] + MARGIN)
                    for row in audit
                )
            )
            if witnessed:
                for score, masses, x in extension_starts(event, *best_single, batch_shapes=True):
                    yield (score, masses, x, "long_timed_daughter")

        for score, masses, x, family in chain(forward, extension()):
            proposal = Proposal(masses, x, score, family)
            result = refine(ids, q, bg, proposal, chart="junction", maximum_evaluations=40)
            if result is None:
                continue
            model, state, record = result
            fractions = counted(model, state)
            count = fractions.shape[1] - 1
            gain = float(
                2.0 * (state["ll"] - reference["ll"]) + (dimension - model.dimension) * np.log(len(active))
            )
            accepted, evidence = (False, {})
            if count == 2:
                accepted, evidence = qualify(
                    event,
                    best_single,
                    model,
                    state,
                    fractions,
                    winner[2],
                    (record["order"], 40),
                    compare_lineage="conversion",
                    require_long=family == "long_timed_daughter",
                )
            record.update(
                count=count,
                gain=gain,
                supported=accepted,
                timed_residual=evidence,
                parameters=state["parameters"].tolist(),
            )
            audit.append(record)
            if accepted:
                winner = (gain, fractions, record)
                break
    # Average compatible same-count fits to share charge at overlapping ridges.
    from soft_sharing import average

    fractions, weights = average(winner, alternatives, q)
    if weights:
        winner = (winner[0], fractions, dict(winner[2], sharing_weights=weights))
    # Recheck the winner's photon timing and direct support before releasing its charge.
    if event is not None and len(q) >= 30:
        from compare_count_models import fit as compare_ring_counts

        method = "module_predictive_review"
        result, reports = compare_ring_counts(
            event,
            prototype,
            pred,
            winner,
            audit,
            saved["geometry"],
            shapes,
            reference,
            2,
        )
        if reports:
            audit.extend(
                (
                    dict(
                        family=method,
                        count=report["rival_count"],
                        gain=report["score_gain"],
                        supported=report.get("accepted", False),
                        comparison=report,
                    )
                    for report in reports
                )
            )
        if result is not None:
            fractions, record, report = result
            revised = dict(record, selected_by=method, count_comparison=report)
            winner = (winner[0] + report["score_gain"], fractions, revised)
    # Topology-specific reviews use the same readout likelihood. These account
    # for stopped-pion light, measured beam apertures and returning tracks.
    if event is not None and base_count >= 2 and (winner[2]["family"] == "released"):
        from stopped_pion_photons import fit as fit_stopped_pion

        result, reports = fit_stopped_pion(
            event,
            prototype,
            saved["geometry"],
            shapes,
            np.asarray(reference["amplitudes"]),
            winner[1],
            34,
        )
        audit.extend(reports)
        if result is not None:
            gain, fractions, record = result
            winner = (winner[0] + gain, fractions, record)
    if event is not None and winner[2]["count"] == 1:
        from entrance_turn import fit as review_turn

        count_scores = (old.get("counter") or {}).get("class_scores")
        if not count_scores or sum(count_scores[2:]) >= 0.05:
            result, reports = review_turn(
                event,
                prototype,
                saved["geometry"],
                shapes,
                np.asarray(reference["amplitudes"]),
                bank,
                winner,
                best_single,
                38,
            )
            audit.extend(reports)
            if result is not None:
                gain, fractions, record = result
                winner = (winner[0] + gain, fractions, record)
    if event is not None and winner[2]["count"] == 3:
        from reassigned_optics import review as review_optical_bands

        result, reports = review_optical_bands(
            event,
            prototype,
            saved["geometry"],
            shapes,
            np.asarray(reference["amplitudes"]),
            baseline,
            winner,
            bank.get("2"),
            audit,
            65,
        )
        audit.extend(reports)
        if result is not None:
            gain, fractions, record = result
            winner = (winner[0] + gain, fractions, record)
    if event is not None and winner[2]["count"] == 3:
        from aperture_lepton_pair import review as review_lepton_pair

        result, reports = review_lepton_pair(
            event,
            prototype,
            saved["geometry"],
            shapes,
            np.asarray(reference["amplitudes"]),
            winner,
            bank.get("2"),
            68,
            bank_three=bank.get("3"),
        )
        audit.extend(reports)
        if result is not None:
            gain, fractions, record = result
            winner = (winner[0] + gain, fractions, record)
    if event is not None and winner[2]["count"] == 1:
        from window_turn import fit as review_window

        result, reports = review_window(
            event,
            prototype,
            saved["geometry"],
            shapes,
            np.asarray(reference["amplitudes"]),
            winner,
            120,
            bank=audit,
        )
        audit.extend(reports)
        if result is not None:
            gain, fractions, record = result
            winner = (winner[0] + gain, fractions, record)
    return (
        winner[1],
        dict(
            selected=winner[2],
            candidates=audit,
            score_gain=winner[0],
            same_count_gain=incumbent[0],
            initial_counter_scores=(old.get("counter") or {}).get("class_scores"),
            truth_used=False,
        ),
    )


def add(alternatives, audit, model, state, record, reference_ll, reference_dimension):
    fractions = counted(model, state)
    count = fractions.shape[1] - 1
    gain = 2 * (state["log_joint"] - reference_ll) + (reference_dimension - model.dimension) * np.log(
        len(model.active)
    )
    supported = 1 <= count <= 4 and np.all(np.bincount(fractions.argmax(1), minlength=count + 1)[1:])
    record.update(
        count=count, gain=float(gain), supported=bool(supported), parameters=state["parameters"].tolist()
    )
    audit.append(record)
    if supported:
        alternatives.append((gain, fractions, record))


def fit_event(raw, event_id, *, use_water_chord=True, early_candidate=None):
    """Run the qualified candidate search on one event in one numerical thread."""
    with threadpool_limits(limits=1), event_cache(64):
        begin = time.process_time()
        prediction, optical, bank, event = prepare(raw, event_id, use_water_chord=use_water_chord)
        prepared = dict(event=event, prediction=prediction, detail=optical, bank=bank)
        preparation_cpu = time.process_time() - begin
        begin = time.process_time()
        if optical.get("chord") is not None:
            fractions = prediction["fractions"]
            diagnostics = dict(selected=dict(family="coherent_track", count=1), chord=optical["chord"])
        else:
            from arc_render_cache import cached_arcs
            from charge_starts import fixed_recoil
            from geometry_cache import reuse_geometry
            from physical_pulse import clock_values
            from track_curve import scratch_curve
            from transport_panels import cached_panels

            # Exact evaluation reuse is scoped to this fit. The original
            # likelihood is restored if an optimizer or candidate fails.
            previous_kernel = readout.positive
            readout.positive = positive_without_source
            try:
                with (
                    cached_arcs(True),
                    fixed_recoil(True),
                    reuse_geometry(True),
                    clock_values(True),
                    cached_panels(True),
                    scratch_curve(True),
                ):
                    result = early_candidate(prediction, optical, bank, event) if early_candidate else None
                    if result is None:
                        result = search(prediction, optical, bank, event=event)
                    fractions, diagnostics = result
            finally:
                readout.positive = previous_kernel
        search_cpu = time.process_time() - begin
        np.testing.assert_allclose(fractions.sum(1), 1.0, atol=2e-14, rtol=0)
        if rejected := optical.get("ring_definition_rejected_counts"):
            diagnostics["ring_definition_rejected_optical_counts"] = rejected
        detail = dict(
            diagnostics,
            cpu_seconds=preparation_cpu + search_cpu,
            reconstruction_cpu_seconds=search_cpu,
        )
        return dict(prediction, fractions=fractions), detail, prepared
