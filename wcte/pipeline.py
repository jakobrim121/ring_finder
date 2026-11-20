"""Choose physical ring hypotheses and refine their shared charge."""

from dataclasses import dataclass, field
from functools import partial
from time import process_time

from . import engine


CANDIDATE_ERRORS = (ArithmeticError, KeyError, ValueError)
INITIAL_REVIEWS = ("continuation", "kink", "fast_primary")
SINGLE_PRIMARY_REVIEWS = (*INITIAL_REVIEWS, "primary_repair", "lineage_turn")
PRIMARY_REVIEWS = (*SINGLE_PRIMARY_REVIEWS, "primary_multiring", "beam_reallocation")
COUNT_REVIEWS = (*PRIMARY_REVIEWS, "aperture_multiplicity", "count_review")
COHERENCE_GUARDS = (
    "kink",
    "primary_repair",
    "lineage_turn",
    "primary_multiring",
    "beam_reallocation",
    "retained_residual",
    "proton_vertex",
    "supported_second",
    "aperture_multiplicity",
)


@dataclass
class EventReview:
    """Working state for one event and the receipt for each attempted review."""

    prediction: dict
    detail: dict
    prepared: dict
    particle: str
    diagnostics: dict
    energy_history: dict = field(default_factory=dict)

    @property
    def count(self):
        # Column zero always belongs to background.
        return self.prediction["fractions"].shape[1] - 1

    def applied(self, *names):
        return any(self.diagnostics.get(name, {}).get("applied") for name in names)

    def attempt(self, name, review, *args, errors=CANDIDATE_ERRORS, **kwargs):
        """Commit a review's returned prediction only after it finishes."""
        start = process_time()
        try:
            self.prediction, report = review(*args, **kwargs)
        except errors as error:
            report = {"applied": False, "failure": str(error)}
        self.diagnostics[name] = dict(report, comparison_cpu_seconds=process_time() - start)

    def track_review(self, name, review, *args, **kwargs):
        self.attempt(
            name, review, self.prepared, self.prediction, self.detail, self.particle, *args, **kwargs
        )

    def result(self):
        return self.prediction, self.detail, self.diagnostics


def reconstruct(raw, event_id, particle):
    """Fit one digit event and return its prediction and diagnostics."""
    if particle == "muon":
        return _fit_muon(raw, event_id)
    if particle not in ("pion", "proton"):
        raise ValueError(f"Unsupported incoming particle {particle!r}")
    return _fit_beam(raw, event_id, particle, use_water_chord=True)


def _fit_beam(raw, event_id, particle, *, use_water_chord):
    from early_primary import probe

    # The early probe can replace the full search when its physical fit is decisive.
    fast_report = {}
    prediction, detail, prepared = engine.fit_base(
        raw,
        event_id,
        use_water_chord=use_water_chord,
        early_candidate=partial(probe, particle=particle, receipt=fast_report),
    )
    prepared["primary_models"] = {}
    if use_water_chord:
        prepared["keep_coherence_reference"] = True

    prediction, detail, continuation = engine.review_continuation(prepared, prediction, detail, particle)
    event = EventReview(
        prediction,
        detail,
        prepared,
        particle,
        {"continuation": continuation, "kink": {"applied": False}, "fast_primary": fast_report},
    )
    _review_early_history(event)
    _recover_primary(event, use_water_chord)
    before_count_review = event.prediction
    _review_multiplicity(event, before_count_review)
    _review_residuals(event)
    from charge_adjustment import review as adjust_charge

    event.attempt("charge_adjustment", adjust_charge, prepared, event.prediction, particle)
    if use_water_chord and detail.get("chord"):
        _reconsider_water_chord(event, raw, event_id)
    if use_water_chord and not event.applied("beam_chord_reconsideration"):
        _review_coherence(event)
    _protect_calibrated_third(event, before_count_review)
    return event.result()


def _fit_muon(raw, event_id):
    from muon_history_review import review

    prediction, detail, prepared = engine.fit_base(raw, event_id)
    event = EventReview(prediction, detail, prepared, "muon", {})
    event.attempt("muon_review", review, prepared, prediction, detail)
    return event.result()


def _review_early_history(event):
    """Compare two-leg histories or a short turn against the main candidate."""
    if event.applied("continuation"):
        return
    if event.count == 2:
        from energy_sharing import review

        event.track_review("energy_sharing", review, selected_history=event.energy_history)
    elif event.count == 1:
        from kink_review import review

        event.track_review("kink", review, errors=(ArithmeticError, ValueError))


def _recover_primary(event, use_water_chord):
    """Try entrance-light recovery while the earlier hypotheses are intact."""
    if not event.applied(*INITIAL_REVIEWS):
        from primary_repair_review import review

        event.track_review("primary_repair", review)
    if not event.applied(*INITIAL_REVIEWS, "primary_repair"):
        from lineage_turn_review import review

        event.track_review("lineage_turn", review)
    if not event.applied(*SINGLE_PRIMARY_REVIEWS):
        from multiring_primary_review import review

        event.track_review("primary_multiring", review)
        if not event.applied("primary_multiring"):
            from beam_prefix_reallocation import review as reallocate

            event.track_review("beam_reallocation", reallocate, chord_retry=not use_water_chord)


def _review_multiplicity(event, before):
    """Check rival counts and protect the fitted light of a third ring."""
    if not event.applied(*PRIMARY_REVIEWS):
        from aperture_multiplicity_review import review

        event.track_review(
            "aperture_multiplicity",
            review,
        )
    if not event.applied(*PRIMARY_REVIEWS, "aperture_multiplicity"):
        from calibrated_count_review import review

        aperture = event.diagnostics.get("aperture_multiplicity", {})
        protected = any(
            aperture.get(name)
            for name in (
                "protected_faint_section",
                "protected_old_core",
                "protected_count_margin",
            )
        )
        event.track_review(
            "count_review",
            review,
            engine.DATA / "count_review_model.json",
            allow_reduction=not protected,
        )
    if before["fractions"].shape[1] == 4 and event.count == 2:
        from third_ring_preservation import review

        event.attempt(
            "third_preservation",
            review,
            event.prepared,
            before,
            event.prediction,
            event.detail,
            event.particle,
            event.diagnostics,
        )


def _review_residuals(event):
    """Fit supported light left over from the chosen beam history."""
    if event.applied("continuation"):
        from retained_residual_review import review

        event.attempt(
            "retained_residual",
            review,
            event.prepared,
            event.prediction,
            event.detail,
            event.diagnostics["continuation"],
        )
    if not event.applied(*COUNT_REVIEWS):
        if event.particle == "proton" and event.count == 1:
            from supported_second_review import review

            event.track_review("supported_second", review)
        if not event.applied("supported_second"):
            from residual_extension_review import review

            event.track_review(
                "residual_extension",
                review,
                event.energy_history,
            )
    if event.count in (2, 3):
        from matched_soft_review import review

        event.track_review("matched_soft", review, event.diagnostics)
    if (
        event.particle == "proton"
        and event.count == 1
        and not event.applied("fast_primary", "aperture_multiplicity")
    ):
        from proton_vertex_residual import review

        event.attempt(
            "proton_vertex",
            review,
            event.prepared,
            event.prediction,
            event.detail,
        )


def _reconsider_water_chord(event, raw, event_id):
    from beam_chord_reconsideration import nominated, qualify

    if not nominated(event.detail, event.prediction):
        return

    def retry_and_compare():
        # The retry uses a finite track without starting a second coherence review.
        retry, detail, _ = _fit_beam(raw, event_id, event.particle, use_water_chord=False)
        prediction, report = qualify(event.prediction, retry, protected_primary=event.applied("continuation"))
        report["retried_family"] = (detail.get("selected") or {}).get("family")
        return prediction, report

    event.attempt("beam_chord_reconsideration", retry_and_compare)


def _review_coherence(event):
    """Check species-specific explanations at the count entering this stage."""
    count = event.count
    if count == 1 and event.particle == "proton":
        from aperture_partition_split import eligible, review

        if eligible(event.prediction, event.detail, event.particle):
            event.track_review("aperture_partition_split", review)
    if count in (2, 3) and not event.applied(*COHERENCE_GUARDS):
        if count == 2:
            from coherent_single_review import review

            event.track_review("coherence_review", review, event.diagnostics["continuation"])
        else:
            from aperture_same_count import review

            event.track_review("coherence_review", review)
    count_review_applied = event.applied(
        *COHERENCE_GUARDS,
        "continuation",
        "count_review",
        "residual_extension",
        "matched_soft",
        "coherence_review",
    )
    if count == 1 and event.particle == "pion":
        _review_pion_partners(event)
    if count == 2 and event.count == 2 and event.particle == "pion" and not count_review_applied:
        from retained_light_charged_kink import review

        event.track_review("retained_pion_kink", review)
    if count == 3 and not count_review_applied:
        from weak_third_review import review

        event.track_review("weak_third_review", review)


def _review_pion_partners(event):
    from pion_partner_review import eligible, review

    if eligible(event.prediction, event.detail, event.particle):
        event.track_review("pion_partner_review", review)
    if event.count == 1:
        from successive_pion_turns import eligible as can_test_turns, review as review_turns

        if can_test_turns(event.prediction, event.detail, event.particle):
            event.track_review("successive_pion_turns", review_turns)


def _protect_calibrated_third(event, before):
    from third_retention_calibrated import review

    event.prediction, report = review(
        event.prepared["event"],
        event.particle,
        before,
        event.prediction,
        event.diagnostics,
        engine.DATA / "retained_third_model",
    )
    if report is not None:
        event.diagnostics["retained_supported_third"] = report
