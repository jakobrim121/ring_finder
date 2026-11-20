"""In-memory fitting of a single event's aligned detector digit arrays."""

from contextlib import nullcontext

from .counting import DEFAULT_TRACK_COUNTING, TrackCounting, use_track_counting
from .engine import initialize
from .pipeline import reconstruct


PARTICLES = ("pion", "proton", "muon")
PMT_ID_SPACES = ("wcte", "wcsim")


def fit_hits(
    pmt_ids, charges_pe, times_ns, *, incoming_particle, event_id=0,
    pmt_id_space="wcte", prompt_window_ns=None, clock_decimals=None,
    min_track_pmts=DEFAULT_TRACK_COUNTING.min_track_pmts,
    track_merge_angle_deg=DEFAULT_TRACK_COUNTING.track_merge_angle_deg,
):
    """Return the prediction, optical hypotheses, and review diagnostics.

    Pass PMT IDs as 100*slot + zero-based position for WCTE, or zero-based
    tube IDs for WCSim. Use PE and ns. The default window is located around
    the prompt peak. A supplied [low, high) window is used as-is by the
    timing likelihood. No event selection is performed.

    Fitted track sections are counted using their effective PMT support.
    Adjacent sections can share one ring if their estimated turn is at
    most track_merge_angle_deg. Optical and timing evidence is also used
    to select the final ring hypothesis.
    """
    if incoming_particle not in PARTICLES:
        raise ValueError(f"Unsupported incoming particle {incoming_particle!r}")
    if pmt_id_space not in PMT_ID_SPACES:
        raise ValueError(f"Unsupported PMT ID space {pmt_id_space!r}")
    counting = TrackCounting(min_track_pmts, track_merge_angle_deg)
    initialize()
    import numpy as np

    raw = dict(raw_pmt_ids=pmt_ids, charges=charges_pe, times=times_ns,
               pmt_id_space=pmt_id_space, prompt_window_ns=prompt_window_ns,
               clock_decimals=clock_decimals)
    optics = nullcontext()
    if incoming_particle in ("pion", "proton"):
        from native_optics import enabled

        optics = enabled()
    with optics as acceleration, use_track_counting(counting):
        prediction, detail, comparison = reconstruct(raw, event_id, incoming_particle)
    if acceleration is not None:
        comparison["native_optics"] = acceleration
    if counting != DEFAULT_TRACK_COUNTING:
        comparison["track_counting"] = dict(
            min_track_pmts=counting.min_track_pmts,
            track_merge_angle_deg=counting.track_merge_angle_deg,
        )

    # Column zero is background, and each pulse's measured charge is conserved.
    fractions = prediction["fractions"]
    if not np.isfinite(fractions).all() or np.min(fractions, initial=0.0) < -1e-12:
        raise ArithmeticError("Invalid fitted charge shares")
    np.testing.assert_allclose(fractions.sum(1), 1.0, atol=1e-10, rtol=0.0)
    return dict(prediction, hard_labels=fractions.argmax(1),
                shared_charge_pe=prediction["charges"][:, None] * fractions), detail, comparison
