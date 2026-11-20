"""In-memory fitting of a single event's aligned detector digit arrays."""

from contextlib import nullcontext

from .engine import initialize
from .pipeline import reconstruct


PARTICLES = ("pion", "proton", "muon")
PMT_ID_SPACES = ("wcte", "wcsim")


def fit_hits(
    pmt_ids, charges_pe, times_ns, *, incoming_particle, event_id=0,
    pmt_id_space="wcte", prompt_window_ns=None, clock_decimals=None,
):
    """Return the prediction, optical hypotheses, and review diagnostics.

    Pass PMT IDs as 100*slot + zero-based position for WCTE, or zero-based
    tube IDs for WCSim. Use PE and ns. The default window is located around
    the prompt peak. A supplied [low, high) window is used as-is by the
    timing likelihood. No event selection is performed.
    """
    if incoming_particle not in PARTICLES:
        raise ValueError(f"Unsupported incoming particle {incoming_particle!r}")
    if pmt_id_space not in PMT_ID_SPACES:
        raise ValueError(f"Unsupported PMT ID space {pmt_id_space!r}")
    initialize()
    import numpy as np

    raw = dict(raw_pmt_ids=pmt_ids, charges=charges_pe, times=times_ns,
               pmt_id_space=pmt_id_space, prompt_window_ns=prompt_window_ns,
               clock_decimals=clock_decimals)
    optics = nullcontext()
    if incoming_particle in ("pion", "proton"):
        from native_optics import enabled

        optics = enabled()
    with optics as acceleration:
        prediction, detail, comparison = reconstruct(raw, event_id, incoming_particle)
    if acceleration is not None:
        comparison["native_optics"] = acceleration

    # Column zero is background, and each pulse's measured charge is conserved.
    fractions = prediction["fractions"]
    if not np.isfinite(fractions).all() or np.min(fractions, initial=0.0) < -1e-12:
        raise ArithmeticError("Invalid fitted charge shares")
    np.testing.assert_allclose(fractions.sum(1), 1.0, atol=1e-10, rtol=0.0)
    return dict(prediction, hard_labels=fractions.argmax(1),
                shared_charge_pe=prediction["charges"][:, None] * fractions), detail, comparison
