"""Let one slowing trajectory absorb longitudinal shape error before counting."""

import numpy as np

from throughgoing import fit as fit_chord
from wctering.likelihood.geometry import common_background


def coherent_chord(line, span=2.1, *, relative=False):
    # A short track can illuminate part of a long allowed chord. Require a
    # measured span and prompt support before testing other trajectories.
    if line["coherent_coverage"] < 0.68 or line["chord_m"] < 2.5:
        return False
    observed = line["coherent_longitudinal_span_m"]
    if relative and relative != "hybrid":
        # The vertical and diagonal water chords have different lengths.
        # Require prompt occupancy near both physical ends in either case.
        entry, exit = line["chord_end_support"]
        return observed >= span * line["chord_m"] and entry >= 0.10 and exit >= 0.025
    return observed >= span


def test(event, bank, *, span=2.5, relative=False):
    """Return a detector-only one-track candidate and competing-cone evidence."""
    from chord_residual import residuals

    chord = fit_chord(event, bank)
    if chord is None:
        return None
    _, fractions, details = chord
    from chord_yield import fraction
    from wctering.likelihood.geometry import active_geometry

    _, positions, normals = active_geometry()
    ratio, expected = fraction(positions, normals, details)
    details.update(direct_yield_fraction=ratio, expected_direct_pe=expected)
    span, relative = chord_requirement(details, span, relative)
    if not coherent_chord(details, span, relative=relative):
        return (fractions, dict(chord=details, physical=None, competitors=[]))
    if details["direct_yield_fraction"] < 0.55:
        return (fractions, dict(chord=details, physical=None, competitors=[], skipped_low_direct_yield=True))
    bg = common_background(np.asarray(bank[1]["vertices"][0]))
    track = None
    from chord_profile import profile_chord

    shape = profile_chord(event, details, bg)
    if shape is None:
        return (fractions, dict(chord=details, profile_failed=True, competitors=[]))
    _, light, diagnostic = shape
    records = residuals(event, bank, details, background=bg, profiled=light)
    direction = np.asarray(details["axis"])
    anchor = np.asarray(details["anchor_m"])
    for row in records:
        delta = np.asarray(row["vertex_m"]) - anchor
        row["transverse_displacement_m"] = float(np.linalg.norm(delta - delta @ direction * direction))
    return (fractions, dict(chord=details, physical=track, competitors=records, profile=diagnostic))


def accepts_profile(record, span, *, relative=False):
    """Require enough observed extent and no well supported residual cone."""
    span, relative = chord_requirement(record["chord"], span, relative)
    if not record.get("profile") or not coherent_chord(record["chord"], span, relative=relative):
        return False
    if record["chord"]["direct_yield_fraction"] < 0.55:
        return False
    for row in record["competitors"]:
        if row["photon_support"] < 30 or row["hard_support"] < 18:
            continue
        if row["opening_deg"] > 42.5 and row["width_deg"] >= 8.0 and (row["relative_timing_gain"] < 40.0):
            continue
        displaced = row["axis_separation_deg"] >= 12.0 or row["transverse_displacement_m"] >= 0.2
        independent = row["gain"] >= 10.0 and row["relative_timing_gain"] >= 15.0
        strong_charge = row["gain"] >= 225.0 and row["relative_timing_gain"] >= -2.0
        if displaced and (independent or strong_charge):
            return False
    if relative:
        line = record["chord"]
        if min(line.get("longitudinal_quarter_fractions", [0.0])) < 0.07:
            return False
    return True


def chord_requirement(line, span, relative):
    """Allow the old absolute span only when it occupies most of the chord."""
    if relative == "hybrid":
        # The observed support is a 5–95% interval, so even a complete
        # trajectory can have unequal longitudinal quartile occupancies.
        # Use the relative end evidence only for short available water paths.
        return (2.5, False) if line["chord_m"] >= 3.2 else (span, True)
    return span, relative
