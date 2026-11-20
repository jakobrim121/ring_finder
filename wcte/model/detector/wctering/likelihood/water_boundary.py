"""Exact line-segment exclusion of the detector's truncated spherical domes.

For each dome, minimize max(norm(r)-R, cut-normal.r) along the segment.
Nonnegative minima mean that the entire emitting segment is outside the dome.
The minimum of this convex one-dimensional function is at an endpoint, the
radial minimum, or an intersection of its two analytic branches. Gradients use
the envelope theorem; a branch crossing uses the stationary convex combination.
All distances and gradients use metres. No detector measurements enter geometry.
"""

from pathlib import Path
import hashlib, json
import numpy as np
from wctering.seeding.detector import load_run_survey

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "assets/water/wcte_simulation.npz"
META = ROOT / "assets/water/wcte_simulation.json"
assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == json.loads(META.read_text())["sha256"]
with np.load(ASSET, allow_pickle=False) as z:
    CENTRES = z["water_exclusion_centres_mm"] / 1000
    survey = load_run_survey()
    if survey is not None:
        # A readout mask removes channels, while all surveyed domes remain physical.
        slots = z["water_exclusion_slots"]
        if set(survey.module_centers_mm) != set(slots):
            raise ValueError("The WCTE survey must include every physical mPMT dome")
        CENTRES = np.asarray([survey.module_centers_mm[int(slot)] for slot in slots]) / 1000
    NORMALS = z["water_exclusion_axes"].copy()
    RADIUS = float(z["water_exclusion_radius_mm"]) / 1000
    CUT = float(z["water_exclusion_cap_cut_mm"]) / 1000


def clearance(center, axis, length, centres=CENTRES, normals=NORMALS, radius=RADIUS, cut=CUT):
    """Return minimum margin and derivatives wrt center, unit axis and length."""
    c = np.asarray(center, float)
    a = np.asarray(axis, float)
    d = c - centres
    if length == 0.0:
        # A point has no interior segment extrema. The same envelope derivative
        # still handles an exact tie of the spherical and planar boundaries.
        distance = np.linalg.norm(d, axis=1)
        radial = distance - radius
        planar = cut - np.einsum("ij,ij->i", d, normals)
        unit = np.divide(d, distance[:, None], out=np.zeros_like(d), where=distance[:, None] > 1e-15)
        alpha = (radial > planar).astype(float)
        tie = abs(radial - planar) <= 2e-12
        if tie.any():
            parallel = normals @ a
            denominator = unit @ a + parallel
            stationary = np.divide(
                parallel,
                denominator,
                out=np.full(len(d), 0.5),
                where=abs(denominator) > 1e-14,
            )
            alpha[tie] = np.clip(stationary[tie], 0.0, 1.0)
        dc = alpha[:, None] * unit - (1 - alpha[:, None]) * normals
        return np.maximum(radial, planar), dc, np.zeros_like(dc), -0.5 * abs(dc @ a)
    b = d @ a
    e = normals @ a
    h = radius + cut - np.einsum("ij,ij->i", d, normals)
    aa = np.maximum(0.0, 1 - e * e)
    bb = 2 * (b + h * e)
    cc = np.einsum("ij,ij->i", d, d) - h * h
    disc = bb * bb - 4 * aa * cc
    root = np.sqrt(np.maximum(0.0, disc))
    q = -0.5 * (bb + np.where(bb < 0, -root, root))
    r1 = np.divide(q, aa, out=np.full(len(d), np.nan), where=aa > 1e-12)
    r2 = np.divide(cc, q, out=np.full(len(d), np.nan), where=abs(q) > 1e-30)
    linear = aa <= 1e-12
    r1[linear] = np.divide(
        -cc[linear],
        bb[linear],
        out=np.full(linear.sum(), np.nan),
        where=abs(bb[linear]) > 1e-14,
    )
    r2[linear] = np.nan
    candidates = np.column_stack(
        (
            np.full(len(d), -0.5 * length),
            np.full(len(d), 0.5 * length),
            np.clip(-b, -0.5 * length, 0.5 * length),
            r1,
            r2,
        )
    )
    valid = np.isfinite(candidates) & (candidates >= -0.5 * length) & (candidates <= 0.5 * length)
    valid[:, 3:] &= (disc[:, None] >= 0) & (h[:, None] - e[:, None] * candidates[:, 3:] >= 0)
    # NaN roots are never selected; finite substitutes avoid contaminating arrays.
    candidates = np.where(valid, candidates, 0.0)
    relative = d[:, None, :] + candidates[:, :, None] * a
    distance = np.linalg.norm(relative, axis=2)
    radial = distance - radius
    planar = cut - np.einsum("mti,mi->mt", relative, normals)
    values = np.where(valid, np.maximum(radial, planar), np.inf)
    ix = values.argmin(axis=1)
    rows = np.arange(len(d))
    s = candidates[rows, ix]
    # Exactly parallel to the flat cap base, the planar branch can attain its
    # minimum on an interval. Its midpoint gives the symmetric generalized
    # derivative; an arbitrary endpoint gives a biased one-sided derivative.
    plateau_disc = h * h - (np.einsum("ij,ij->i", d, d) - b * b)
    half = np.sqrt(np.maximum(0.0, plateau_disc))
    plo = np.maximum(-0.5 * length, -b - half)
    phi = np.minimum(0.5 * length, -b + half)
    flat = (abs(e) <= 1e-14) & (h >= 0) & (plateau_disc >= 0) & (phi > plo + 1e-12)
    flat &= abs(values[rows, ix] - (h - radius)) <= 2e-12
    s = np.where(flat, 0.5 * (plo + phi), s)
    v = d + s[:, None] * a
    dist = np.linalg.norm(v, axis=1)
    rv = dist - radius
    pv = cut - np.einsum("ij,ij->i", v, normals)
    u = np.divide(v, dist[:, None], out=np.zeros_like(v), where=dist[:, None] > 1e-15)
    alpha = (rv > pv).astype(float)
    tie = abs(rv - pv) <= 2e-12
    denom = u @ a + e
    stationary = np.divide(e, denom, out=np.full(len(d), 0.5), where=abs(denom) > 1e-14)
    alpha[tie] = np.clip(stationary[tie], 0.0, 1.0)
    dc = alpha[:, None] * u - (1 - alpha[:, None]) * normals
    da = s[:, None] * dc
    endpoint = np.where(ix == 0, -0.5, np.where(ix == 1, 0.5, 0.0))
    endpoint[flat] = 0.0
    dl = endpoint * (dc @ a)
    if length == 0:
        dl = -0.5 * abs(dc @ a)
    return values[rows, ix], dc, da, dl
