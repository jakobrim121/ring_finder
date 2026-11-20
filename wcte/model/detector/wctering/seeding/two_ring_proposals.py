"""Force two-cone partitions, then check alternate vertices and coherence.

The counter needs the best H2 explanation even when H1 eventually wins. These
functions construct that explanation; they do not make the final count choice."""

from __future__ import annotations


import numpy as np


from wctering.seeding.cone_seeds import (
    _best_coherence_cone,
    _best_localized_cone,
    _fit_vertex_t0,
    _ring_params,
    fit_dominant_sharp,
)
from wctering.seeding.event_cache import _cached_call
from wctering.seeding.seed_constants import (
    _COH_BG,
    _COH_GAP_BAND,
    _COH_GAP_MAX,
    _COH_GAP_REL,
    _COH_MARGIN,
    _COH_SUPPRESS_K,
    _DV_BAND,
    _DV_GMAX,
    _DV_GMIN,
    _DV_GRID,
    _DV_RMS_MAX,
    _DV_THC_HI,
    _DV_THC_LO,
    _DV_Y_BOT,
    _DV_Y_TOP,
    _GATE_COV2,
    _GATE_DOMFRAC,
    _GATE_SPR2C,
    _GATE_STD_MAXTHC,
    _GATE_THC2,
    _GATE_THC2_OWN,
    _PEEL_MARGIN,
    _QUAL_BAND,
    _QUAL_DISC,
    _QUAL_MARGIN,
    _QUAL_PHYS,
    _SIGMA_THETA,
    _THETA_C_MAX,
    _VG,
)


def _peel_sharp_labels(pos, t, q, K, margin=_PEEL_MARGIN, nmin=5):
    """Peeling that uses fit_dominant_sharp for each peeled ring. Returns labels in 1..K."""
    N = len(pos)
    labels = np.zeros(N, int)
    rem = np.ones(N, bool)
    for k in range(int(K) - 1):
        if rem.sum() < 2 * nmin:
            break
        R = fit_dominant_sharp(pos[rem], t[rem], q[rem])
        d = pos[rem] - R["vertex"]
        dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        ang = np.degrees(np.arccos(np.clip((d / dn[:, None]) @ R["direction"], -1, 1)))
        on = np.abs(ang - R["theta_c"]) < margin
        idx = np.where(rem)[0][on]
        if len(idx) < nmin:
            break
        labels[idx] = k + 1
        rem[idx] = False
    if (labels > 0).any():
        labels[rem] = labels.max() + 1
    else:
        labels[:] = 1
    out = np.zeros(N, int)
    for j, lv in enumerate(sorted(set(labels[labels > 0].tolist())), 1):
        out[labels == lv] = j
    return out


def _coherence_two_ring(uh, q, K=_COH_SUPPRESS_K):
    """Alternative 2-ring partition for a FAINT second ring under a bright one + noise.

    Fits the bright dominant cone, suppresses it (drops hits within K of its own
    radial spread), then finds the second ring with the tightness-weighted coherence
    finder so a weak coherent ring is not out-voted by the dominant ring's tail or by
    noise. Every hit is then assigned to the nearer of the two cones.

    Returns labels (1=dominant, 2=second) or None if no usable second cone. This is
    NOT applied unconditionally: it is worse than the standard peel/competitive path
    on comparable-ring events, so partition_two_rings only adopts it when it wins a
    noise-robust likelihood by a margin (_COH_MARGIN).
    """
    uh = np.asarray(uh, float)
    q = np.asarray(q, float)
    ax1, thc1 = _best_localized_cone(uh, q)
    ang1 = np.degrees(np.arccos(np.clip(uh @ ax1, -1, 1)))
    near = np.abs(ang1 - thc1) < 18.0
    spr1 = float(np.sqrt(np.average((ang1[near] - thc1) ** 2, weights=q[near]))) if near.sum() > 3 else 8.0
    keep = np.abs(ang1 - thc1) >= K * max(spr1, 5.0)  # suppress the dominant ring (incl. tail)
    if keep.sum() < 12:
        return None
    ax2, thc2 = _best_coherence_cone(uh[keep], q[keep])
    ang2 = np.degrees(np.arccos(np.clip(uh @ ax2, -1, 1)))
    return np.where(np.abs(ang1 - thc1) <= np.abs(ang2 - thc2), 1, 2)


def _coherence_loglik(q, uh, labels, bg=_COH_BG, sigma=_SIGMA_THETA):
    """Shared-vertex angular two-cone mixture log-lik with a uniform background floor.

    Identical in spirit to _shared_vertex_loglik but adds a constant background `bg`
    to the per-hit mixture density. The floor lets diffuse noise be explained by the
    background instead of forcing a ring to WIDEN to cover it -- without it the plain
    likelihood rewards a diffuse cone that soaks up noise and mis-ranks a correct
    tight-ring partition below a wrong diffuse one. Used ONLY to decide whether the
    coherence partition should override the standard one; kept separate so the
    peel-vs-competitive selection (_shared_vertex_loglik) is left untouched.
    """
    labels = np.asarray(labels)
    ks = sorted(set(labels[labels > 0].tolist()))
    if len(ks) < 2:
        return -np.inf
    Q = q.sum() + 1e-9
    comp = []
    for k in ks:
        m = labels == k
        ax = np.average(uh[m], 0, weights=q[m])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
        thc = float(np.average(ang[m], weights=q[m]))
        comp.append((q[m].sum() / Q) * np.exp(-0.5 * ((ang - thc) / sigma) ** 2))
    return float((q * np.log(np.sum(comp, 0) + bg)).sum())


def _coherence_max_gap(uh, q, labels, band=_COH_GAP_BAND):
    """Largest azimuthal gap [deg] over the rings of a partition (worst ring).

    For each ring, the on-ring hits (within `band` of its fitted cone) are projected to
    azimuth around the ring axis; the largest angular gap between consecutive occupied
    azimuths (with wraparound) measures how incomplete the observed ring is. A large
    gap can indicate a spurious partial arc, but detector geometry can also truncate
    a real ring. This is a diagnostic feature, not a universal physical veto. On-ring hits
    are used rather than all assigned hits, so off-ring contamination cannot fill and mask
    the gap. Returns the maximum gap across rings (360 if a ring is too sparse to judge).
    """
    labels = np.asarray(labels)
    ks = sorted(set(labels[labels > 0].tolist()))
    worst = 0.0
    for k in ks:
        m = labels == k
        if q[m].sum() <= 0 or m.sum() < 4:
            continue
        ax = np.average(uh[m], 0, weights=q[m])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
        thc = float(np.average(ang[m], weights=q[m]))
        sel = m & (np.abs(ang - thc) < band)
        if sel.sum() < 4:
            return 360.0
        a0 = np.array([1, 0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0, 1, 0.0])
        e1 = np.cross(ax, a0)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        phi = np.sort(np.degrees(np.arctan2(uh[sel] @ e2, uh[sel] @ e1)) % 360)
        gap = float(np.diff(np.concatenate([phi, [phi[0] + 360]])).max())
        worst = max(worst, gap)
    return worst


def _partition_quality(uh, q, labels, band=_QUAL_BAND, thc_phys=_QUAL_PHYS, disc=_QUAL_DISC):
    """Charge-explained-on-physical-tight-rings score for a 2-ring partition.

    For each labelled ring, take the charge lying within `band` of its fitted cone,
    weight it by the ring's azimuthal completeness (fraction of 12 sectors occupied),
    and DISCOUNT it if the cone is unphysically wide (theta_c > thc_phys, beyond the
    Cherenkov maximum). Summed over rings and normalized by total charge.

    Unlike the angular likelihood, this rewards partitions whose rings are tight,
    azimuthally-complete, PHYSICAL circles and refuses to credit a diffuse cone that
    only exists to drape over leakage and noise -- so it ranks a clean two-ring split
    above one carrying a junk cone, where the likelihood (which the noise floor cannot
    fully rescue) does not. Used as an additional, likelihood-independent trigger for
    the coherence override. Higher is better; ~[0,1].
    """
    labels = np.asarray(labels)
    ks = sorted(set(labels[labels > 0].tolist()))
    if len(ks) < 2:
        return -1e9
    Q = q.sum() + 1e-9
    tot = 0.0
    for k in ks:
        m = labels == k
        if q[m].sum() <= 0:
            continue
        ax = np.average(uh[m], 0, weights=q[m])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
        thc = float(np.average(ang[m], weights=q[m]))
        sel = (np.abs(ang - thc) < band) & m
        if sel.sum() >= 3:
            a0 = np.array([1, 0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0, 1, 0.0])
            e1 = np.cross(ax, a0)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(ax, e1)
            phi = np.degrees(np.arctan2(uh[sel] @ e2, uh[sel] @ e1)) % 360
            cov = sum(1 for s in range(12) if ((phi >= s * 30) & (phi < (s + 1) * 30)).any()) / 12.0
        else:
            cov = 0.0
        w = 1.0 if thc <= thc_phys else disc
        tot += w * (q[sel].sum() / Q) * cov
    return tot


def _coherence_gate(pos, t, q, uh):
    """Structural + physicality gate: is the coherence faint-ring partition trustworthy?

    Fires only when the coherence finder returns a genuinely dominant ring plus a tight,
    azimuthally-complete, well-separated faint ring -- AND that faint ring, refit at its
    OWN vertex, has a physical Cherenkov angle (<= _GATE_THC2_OWN). The own-vertex check
    is the crux: a ghost ring (hits that only align into a circle when viewed from the
    contaminated all-hit vertex) refits to an unphysically wide cone and is rejected,
    while a real faint ring stays physical. This lets the override fire on true faint
    rings the likelihood and quality scores miss, without firing on ghosts.

    Catches the case the likelihood rewards backwards (a diffuse standard cone draped
    over leakage+noise). Returns True/False; the caller supplies the standard `lab`
    separately for the near-junk-cone precondition.
    """
    ax1, thc1 = _best_localized_cone(uh, q)
    ang1 = np.degrees(np.arccos(np.clip(uh @ ax1, -1, 1)))
    near = np.abs(ang1 - thc1) < 18.0
    spr1 = float(np.sqrt(np.average((ang1[near] - thc1) ** 2, weights=q[near]))) if near.sum() > 3 else 8.0
    keep = np.abs(ang1 - thc1) >= _COH_SUPPRESS_K * max(spr1, 5.0)
    if keep.sum() < 12:
        return False
    ax2, thc2 = _best_coherence_cone(uh[keep], q[keep])
    ang2 = np.degrees(np.arccos(np.clip(uh @ ax2, -1, 1)))
    lab = np.where(np.abs(ang1 - thc1) <= np.abs(ang2 - thc2), 1, 2)
    if (lab == 2).sum() < 8 or q[lab == 1].sum() <= 0:
        return False
    domfrac = q[lab == 1].sum() / (q[lab > 0].sum() + 1e-9)
    on2 = np.abs(ang2 - thc2) < 10.0
    if on2.sum() < 3:
        return False
    spr2c = float(np.sqrt(np.average((ang2[on2] - thc2) ** 2, weights=q[on2])))
    a0 = np.array([1, 0, 0.0]) if abs(ax2[0]) < 0.9 else np.array([0, 1, 0.0])
    e1 = np.cross(ax2, a0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax2, e1)
    phi = np.degrees(np.arctan2(uh[on2] @ e2, uh[on2] @ e1)) % 360
    cov2 = sum(1 for s in range(12) if ((phi >= s * 30) & (phi < (s + 1) * 30)).any())
    # own-vertex physicality of the faint ring
    m2 = lab == 2
    Vr, _ = _fit_vertex_t0(pos[m2], t[m2], q[m2])
    dr = pos[m2] - Vr
    uhr = dr / np.maximum(np.linalg.norm(dr, axis=1), 1e-9)[:, None]
    axr = np.average(uhr, 0, weights=q[m2])
    axr /= np.linalg.norm(axr)
    angr = np.degrees(np.arccos(np.clip(uhr @ axr, -1, 1)))
    thc2_own = float(np.average(angr, weights=q[m2]))
    return bool(
        domfrac >= _GATE_DOMFRAC
        and thc2 <= _GATE_THC2
        and cov2 >= _GATE_COV2
        and spr2c <= _GATE_SPR2C
        and thc2_own <= _GATE_THC2_OWN
    )


def _competitive_shared_vertex(pos, q, uh, seed_labels, niter=8, cap=_THETA_C_MAX):
    """Competitive nearest-cone reassignment with a FIXED shared vertex.


    `uh` are the unit directions of every hit from ONE common (well-conditioned,
    all-hit) vertex. Each iteration refits only the axis (charge-weighted mean
    direction) and theta_c (charge-weighted mean opening angle) of each ring, then
    reassigns every hit to the nearer cone. Holding the vertex fixed removes the
    per-ring vertex runaway that made the independent-vertex version numerically
    unstable.

    JUNK-CONE REPAIR: if a ring's mean-direction fit yields an unphysical opening
    angle (theta_c > cap), the assigned hit-set is diffuse/contaminated and the cone
    is meaningless -- it would otherwise lose every hit to the other cone and let the
    other cone over-claim. We re-seed it with the brightest localized cone among its
    own hits (_best_localized_cone), keeping every cone physical. Labels 1..2.
    """
    lab = np.asarray(seed_labels).copy()
    if len(set(lab[lab > 0].tolist())) < 2:
        return lab

    def cone(m):
        ax = np.average(uh[m], 0, weights=q[m])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
        thc = float(np.average(ang[m], weights=q[m]))
        if thc > cap:  # unphysical -> re-seed a real localized cone
            ax, thc = _best_localized_cone(uh[m], q[m], cap)
        return ax, thc

    a1, c1 = cone(lab == 1)
    a2, c2 = cone(lab == 2)
    for _ in range(niter):
        an1 = np.degrees(np.arccos(np.clip(uh @ a1, -1, 1)))
        an2 = np.degrees(np.arccos(np.clip(uh @ a2, -1, 1)))
        nl = np.where(np.abs(an1 - c1) <= np.abs(an2 - c2), 1, 2)
        if np.array_equal(nl, lab) or (nl == 1).sum() < 5 or (nl == 2).sum() < 5:
            break
        lab = nl
        a1, c1 = cone(lab == 1)
        a2, c2 = cone(lab == 2)
    return lab


def _competitive_shared_vertex_k(
    pos,
    q,
    uh,
    seed_labels,
    n_rings,
    niter=8,
    cap=_THETA_C_MAX,
):
    """K-cone version of the frozen shared-vertex competitive refinement.

    The two-cone routine above is left untouched for numerical compatibility.
    This extension applies the same axis/opening-angle refit and nearest-cone
    reassignment to a fixed number of components at one shared vertex.
    """

    lab = np.asarray(seed_labels, dtype=int).copy()
    expected = list(range(1, int(n_rings) + 1))
    if sorted(set(lab[lab > 0].tolist())) != expected:
        return lab

    def cone(mask):
        axis = np.average(uh[mask], 0, weights=q[mask])
        axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
        angles = np.degrees(np.arccos(np.clip(uh @ axis, -1, 1)))
        theta_c = float(np.average(angles[mask], weights=q[mask]))
        if theta_c > cap:
            axis, theta_c = _best_localized_cone(uh[mask], q[mask], cap)
        return axis, theta_c

    cones = [cone(lab == label) for label in expected]
    for _ in range(int(niter)):
        residuals = np.column_stack(
            [np.abs(np.degrees(np.arccos(np.clip(uh @ axis, -1, 1))) - theta_c) for axis, theta_c in cones]
        )
        proposed = np.argmin(residuals, axis=1).astype(int) + 1
        if any(np.count_nonzero(proposed == label) < 5 for label in expected):
            break
        if np.array_equal(proposed, lab):
            break
        lab = proposed
        cones = [cone(lab == label) for label in expected]
    return lab


def _shared_vertex_loglik(pos, q, uh, labels, sigma=_SIGMA_THETA):
    """Angular two-cone mixture log-likelihood at the shared vertex (charge-weighted).

    Cones (axis, theta_c) are refit from `uh` per labelled ring; the per-hit mixture
    density (charge-prior x angular Gaussian) is summed in log. Used to choose
    between the sharp peel and the shared-vertex competitive partition. Stable
    because it uses the same fixed vertex as the competitive step.
    """
    ks = sorted(set(labels[labels > 0].tolist()))
    if len(ks) < 2:
        return -np.inf
    Q = q.sum() + 1e-9
    comp = []
    for k in ks:
        m = labels == k
        ax = np.average(uh[m], 0, weights=q[m])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
        thc = float(np.average(ang[m], weights=q[m]))
        comp.append((q[m].sum() / Q) * np.exp(-0.5 * ((ang - thc) / sigma) ** 2) + 1e-9)
    return float((q * np.log(np.sum(comp, 0))).sum())


def _ring_params_shared(pos, t, q, V, cap=_THETA_C_MAX):
    """Ring parameters fit at a FIXED shared vertex V, with a physical theta_c guard.

    The two rings of an event share one interaction vertex, so fitting both at the
    common all-hit vertex is more correct than independent per-ring vertex fits, and
    it removes the per-ring degeneracy that produced unphysical 60-80 deg cones in
    the returned dicts. If the charge-weighted opening angle still exceeds cap (the
    ring's hits span a wide arc, so no tight cone exists at this vertex), the axis is
    re-seeded to the brightest localized cone so the returned theta_c is always
    physical. A theta_c pinned exactly at cap is therefore a flag that the ring is
    poorly constrained -- the label partition is the reliable output, not this cone.
    """
    w = q + 1e-9
    d = pos - V
    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    uh = d / dn[:, None]
    ax = np.average(uh, 0, weights=w)
    ax /= np.linalg.norm(ax)
    ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
    thc = float(np.average(ang, weights=w))
    if thc > cap:  # wide-arc ring -> re-seed a physical localized cone
        ax, thc = _best_localized_cone(uh, w, cap)
    t0 = float(np.average(t - dn / _VG, weights=w))
    dt = t - t0 - dn / _VG
    sigt = max(
        float(np.sqrt(np.average((dt - np.average(dt, weights=w)) ** 2, weights=w))),
        1.0,
    )
    return dict(
        vertex=np.asarray(V, float),
        t0=t0,
        direction=ax,
        theta_c=thc,
        sigma_t=min(sigt, 5.0),
        charge=float(w.sum()),
    )


def _dv_region(pos):
    y = pos[:, 1]
    return np.where(y > _DV_Y_TOP, 1, np.where(y < _DV_Y_BOT, 2, 0))


def _dv_cone(pos, V, m, q):
    d = pos - V
    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    uh = d / dn[:, None]
    ax = np.average(uh[m], 0, weights=q[m])
    ax /= np.linalg.norm(ax)
    ang = np.degrees(np.arccos(np.clip(uh @ ax, -1, 1)))
    thc = float(np.average(ang[m], weights=q[m]))
    return ax, thc, ang


def _dv_bandq(ang, thc, q, m, band=_DV_BAND):
    """On-cone charge fraction within `band`, discounted if the cone is unphysical."""
    sel = m & (np.abs(ang - thc) < band)
    if sel.sum() < 3:
        return 0.0
    return (q[sel].sum() / (q[m].sum() + 1e-9)) * (1.0 if thc <= _QUAL_PHYS else 0.3)


def _displaced_vertex_recover(pos, t, q, lab, V):
    """Gated two-vertex recovery. Returns (labels, displaced_ring_label, displaced_vertex).

    If the gate does not fire, returns the input labels unchanged with (None, None).
    See the section header for the (important) caveats: the gate is experimental and
    transfer-sensitive. Operates only on a 2-ring labelling at a shared vertex V.
    """
    lab = np.asarray(lab).copy()
    ks = sorted(set(lab[lab > 0].tolist()))
    if len(ks) != 2:
        return lab, None, None
    reg = _dv_region(pos)
    # pick the localized (top/bottom-endcap-dominant) ring as the displaced candidate
    endk = None
    bestf = 0.0
    for k in ks:
        mk = lab == k
        if q[mk].sum() <= 0:
            continue
        rq = [q[mk & (reg == rr)].sum() for rr in (0, 1, 2)]
        dom = int(np.argmax(rq))
        df = rq[dom] / (q[mk].sum() + 1e-9)
        if dom != 0 and df > bestf:
            bestf = df
            endk = k
    if endk is None:
        return lab, None, None
    bark = ks[1] if endk == ks[0] else ks[0]
    ax1, thc1, ang1 = _dv_cone(pos, V, lab == bark, q)  # barrel ring: fixed at shared V
    axe, the0, ane0 = _dv_cone(pos, V, lab == endk, q)  # displaced ring: search direction
    me = lab == endk
    drs = pos[me] - V
    dns = np.maximum(np.linalg.norm(drs, axis=1), 1e-9)
    ress = t[me] - dns / _VG
    ress = ress - np.average(ress, weights=q[me])
    rms_sh = float(np.sqrt(np.average(ress**2, weights=q[me])))
    q0 = _dv_bandq(ang1, thc1, q, lab == bark) + _dv_bandq(ane0, the0, q, lab == endk)
    best = None
    for s in _DV_GRID:
        Ve = V + s * axe
        cur = lab.copy()
        valid_assignment = True
        for _ in range(4):
            displaced = cur == endk
            if np.count_nonzero(displaced) == 0 or float(np.sum(q[displaced])) <= 0.0:
                valid_assignment = False
                break
            _a2, c2, an2 = _dv_cone(pos, Ve, displaced, q)
            new = np.where(np.abs(ang1 - thc1) <= np.abs(an2 - c2), bark, endk)
            if np.array_equal(new, cur):
                cur = new
                break
            cur = new
        if not valid_assignment:
            continue
        if (cur == bark).sum() < 5 or (cur == endk).sum() < 5:
            continue
        _a2, c2, an2 = _dv_cone(pos, Ve, cur == endk, q)
        Qv = _dv_bandq(ang1, thc1, q, cur == bark) + _dv_bandq(an2, c2, q, cur == endk)
        if best is None or Qv > best[0]:
            best = (Qv, float(s), c2, cur.copy(), Ve)
    if best is None:
        return lab, None, None
    Qv, s, thc2, cur, Ve = best
    gain = Qv - q0
    fire = (
        _DV_GRID[0] < s < _DV_GRID[-1]  # interior optimum (reject runaway to edge)
        and _DV_THC_LO <= thc2 <= _DV_THC_HI  # recovered cone is physical
        and _DV_GMIN <= gain <= _DV_GMAX  # real but not gamed quality gain
        and rms_sh < _DV_RMS_MAX
    )  # shared-vertex timing not already terrible
    if fire:
        return cur, endk, Ve
    return lab, None, None


def _partition_two_rings_uncached(pos, t, q, margin=_PEEL_MARGIN):
    """Robust 2-ring partition (see section header). Returns (labels in 1..K, ring dicts).

    Sharp-dominant peel for a stable seed, then a competitive reassignment that
    holds ONE shared vertex fixed (so it cannot run away as the independent-vertex
    version did). The two candidate partitions -- sharp peel and shared-vertex
    competitive -- are scored by a shared-vertex angular mixture likelihood and the
    higher one is returned. This recovers the hard overlapping-comparable events
    (e.g. ev54) the peel alone cannot split, while staying reproducible.

    COHERENCE OVERRIDE (faint second ring + noise): the above path fails when one
    ring is bright and wide and the second is faint and on a well-separated axis amid
    heavy noise (e.g. 500 MeV ev8) -- it parks a diffuse cone in the faint ring's
    neighborhood and sweeps up the bright ring's tail and noise instead of finding
    the faint ring's own axis. As an extra candidate we suppress the dominant ring
    and find the second with a tightness-weighted coherence finder
    (_coherence_two_ring). Because that candidate is worse on comparable-ring events,
    it is adopted ONLY when it beats the standard partition on a NOISE-ROBUST
    likelihood (_coherence_loglik, background floor) by a margin (_COH_MARGIN), so
    noise cannot make a diffuse cone win. Validated to fix ev8 (0.60->0.84) with zero
    regressions on the 300 MeV 2-ring set (mean 0.818->0.835).

    Returned ring-parameter dicts are fit at the shared vertex with a physical
    theta_c guard (_ring_params_shared), so every returned cone is <= _THETA_C_MAX;
    a theta_c pinned at the cap flags a poorly-constrained (wide-arc) second ring.
    """
    pos = np.asarray(pos, float)
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    N = len(pos)
    lab = _peel_sharp_labels(pos, t, q, 2, margin=margin)
    V = None
    dv_endk = None
    dv_V = None
    if len(set(lab[lab > 0].tolist())) >= 2:
        V, _t0 = _fit_vertex_t0(pos, t, q)  # one well-conditioned shared vertex
        d = pos - V
        dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        uh = d / dn[:, None]
        lab_c = _competitive_shared_vertex(pos, q, uh, lab)
        if len(set(lab_c[lab_c > 0].tolist())) >= 2 and _shared_vertex_loglik(
            pos, q, uh, lab_c
        ) > _shared_vertex_loglik(pos, q, uh, lab):
            lab = lab_c
        # coherence override: adopt the coherence partition when it wins EITHER the
        # noise-robust likelihood by _COH_MARGIN AND is azimuthally complete (no ring is a
        # large partial arc), OR the physicality-aware quality score by _QUAL_MARGIN, OR
        # the structural+physicality gate fires (only when the standard partition carries a
        # near-junk wide cone). The completeness guard on the likelihood path rejects
        # spurious partitions whose "ring" is mis-grouped hits (large azimuthal gap, e.g.
        # ev44) while admitting correct well-separated splits (e.g. ev39) and real rings
        # whose apparent angle is distorted by a displaced vertex (e.g. ev83). The gate's
        # own-vertex check rejects ghost rings (e.g. ev91).
        coh = _coherence_two_ring(uh, q)
        if coh is not None and len(set(coh[coh > 0].tolist())) >= 2:
            std_maxthc = 0.0
            for k in sorted(set(lab[lab > 0].tolist())):
                mk = lab == k
                axk = np.average(uh[mk], 0, weights=q[mk])
                axk /= np.linalg.norm(axk)
                angk = np.degrees(np.arccos(np.clip(uh @ axk, -1, 1)))
                std_maxthc = max(std_maxthc, float(np.average(angk[mk], weights=q[mk])))
            g_coh = _coherence_max_gap(uh, q, coh)
            g_std = _coherence_max_gap(uh, q, lab)
            win_ll = (
                _coherence_loglik(q, uh, coh) > _coherence_loglik(q, uh, lab) + _COH_MARGIN
                and g_coh <= _COH_GAP_MAX
            )
            win_q = _partition_quality(uh, q, coh) > _partition_quality(uh, q, lab) + _QUAL_MARGIN
            win_gate = std_maxthc >= _GATE_STD_MAXTHC and _coherence_gate(pos, t, q, uh)
            # relative-completeness trigger: adopt coherence when it is a complete pair of
            # rings while the standard partition contains a grossly incomplete one (a tight
            # clump / partial arc). This is the likelihood-independent route to the correct
            # split for well-separated comparable rings, where the mixture likelihood can
            # prefer the bad split (it scores a tight off-ring clump well). Completeness is
            # robust to the input variations that make the likelihood unreliable here.
            win_complete = g_coh <= _COH_GAP_MAX and g_std - g_coh >= _COH_GAP_REL
            if win_ll or win_q or win_gate or win_complete:
                lab = coh
        # displaced-vertex recovery (genuine two-vertex events; EXPERIMENTAL, see the
        # _displaced_vertex_recover section header). Tightly gated; returns the labels
        # unchanged unless the gate fires. dv_endk/dv_V let the displaced ring's returned
        # parameters be fit at its OWN recovered vertex rather than the shared one.
        lab_dv, dv_endk, dv_V = _displaced_vertex_recover(pos, t, q, lab, V)
        if dv_endk is not None:
            lab = lab_dv
    uniq = sorted(set(lab[lab > 0].tolist()))
    relab = np.zeros(N, int)
    rings = []
    for j, lv in enumerate(uniq, 1):
        m = lab == lv
        relab[m] = j
        if V is not None:  # >=2 rings: shared vertex + physical guard
            Vk = dv_V if (dv_endk is not None and lv == dv_endk) else V
            rings.append(_ring_params_shared(pos[m], t[m], q[m], Vk))
        else:
            rings.append(_ring_params(pos[m], t[m], q[m], np.ones(int(m.sum()))))
    if not rings:
        relab[:] = 1
        rings = [_ring_params(pos, t, q, np.ones(N))]
    return relab, rings


def partition_two_rings(pos, t, q, margin=_PEEL_MARGIN):
    return _cached_call("partition_two_rings", _partition_two_rings_uncached, pos, t, q, margin)
