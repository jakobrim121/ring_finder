"""Detector-only evidence from independently optimized optical hypotheses.

All cones and mixture amplitudes are fitted within the event. The optional
counter may learn how to interpret these physical diagnostics; this module
does not consume truth or source metadata. Conditional capped-charge
likelihood differences are evidence features, not Wilks significances.
"""

import numpy as np
from wctering.seeding.optical_seed_fit import (
    active_geometry,
    seeds,
    _admissible,
)
from wctering.seeding.geometry import wcte_contains_vertex
from wctering.seeding.folds import fold_ids_for_hits
from wctering.seeding.arc_evidence import _amplitudes, _deviance
from wcte.counting import track_counting

SCHEMA = "wcte-optimized-optical-evidence-v1"


def fit_bank(ids, positions, charges, times, diagnostics, *, optimizer):
    """Optimize each multiplicity regardless of the earlier counter decision."""
    bank = {}
    summaries = {}
    if len(ids) < max(12, track_counting.get().min_unique_pmts):
        return bank, summaries
    for k in [1, 2, 3]:
        candidates = []
        for family in ["legacy", "circle", "independent"]:
            seed = seeds(diagnostics, k, positions, charges, family)
            if seed is None or any(not wcte_contains_vertex(r["vertex"], tolerance_mm=1e-5) for r in seed):
                continue
            r = optimizer(ids, positions, charges, times, seed, maxiter=75)
            r["family"] = family
            if np.isfinite(r["ll"]) and np.all(np.isfinite(r["posterior"])):
                candidates.append(r)
        if not candidates:
            continue
        # Count evidence retains collapsed optima as evidence against that K.
        best = max(candidates, key=lambda r: r["ll"])
        if not best["converged"]:
            r = optimizer(
                ids,
                positions,
                charges,
                times,
                best["seed"],
                maxiter=150,
                _initial_parameters=best["parameters"],
            )
            if r["ll"] >= best["ll"]:
                r["family"] = best["family"]
                best = r
        bank[k] = best
        summaries[k] = dict(
            candidate_ll=[r["ll"] for r in candidates],
            admissible_count=sum(_admissible(r, k, len(ids)) for r in candidates),
            _candidates=candidates,
        )
    return bank, summaries


def evidence_from_bank(ids, q, t, bank, summaries):
    active, allpos, _ = active_geometry()
    ix = np.searchsorted(active, ids)
    w = np.minimum(q, 2.0)
    W = max(w.sum(), 1.0)
    unique = np.unique(ids, return_inverse=True)[1]
    charge = np.zeros(len(active))
    np.add.at(charge, ix, w)
    fold = fold_ids_for_hits(active)
    features = {}
    predictions = {}
    for k, r in bank.items():
        pre = f"optical.h{k}"
        p = r["posterior"]
        shapes = r["shapes"]
        pi = r["weights"]
        mu = W * (pi @ shapes)
        features.update(
            {
                pre + ".ll_per_charge": r["ll"] / W,
                pre + ".background_fraction": float(pi[0]),
                pre + ".admissible_candidates": float(summaries[k]["admissible_count"]),
                pre + ".seed_ll_spread_per_charge": float(np.ptp(summaries[k]["candidate_ll"]) / W),
                pre + ".unhit_expected_fraction": float(mu[charge == 0].sum() / W),
                pre + ".converged": float(r["converged"]),
            }
        )
        vals = {
            name: []
            for name in [
                "mixture_fraction",
                "soft_pmt_support",
                "pe_occupancy_support",
                "hard_pmt_support",
                "opening_deg",
                "width_deg",
                "posterior_purity",
                "effective_modules",
                "angular_time_covariance",
            ]
        }
        for j in range(k):
            pp = p[j + 1]
            pm = np.zeros(int(unique.max()) + 1)
            np.maximum.at(pm, unique, pp)
            # Expected PMTs contributing >=1 PE in a Poisson split; only a
            # diagnostic because digitized q is not an exact photon count.
            soft = np.zeros(len(active))
            np.add.at(soft, ix, pp * q)
            occ = np.sum(-np.expm1(-soft))
            module = np.zeros(int(np.max(ids // 100)) + 1)
            np.add.at(module, ids // 100, w * pp)
            total = module.sum()
            eff = total * total / max(float(module @ module), 1e-10)
            delta = allpos[ix] - r["vertices"][j]
            dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            angle = (
                np.rad2deg(np.arccos(np.clip((delta / dist[:, None]) @ r["axes"][j], -1, 1)))
                - r["openings_deg"][j]
            )
            tr = t - 1.388 * dist / 299.792458
            ww = w * pp
            mn = max(ww.sum(), 1e-10)
            ar = angle - np.sum(ww * angle) / mn
            tt = tr - np.sum(ww * tr) / mn
            covariance = np.sum(ww * ar * tt) / max(
                np.sqrt(np.sum(ww * ar * ar) * np.sum(ww * tt * tt)), 1e-10
            )
            v = [
                pi[j + 1],
                pm.sum(),
                occ,
                len(set(ids[r["labels"] == j + 1].tolist())),
                r["openings_deg"][j],
                r["widths_deg"][j],
                np.sum(w * pp * pp) / mn,
                eff,
                covariance,
            ]
            for name, value in zip(vals, v):
                vals[name].append(float(value))
        for name, value in vals.items():
            for suffix, fn in [("min", np.min), ("mean", np.mean), ("max", np.max)]:
                features[pre + "." + name + "." + suffix] = float(fn(value))
        features[pre + ".membership_entropy"] = float(
            -np.sum(w * np.sum(p * np.log(np.maximum(p, 1e-30)), axis=0)) / W
        )
        if k > 1:
            dots = np.clip(r["axes"] @ r["axes"].T, -1, 1)
            angles = np.rad2deg(np.arccos(dots[np.triu_indices(k, 1)]))
            features[pre + ".minimum_axis_separation_deg"] = float(angles.min())
            features[pre + ".maximum_axis_separation_deg"] = float(angles.max())
        # Only amplitudes are withheld. Geometry was optimized on all hits;
        # do not describe this as a fully independent reconstruction test.
        pred = np.zeros(len(active))
        folddev = []
        for f in range(4):
            train = fold != f
            test = ~train
            amp = _amplitudes(shapes, charge, train, iterations=24)
            lam = amp @ shapes
            pred[test] = lam[test]
            folddev.append(_deviance(charge[test], lam[test]) / max(charge[test].sum(), 1.0))
        predictions[k] = pred
        features[pre + ".amplitude_cv_deviance_per_charge"] = _deviance(charge, pred) / W
        features[pre + ".amplitude_cv_worst_fold"] = float(max(folddev))
        features[pre + ".amplitude_cv_spread"] = float(np.std(folddev))
    for k in [2, 3]:
        if k not in bank or k - 1 not in bank:
            continue
        pre = f"optical.h{k}_over_h{k - 1}"
        gain = 2 * (bank[k]["ll"] - bank[k - 1]["ll"])
        features[pre + ".twice_ll_gain"] = float(gain)
        features[pre + ".ll_gain_per_charge"] = gain / (2 * W)
        features[pre + ".bic_gain"] = float(gain - 5 * np.log(W))
        p = predictions[k]
        p0 = predictions[k - 1]
        terms = 2 * (charge * np.log(np.maximum(p, 1e-20) / np.maximum(p0, 1e-20)) - p + p0)
        fg = [terms[fold == f].sum() / max(charge[fold == f].sum(), 1.0) for f in range(4)]
        features[pre + ".amplitude_cv_gain_per_charge"] = float(terms.sum() / W)
        features[pre + ".amplitude_cv_minimum_fold_gain"] = float(min(fg))
        features[pre + ".amplitude_cv_positive_folds"] = float(sum(v > 0 for v in fg))
    return features
