"""Assemble timing, angular, arc, isotropy and support features for multiplicity."""

import numpy as np


def detector_evidence(pmt_ids, positions, charges, times, payload):
    from wctering.seeding.evidence_features import evidence_features
    from wctering.seeding.coherent_timing import coherent_timing_features
    from wctering.seeding.angular_evidence import angular_evidence_features
    from wctering.seeding.arc_evidence import arc_evidence_features
    from wctering.seeding.isotropy_evidence import isotropy_evidence_features

    p = coherent_timing_features(pmt_ids, positions, charges, times, payload)
    p.update(angular_evidence_features(pmt_ids, positions, charges, times, payload))
    p.update(arc_evidence_features(pmt_ids, positions, charges, times, payload))
    p.update(isotropy_evidence_features(pmt_ids, positions, charges, times, payload))
    n = len(set(np.asarray(pmt_ids).tolist()))
    charge = float(np.asarray(charges, float).sum())
    p.update(
        {
            "support.observed_pmts": float(n),
            "support.observed_charge": charge,
            "support.mean_charge": charge / max(n, 1),
        }
    )
    return evidence_features(payload, extra_physics=p)
