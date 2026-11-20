"""Review a hadron chord whose measured longitudinal profile leaves strong evidence."""

import numpy as np

from shared_scoring import match_charge_shares


def nominated(detail, prediction):
    choice = detail.get("selected", {})
    record = detail.get("chord", {})
    gain = record.get("profile", {}).get("log_likelihood_gain") or 0.0
    if choice.get("family") != "coherent_track" or prediction["fractions"].shape[1] >= 4:
        return False
    if gain >= 225.0:
        return True
    if gain < 190.0 or prediction["fractions"].shape[1] != 2 or not record.get("chord"):
        return False
    from staged_physics.primary import Aperture

    line = record["chord"]
    anchor, axis = (np.asarray(line[k], float) for k in ("anchor_m", "axis"))
    if axis[2] < 0.6 or axis[2] >= np.cos(np.radians(15.0)) or line["coherent_coverage"] >= 0.80:
        return False
    aperture = Aperture()
    entry = anchor + (aperture.z - anchor[2]) * axis / axis[2]
    return bool(np.linalg.norm(entry[:2]) > aperture.radius + 0.12)


def qualify(prediction, alternate, *, protected_primary):
    old, new = prediction["fractions"], alternate["fractions"]
    report = dict(applied=False, previous_count=old.shape[1] - 1, retried_count=new.shape[1] - 1)
    if new.shape[1] < 4 or new.shape[1] <= old.shape[1]:
        return prediction, report
    if any(
        not np.array_equal(prediction[name], alternate[name])
        for name in ("pmt_ids", "charges", "input_indices")
    ):
        raise ValueError("Chord comparison uses different selected PMTs")
    q = np.minimum(prediction["charges"], 2.0)
    signals = old[:, 1:].sum(1), new[:, 1:].sum(1)
    retained = (q @ np.minimum(*signals)) / max(q @ signals[0], 1e-12)
    report["retained_signal"] = float(retained)
    if retained < 0.90:
        return prediction, report
    if protected_primary:
        before, after, _ = match_charge_shares(old, new[:, 1:].T, q)
        if 0 not in before:
            return prediction, report
        column = int(after[np.flatnonzero(before == 0)[0]]) + 1
        replaced = new.copy()
        replaced[:, column] = 0.0
        unclaimed = 1.0 - old[:, 1]
        remainder = replaced.sum(1)
        scale = np.divide(unclaimed, remainder, out=np.zeros_like(unclaimed), where=remainder > 1e-12)
        replaced *= scale[:, None]
        replaced[:, 0] += np.where(remainder > 1e-12, 0.0, unclaimed)
        replaced[:, column] = old[:, 1]
        new = replaced
    if np.min(q @ new[:, 1:]) < 25.0:
        return prediction, report
    report.update(applied=True, preserved_incoming_charge=bool(protected_primary))
    return dict(alternate, fractions=new), report
