"""The public prediction and the seed partition used when a fit cannot qualify.

Every returned array follows the retained prompt-hit order. ``input_indices``
maps those rows back to the caller's original pulses; excluded pulses are not
silently mixed into the fitted sample. Fraction column 0 is background and
columns 1..K are rings. A ring's allocated charge is always measured q*fraction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wctering.seeding.io import EventHits


@dataclass(frozen=True)
class Reconstruction:
    """One event: ring count, prepared detector hits, and N by (K+1) fractions."""

    count: int
    hits: EventHits
    fractions: np.ndarray
    diagnostics: dict

    @property
    def labels(self):
        """Exclusive labels for plots/scoring; sharing remains in fractions."""
        return self.fractions.argmax(axis=1)

    @property
    def shared_charge_pe(self):
        return self.hits.charges_pe[:, None] * self.fractions

    @property
    def geometry(self):
        """Selected geometry in mm/degrees, or None for a seed-only fallback.

        A crossing model explicitly labels its anchor as a line coordinate,
        rather than a physical emission vertex; its water segments are included.
        """
        index = self.diagnostics.get("selected_fit_index")
        return None if index is None else self.diagnostics["fits"][index]["geometry"]

    def validate(self):
        """Check the output contract without consulting simulation truth."""
        fractions = self.fractions
        if fractions.shape != (len(self.hits.pmt_ids), self.count + 1):
            raise ValueError("Fractions do not align with retained hits/ring count")
        if not np.isfinite(fractions).all() or np.any(fractions < 0):
            raise ValueError("Fractions must be finite and nonnegative")
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        if sorted(set(self.labels) - {0}) != list(range(1, self.count + 1)):
            raise ValueError("A claimed ring has no retained hard-label support")

    def save(self, path):
        """Write the same six-array prediction format used by the V101 scorer."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pmt_ids=self.hits.pmt_ids,
            input_indices=self.hits.input_indices,
            charges=self.hits.charges_pe,
            fractions=self.fractions,
            shared_charge_pe=self.shared_charge_pe,
            hard_labels=self.labels,
        )


def seed_partition(event, diagnostics):
    """Read the current seed counter's aligned partition, including H0.

    The old adapter also constructed several unused legacy result arrays. This
    function keeps only the labels needed for profile proposals and fallback.
    The frozen background policy, support requirements, and sharing are retained.
    """
    count = int(diagnostics.get("predicted_ring_class", 1))
    size = len(event.pmt_ids)
    if count not in (0, 1, 2, 3):
        raise ValueError(f"Unexpected seed count {count}")
    if count < 2:
        labels = np.full(size, count, dtype=np.int64)
    else:
        hypotheses = (
            diagnostics.get("full_event_hypotheses", {})
            if count == 2
            else (diagnostics.get("three_ring_hypothesis_test", {}).get("full_event_hypotheses", {}))
        )
        labels = np.asarray(hypotheses.get("partition_labels", []), dtype=np.int64)
        if labels.shape != (size,) or not np.isin(labels, range(1, count + 1)).all():
            raise ValueError("Seed count has no aligned foreground partition")
        labels = labels.copy()

    policy = diagnostics.get("final_partition_background_assignment", {})
    if count > 0 and policy.get("background_assignment_applied", False):
        labels = np.asarray(policy.get("partition_labels", []), dtype=np.int64)
        if labels.shape != (size,) or not np.isin(labels, range(count + 1)).all():
            raise ValueError("Background policy has no aligned partition")
        if sorted(set(labels) - {0}) != list(range(1, count + 1)):
            raise ValueError("Background cleanup lost a seed component")

    partition = diagnostics.get("analytic_partition", {})
    posterior = (
        np.asarray(partition["membership_probabilities"]).T
        if partition.get("applied")
        else np.eye(count + 1)[labels].T
    )
    return count, labels, posterior
