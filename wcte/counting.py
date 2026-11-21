"""Per-event definition of countable rings, shared by the search and reviews."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import ceil, cos, isfinite, radians


@dataclass(frozen=True)
class TrackCounting:
    min_track_pmts: float = 30.0
    track_merge_angle_deg: float = 15.0
    merge_angle_rad: float = field(init=False, repr=False)
    merge_cosine: float = field(init=False, repr=False)

    def __post_init__(self):
        if not isfinite(self.min_track_pmts) or self.min_track_pmts <= 0:
            raise ValueError("min_track_pmts must be finite and positive")
        angle = self.track_merge_angle_deg
        if not isfinite(angle) or not 0 <= angle <= 180:
            raise ValueError("track_merge_angle_deg must be between 0 and 180 degrees")
        angle_rad = radians(angle)
        object.__setattr__(self, "merge_angle_rad", angle_rad)
        object.__setattr__(self, "merge_cosine", cos(angle_rad))

    @property
    def min_unique_pmts(self):
        """Apply the effective-PMT threshold to integer PMT counts."""
        return ceil(self.min_track_pmts)

    def scaled_support(self, nominal_pmts):
        """Scale a calibrated partial-footprint or screening requirement."""
        return nominal_pmts * (self.min_track_pmts / DEFAULT_TRACK_COUNTING.min_track_pmts)

    def turn_offset(self, offset_deg):
        """Position a trial or confidence margin relative to the turn cut."""
        return min(180.0, max(0.0, self.track_merge_angle_deg + offset_deg))


DEFAULT_TRACK_COUNTING = TrackCounting()
track_counting = ContextVar("track_counting", default=DEFAULT_TRACK_COUNTING)


@contextmanager
def use_track_counting(rules):
    """Share one event's rule across the search and its reviews."""
    token = track_counting.set(rules)
    try:
        yield
    finally:
        track_counting.reset(token)


def optical_sources_resolved(axes, vertices_mm):
    """Apply the turn cut to cones that could share a fitted trajectory."""
    import numpy as np

    axes = np.asarray(axes, dtype=float)
    if len(axes) < 2:
        return True
    unit = axes / np.linalg.norm(axes, axis=1)[:, None]
    i, j = np.triu_indices(len(unit), 1)
    cosine = np.clip(np.sum(unit[i] * unit[j], axis=1), -1.0, 1.0)
    # A spatially detached cone can have the same direction. Use the
    # transverse proximity from the optical residual test to identify
    # cones for which an angular separation is relevant to counting.
    vertices = np.asarray(vertices_mm)
    offset = vertices[i] - vertices[j]
    across_i = np.linalg.norm(np.cross(offset, unit[i]), axis=1)
    across_j = np.linalg.norm(np.cross(offset, unit[j]), axis=1)
    nearby = np.minimum(across_i, across_j) < 170.0
    return not np.any(nearby & (cosine >= track_counting.get().merge_cosine))
