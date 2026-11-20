"""Per-event thresholds for turning fitted charged-track sections into rings."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import cos, isfinite, radians


@dataclass(frozen=True)
class TrackCounting:
    min_track_pmts: float = 30.0
    track_merge_angle_deg: float = 15.0
    merge_cosine: float = field(init=False, repr=False)

    def __post_init__(self):
        if not isfinite(self.min_track_pmts) or self.min_track_pmts <= 0:
            raise ValueError("min_track_pmts must be finite and positive")
        angle = self.track_merge_angle_deg
        if not isfinite(angle) or not 0 <= angle <= 180:
            raise ValueError("track_merge_angle_deg must be between 0 and 180 degrees")
        object.__setattr__(self, "merge_cosine", cos(radians(angle)))


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
