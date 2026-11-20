"""Only settings actually read by the adopted seed counter.

The earlier research config mixed 59 settings for several incompatible models.
V101 reads these 15; the discarded switches belonged to unreachable decision
branches. Defaults and the active numerical values are unchanged.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
import math


@dataclass(frozen=True, slots=True)
class RingFinderConfig:
    # Prompt selection follows the observed event peak, in nanoseconds.
    prompt_pre_peak_ns: float = 8.0
    prompt_post_peak_ns: float = 7.0
    prompt_peak_bandwidth_ns: float = 1.0
    prompt_peak_charge_cap_pe: float = 2.0

    # A fit can use a ten-PMT seed; the final observable-ring floor is thirty.
    minimum_partition_pmts: int = 10
    minimum_observable_ring_pmts: int = 30
    minimum_three_ring_pmts: int = 30
    decision_threshold: float = 0.01
    three_ring_decision_threshold: float = 0.01
    enable_two_ring_evidence_rescue: bool = True
    enable_three_ring_evidence_rescue: bool = True

    # These identify the single retained scientific route, not tunable models.
    counter_model: str = "unified_v226"
    partition_model: str = "analytic_v225"
    partition_background_policy: str = "preserve"
    mode: str = "default"

    def __post_init__(self):
        if (
            self.counter_model,
            self.partition_model,
            self.partition_background_policy,
        ) != ("unified_v226", "analytic_v225", "preserve"):
            raise ValueError("This package contains the adopted V101 seed model only")
        if self.mode not in ("default", "high_purity_one_ring"):
            raise ValueError("Unknown one-ring selection mode")
        for name in (
            "prompt_pre_peak_ns",
            "prompt_post_peak_ns",
            "prompt_peak_bandwidth_ns",
            "prompt_peak_charge_cap_pe",
            "decision_threshold",
            "three_ring_decision_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "minimum_partition_pmts",
            "minimum_observable_ring_pmts",
            "minimum_three_ring_pmts",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RingFinderConfig()


def coerce_config(
    config: RingFinderConfig | Mapping[str, Any] | None,
) -> RingFinderConfig:
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, RingFinderConfig):
        return config
    if isinstance(config, Mapping):
        return RingFinderConfig(**dict(config))
    raise TypeError("config must be RingFinderConfig, a mapping, or None")
