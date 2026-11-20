"""Reuse detector inputs and charge certificates within one event."""

import traceback

from wctering.likelihood.geometry import common_background
from wctering.optimization.fit import fit_source
from wctering.serialization import copy_geometry


class FitSearch:
    def __init__(self, event, bank):
        self.event = event
        self.bank = {k: copy_geometry(geometry) for k, geometry in bank.items()}
        # Competing geometries share the diffuse background at the leading seed.
        self.background = common_background(bank[1]["vertices"][0])
        self.cache = {}

    def fit(self, count, kind, geometry):
        try:
            return fit_source(self.event, geometry, self.background, kind, cache=self.cache)
        except Exception:
            # Preserve a failed attempt in the ordered fit record.
            return dict(
                k=count,
                kind=kind,
                occupancy=False,
                crossing=False,
                qualified=False,
                traceback=traceback.format_exc(),
            )
