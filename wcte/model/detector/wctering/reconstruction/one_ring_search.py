"""Early exit for a qualified one-ring crossing explanation.

This carries forward only the earlier single-ring comparison. A rejected or
unavailable result falls through to the complete V101 reconstruction. Its
accuracy must be tested independently; this is not an equivalence claim.
"""

import copy
import traceback
from wctering.reconstruction.one_ring_fit import fit_source
from wctering.likelihood.geometry import common_background


class Search:
    def __init__(self, event, bank):
        self.event = event
        # The historical model normalizes some seed axes in place. Give the
        # quick route its own bank so a fall-through leaves V101's seeds intact.
        self.bank = copy.deepcopy(bank)
        self.background = common_background(self.bank[1]["vertices"][0])
        self.fits = []

    def fit(self, count, kind, occupancy, geometry=None):
        assert occupancy
        try:
            result = fit_source(
                self.event,
                self.bank[count] if geometry is None else geometry,
                self.background,
                kind,
            )
        except Exception:
            result = dict(
                k=count,
                kind=kind,
                occupancy=True,
                crossing=False,
                qualified=False,
                traceback=traceback.format_exc(),
            )
        self.fits.append(result)
        return result
