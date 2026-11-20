"""Compare new sources with the complete, previously qualified candidate pool.

The fixed penalty, count-change margin and hard-label support gate are
unchanged. Truth, particle labels and beam energies are absent from this file.
"""

from detector_model import counted as original_counted
from chain_model import Model as Chain, counted as chain_counted


def counted(model, state):
    return chain_counted(model, state) if isinstance(model, Chain) else original_counted(model, state)


from compare import MARGIN
