"""Couple the existing analytic knock-on table to a curved slowing hadron.

The table integrates the PDG collision spectrum and electron slowing and
angular diffusion. Unlike the old fixed-cone trial, the parent speed now
changes along its fitted CSDA path. No new brightness parameter is introduced.

This first diagnostic retains the table's declared approximations: heavy
projectile limit, constant refractive index, co-located electron emission,
and transfers below the fixed ring-energy margin. It adds light only along
the represented parent path and is not an event-specific shower model.
Primary reference: https://pdg.lbl.gov/2025/reviews/rpp2025-rev-passage-particles-matter.pdf
"""

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from material import material, bounded_path
from source import trajectory
from lookup import cap

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(None)
def table():
    folder = ROOT / "delta_table_3444770"
    assert json.loads((folder / "qualification.json").read_text())["passed"]
    with np.load(folder / "table.npz", allow_pickle=False) as saved:
        return dict(saved)
