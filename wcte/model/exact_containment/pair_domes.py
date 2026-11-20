"""The coherent conversion source with compact, complete containment."""

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "conversion_core"))
from electron_parts import legs, core_turn

sys.path.insert(0, str(HERE))
from arc import primary, prism
from domes import certificate
from models import Envelope, electron_constraints


def arc_with_domes(x):
    water, jac = prism(x)
    dome, dome_jac = certificate(x)
    return np.r_[water, dome], np.vstack([jac, dome_jac])


class CompleteEnvelope(Envelope):
    def primary_constraints(self, x, particle):
        value, jac = primary(x, particle)
        dome, dome_jac = certificate(x)
        return np.r_[value, dome], np.vstack([jac, dome_jac])

    def mapped_paths(self, z):
        return self.path_maps(z)

    def secondary_constraints(self, path):
        value, jac = electron_constraints(path, legs, arc_with_domes)
        turn, derivative = core_turn(path)
        return np.r_[value, turn], np.vstack([jac, derivative])
