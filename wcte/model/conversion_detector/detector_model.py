"""Qualified conversion physics, including the simpler one-electron control."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bridge_soft_mesh"))
sys.path.insert(0, str(ROOT / "conversion_core"))
from pair_model import Model as ReferenceElectron, counted

sys.path.insert(0, str(ROOT / "exact_containment"))
from pair_domes import CompleteEnvelope


class Electron(CompleteEnvelope, ReferenceElectron):
    pass
