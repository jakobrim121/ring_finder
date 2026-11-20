"""Select the calibrated spherical-cap integrator for charged tracks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "primary_only"))
# Load the shared path and conversion modules before replacing the raw integral.
from bare_model import ROOT as CHARGED_TRACK_ROOT

sys.path.insert(0, str(CHARGED_TRACK_ROOT / "cap_lookup"))
from integration import integrate
import source

source.integrate = integrate
sys.path[:0] = [str(ROOT / name) for name in ("delta_coupled", "bridge_residual")]
