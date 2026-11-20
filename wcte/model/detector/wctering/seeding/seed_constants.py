"""Frozen proposal constants and the deterministic trial-direction grid.

These values define features consumed by the original trained counter. They
are grouped here for inspection; none is retuned by the fast implementation."""

from __future__ import annotations


import numpy as np


# Timing and trial-cone geometry (mm, ns, degrees).
_C_MM_NS = 299.792458
_N_GROUP = 1.388
_VG = _C_MM_NS / _N_GROUP
_SIGMA_THETA = 10.0
_THETA_C0 = 40.0
_PEEL_MARGIN = 15.0
_THETA_C_MAX = 55.0

# Coherence and angular-support probes.
_COH_SIGMA = 8.0
_COH_SUPPRESS_K = 2.5
_COH_BG = 0.05
_COH_MARGIN = 5.0
_COH_GAP_BAND = 20.0
_COH_GAP_MAX = 65.0
_COH_GAP_REL = 60.0
_QUAL_BAND = 10.0
_QUAL_PHYS = 52.0
_QUAL_DISC = 0.25
_QUAL_MARGIN = 0.02

# Frozen H2 proposal gates.
_GATE_DOMFRAC = 0.66
_GATE_THC2 = 36.0
_GATE_COV2 = 10
_GATE_SPR2C = 7.0
_GATE_STD_MAXTHC = 53.0
_GATE_THC2_OWN = 48.0

# H3 hierarchy: compare candidate ranks and protect the weakest component.
_H3_HIER_PHASE_INDEX = 1.346
_H3_HIER_WEAK_FRACTION_TOLERANCE = 0.10
_H3_HIER_BACKGROUND = 0.05
_H3_HIER_OPENING_TOLERANCE_DEG = 1.0
_H3_HIER_FEATURES = (
    "timing_bic_gain_per_record",
    "angular_loglik_gain_per_capped_charge",
    "resolved_margin_deg",
    "minimum_coverage_sectors",
    "physical_completeness_quality",
    "minimum_capped_charge_fraction",
)


def _fib_sphere(M):
    i = np.arange(M)
    ph = np.arccos(1 - 2 * (i + 0.5) / M)
    th = np.pi * (1 + 5**0.5) * i
    return np.c_[np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)]


_CAND = _fib_sphere(1500)

# Displaced-vertex search navigation.
_DV_Y_TOP = 1160.0
_DV_Y_BOT = -580.0
_DV_GRID = np.arange(-200.0, 1201.0, 100.0)
_DV_BAND = 12.0
_DV_GMIN = 0.04
_DV_GMAX = 0.25
_DV_THC_LO = 30.0
_DV_THC_HI = 52.0
_DV_RMS_MAX = 1.4
