"""Seed a primary charged trajectory from the surveyed WCTE beam aperture."""

from dataclasses import dataclass

import numpy as np

from balanced_search import feasible
from charge_starts import allowed_chord
from coupled_model import Model as Coupled
from material import material
from track_coordinates import Chart
from wctering.seeding.coherent_timing import C, N_GROUP, N_PHASE


@dataclass(frozen=True)
class Aperture:
    """Physical window in the detector frame, with dimensions in metres."""

    z: float = (-1537.963 + 188.0 + 1.2) / 1000.0
    radius: float = 0.077


class EntranceChart(Chart):
    """Keep the upstream endpoint on the aperture while fitting the path."""

    def __init__(self, model, x, port):
        super().__init__(model, x)
        self.radius = port.radius
        self.initial[2] = port.z
        self.bounds[2] = port.z, port.z

    def full(self, y):
        x, jac, margin, derivative = super().full(y)
        # Divide by the aperture radius so the constraint has metre units.
        radius = self.radius
        aperture = (radius**2 - x[:2] @ x[:2]) / radius
        gradient = -(2.0 / radius) * x[:2] @ jac[:2]
        return x, jac, np.r_[margin, aperture], np.vstack([derivative, gradient])


def axes(geometry, port):
    """Include the aperture normal and the observed, forward-pointing axes."""
    point = np.array([0.0, 0.0, port.z])
    results = [np.array([0.0, 0.0, 1.0])]
    for v, a in zip(geometry["vertices"], geometry["axes"]):
        displacement = np.asarray(v) / 1000.0 - point
        for direction in (a, displacement):
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            b = np.asarray(direction) / norm
            if b[2] <= 0.0 or any(b @ old > 0.998 for old in results):
                continue
            results.append(b)
    return results


def wavefronts(event, geometry, port):
    """Estimate the bright initial arc without fixing beam momentum or clock."""
    xyz = event.pmt_positions_mm / 1000.0
    t = event.times_ns - np.median(event.times_ns)
    q = np.minimum(event.charges_pe, 2.0)
    start = np.array([0.0, 0.0, port.z])
    angles = np.unique(
        np.deg2rad(np.r_[[18.0, 24.0, 30.0, 35.0, 38.5, 41.5], np.clip(geometry["openings_deg"], 12.0, 42.2)])
    )
    rows = []
    for axis in axes(geometry, port):
        water = allowed_chord(start, axis)
        if water is None or not water[0] <= 0.0 < water[1]:
            continue
        delta = xyz - start
        along = delta @ axis
        radial = np.linalg.norm(delta - along[:, None] * axis, axis=1)
        for opening in angles:
            s = along - radial / np.tan(opening)
            # Receiver angle resolution sets the uncertainty in the
            # inferred longitudinal emission coordinate.
            sigma = np.maximum(0.035, radial * np.deg2rad(2.4) / np.sin(opening) ** 2)
            flight = 1000.0 * (N_PHASE * np.cos(opening) * s + N_GROUP * radial / np.sin(opening)) / C
            spans = np.array([0.18, 0.38, 0.8, min(1.6, water[1])])
            extent = np.minimum(np.maximum(s, 0.0)[:, None], np.minimum(spans, water[1]))
            weights = q[:, None] * np.exp(-0.5 * ((s[:, None] - extent) / sigma[:, None]) ** 2)
            residual = t - flight
            order = np.argsort(residual)
            # Every extent uses the same residual order. Reuse that sort
            # and integrate its own weighted-median clock in each column.
            cdf = np.cumsum(weights[order], axis=0)
            for j, span in enumerate(spans):
                if weights[:, j].sum() < 25.0:
                    continue
                origin = np.interp(0.5 * cdf[-1, j], cdf[:, j], residual[order])
                prompt = np.exp(-0.5 * ((residual - origin) / 1.7) ** 2)
                timed = (weights[:, j] * prompt).sum()
                if timed >= 24.0:
                    rows.append((timed / np.sqrt(0.16 + span), start, axis, opening, min(span, water[1])))
    seen = []
    for _, point, axis, angle, length in sorted(rows, key=lambda row: -row[0]):
        if any(axis @ a > 0.992 and abs(angle - b) < 0.045 and abs(length - d) < 0.18 for a, b, d in seen):
            continue
        seen.append((axis, angle, length))
        yield point, axis, angle, length
        if len(seen) >= 4:
            return


def rank(event, model, geometry, amplitude, port, masses=("muon", "pion", "proton", "electron")):
    """Give detector-derived optical modes each supported mass and range."""
    args = event.pmt_ids, event.charges_pe, model.bg
    models = {mass: Coupled(*args, (mass,), 4, 4) for mass in masses}
    modes = {}
    for point, axis, angle, length in wavefronts(event, geometry, port):
        for mass, trial in models.items():
            transport = material(mass)
            beta = min(1.0 / (N_PHASE * np.cos(angle)), 0.999)
            midpoint = transport["mass"] * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)
            energy = np.clip(midpoint, transport["energy"][0] + 1.01, transport["energy"][-1])
            remaining = float(transport["range_at"](energy)) / 1000.0
            span = min(length, remaining * 2.0, 0.997 * trial.bounds[9][1])
            ran = min(max(remaining + 0.5 * span, 1.001 * span), trial.bounds[9][1])
            x = np.r_[
                point,
                axis,
                np.zeros(3),
                ran,
                span / ran,
                np.log(np.deg2rad(2.0)),
                np.log([model.response, amplitude[0] + 0.15 * amplitude[1:].sum()]),
            ]
            if not feasible(trial, x) or trial.hardware_margin(x) <= 1e-5:
                continue
            value = trial(x, True)["ll"]
            if mass not in modes or value > modes[mass][0]:
                modes[mass] = value, trial, x
    return sorted(modes.values(), key=lambda row: -row[0])
