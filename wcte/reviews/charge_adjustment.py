"""Apply a bounded adjustment to a completed shared-charge prediction."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from light_competition import Competition, optical_template
from partition_features import features, reshare


@lru_cache(maxsize=1)
def calibration():
    path = Path(__file__).with_name("charge_adjustment_v1.json")
    payload = json.loads(path.read_text())
    parameters = np.asarray(payload["parameters"], float)
    maximum_loss = float(payload["maximum_ring_charge_loss"])
    if payload["schema"] != 1 or parameters.shape != (20,) or not np.isfinite(parameters).all():
        raise ValueError("Incompatible charge adjustment parameters")
    if not 0.0 < maximum_loss <= 0.05:
        raise ValueError("Invalid per-ring charge limit")
    return parameters, maximum_loss


def reference(event, old, bank, count):
    from coupled_model import Model
    from optical_chain_search import geometries
    from wctering.likelihood.geometry import common_background

    if "1" not in bank or not event.timing_selection.get("prompt_window_ns") or count > 4:
        return None, None
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    receiver = Model(event.pmt_ids, event.charges_pe, background, ("muon",), 4, 2)
    choice = Competition(event, receiver)
    shapes = geometries(old, bank, count)
    if len(shapes) > 2:
        shapes = [shapes[0], shapes[-1]]
    best = None
    for geometry in shapes:
        t = optical_template(receiver, geometry, "charge_reference")
        if t is None:
            continue
        result = choice.fit(t)
        if result is not None and (best is None or result["charge"] > best["charge"]):
            best = result
            best["geometry"] = geometry
    if best is None:
        return None, None
    return best["fractions"]["charge"], best


def review(captured, prediction, particle, *, parameters=None, maximum_loss=None):
    event, old, bank = (captured[k] for k in ("event", "detail", "bank"))
    before = prediction["fractions"]
    count = before.shape[1] - 1
    if count == 0 or len(before) == 0 or not event.timing_selection.get("prompt_window_ns"):
        return prediction, dict(applied=False)
    if not np.array_equal(event.pmt_ids, prediction["pmt_ids"]) or not np.array_equal(
        event.input_indices, prediction["input_indices"]
    ):
        raise ArithmeticError("Identical prompt hit order is required for the charge adjustment")
    if parameters is None:
        parameters, maximum_loss = calibration()
    saved, trial = reference(event, old, bank, count)
    design = features(event, before, particle, saved, trial)
    q = np.minimum(event.charges_pe, 2.0).astype(np.float32)
    after = reshare(before, design, q, parameters, maximum_loss)
    old_charge = q @ before[:, 1:]
    depleted = q @ np.maximum(before[:, 1:] - after[:, 1:], 0.0)
    if np.any(depleted > (maximum_loss + 1.0e-6) * old_charge + 1.0e-7):
        raise ArithmeticError("Charge adjustment exceeded its per-ring limit")
    return dict(prediction, fractions=after), dict(
        applied=True,
        calibration_schema=1,
        maximum_ring_charge_loss=maximum_loss,
        retuned_clock_geometry=trial is not None,
    )
