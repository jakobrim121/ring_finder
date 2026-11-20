"""Write NumPy diagnostic values as strict JSON; nonfinite failures become null."""

from collections.abc import Mapping, Sequence
from typing import Any
import math
import numpy as np


def copy_geometry(record):
    """Copy only the inputs consumed by the optical geometry models.

    A seed-bank record also carries large light maps, hit labels and iteration
    histories. Repeating those diagnostics in every start costs time and makes
    the fit records difficult to read. They are not parameters of the model.
    Lists give each new model its own axes, preserving V101's ownership rules.
    """
    fields = (
        "vertices",
        "axes",
        "openings_deg",
        "widths_deg",
        "weights",
        "lengths_mm",
        "training_mask",
        "occupancy_intensity",
        "initial_occupancy_intensity",
    )
    return {key: clean(record[key]) for key in fields if key in record}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def seed_json(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in (str, int, bool):
        return value
    if value_type is float:
        return value if math.isfinite(value) else None
    if value_type is dict:
        return {str(key): seed_json(item) for key, item in value.items()}
    if value_type is list or value_type is tuple:
        return [seed_json(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biu" or (value.dtype.kind == "f" and bool(np.isfinite(value).all())):
            return value.tolist()
        return [seed_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return seed_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): seed_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [seed_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
