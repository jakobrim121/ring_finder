"""Reconstruct the actual frozen hypothesis before a controlled intervention."""

import numpy as np
from wctering.likelihood.geometry import active_geometry, common_background
from wctering.receiver.point_source import point_source
from wctering.receiver.angular_source import source as finite_source
from wctering.likelihood.crossing import source as crossing_source


def frozen_shapes(ids, diagnostic, bank, baseline):
    selected = diagnostic.get("selected_fit_index")
    if selected is None:
        return None
    fit = diagnostic["fits"][selected]
    g = fit["geometry"]
    active, positions, normals = active_geometry()
    ix = np.searchsorted(active, ids)
    background = common_background(np.asarray(bank["1"]["vertices"][0]))
    weights = np.asarray(g["weights"])
    k = len(weights) - 1
    lengths = g.get("lengths_mm", [0.0] * k)
    for order in (16, 32, 64, 128, 256):
        shapes = [background]
        for j in range(k):
            x = np.r_[np.asarray(g["vertices"][j]) / 1000, g["axes"][j]]
            tail = [np.deg2rad(g["openings_deg"][j]), np.log(np.deg2rad(g["widths_deg"][j]))]
            if g.get("emission_model") == "full_water_line":
                value = crossing_source(positions, normals, np.r_[x, tail], order)[0]
            elif fit["kind"] == "finite":
                value = finite_source(positions, normals, np.r_[x, lengths[j] / 1000, tail], order)[0]
            else:
                value = point_source(positions, normals, np.r_[x, tail])[0]
            shapes.append(value)
        shapes = np.asarray(shapes)
        terms = weights[:, None] * shapes[:, ix]
        post = (terms / terms.sum(0)).T
        error = float(np.max(abs(post - baseline)))
        if error < 2e-7:
            break
    if error >= 2e-7:
        raise AssertionError(f"Frozen posterior replay failed: {error}")
    amplitude = float(g.get("occupancy_intensity", len(ids)))
    return shapes, dict(
        geometry=g,
        weights=weights,
        amplitude=amplitude,
        occupancy=fit.get("occupancy", False),
        replay_error=error,
        order=order,
    )
