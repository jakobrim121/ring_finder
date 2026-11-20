"""Keep a measured one-ring path while profiling bright uncounted light."""

import numpy as np

from conversion_light_budget import estimates
from low_angle_continuation import coordinates
from residual_joint_optics import path_from_geometry, prompt
from timed_optical import arrival


def keep_main(event, model, geometry, shapes, amplitudes, previous, optical, extra):
    """Preserve a narrow core when two conversion tails explain the wide lobe."""
    if event.timing_selection["prompt_window_ns"] is None or previous.shape[1] != 2:
        return None
    paths = optical["paths"]
    other = 1 - extra
    small, large = paths[extra], paths[other]
    if np.exp(large[8]) < np.deg2rad(4.0) or small[7] >= large[7]:
        return None
    if np.exp(small[8]) > np.deg2rad(3.0) or large[7] < np.deg2rad(35.0):
        return None
    try:
        old = path_from_geometry(geometry, 0)
    except (ValueError, ArithmeticError):
        return None
    # The narrower candidate must continue the measured original direction.
    # Its apparent dominance is allowed to change after accounting for tails.
    if small[3:6] @ old[3:6] < np.cos(np.deg2rad(25.0)):
        return None
    old_score = model.obs.score(amplitudes @ shapes, model.ix)[0]
    new_geometry = coordinates(paths, optical["photons"])
    pair = estimates(event, model, new_geometry, optical["shapes"], optical["photons"], other)
    best = max(pair, key=lambda row: row["joint_gain"], default=None)
    if best is None or best["cores"] or best["joint_gain"] <= 3.31745:
        return None
    visible = max((row["joint_gain"] for row in pair if row["cores"]), default=-np.inf)
    if best["joint_gain"] - visible <= 3.31745 or best["charge_ll"] <= old_score + 12.0:
        return None
    q = np.minimum(event.charges_pe, 2.0)
    f = best["charge_fractions"]
    dominant = f[:, 1]
    charge = q * dominant
    support = float(np.sum(1.0 - model.obs.no_source(best["means"], dominant)))
    if support < 30.0 or charge.sum() < 20.0:
        return None
    timed, _ = prompt(event, arrival(event, new_geometry)[:, [extra]], f)
    own = (f.argmax(1) == 1) & (timed[:, 0] >= 0.25 * q)
    if own.sum() < 20 or len(np.unique(event.pmt_ids[own] // 100)) < 3:
        return None
    return (
        best["charge_ll"],
        f,
        dict(
            soft_component=other + 1,
            soft_pmts=best["support"],
            kept_pmts=support,
            path=small,
            soft_paths=best["paths"],
            joint_gain=best["joint_gain"],
            quadrature_ll_error=best["quadrature_ll_error"],
        ),
    )
