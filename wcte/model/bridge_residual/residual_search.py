"""Look for missed tracks after fitting the directional soft background.

The earlier residual search ran before fitting electron light, so its peaks
could describe that background or duplicate the dominant track. This search
uses the complete one-core model. Its score only proposes starting points;
the unchanged full physical likelihood decides every fitted result.
"""

import numpy as np
from source import trajectory
from material import material
from electron import turning


def axis_grid(count=768):
    index = np.arange(count) + 0.5
    z = 1 - 2 * index / count
    phi = index * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def starting_paths(x, particle, proposed):
    """Keep a prefix or suffix of the fitted path, joined to a proposed leg.

    A downstream source can begin at any sampled point along the fitted path.
    Incoming alternatives are also included. The existing core's material
    history and circular trajectory are preserved exactly in these starts.
    """
    allowed = {
        ("pion", "pion"),
        ("proton", "proton"),
        ("pion", "electron"),
        ("proton", "electron"),
        ("electron", "electron"),
        ("pion", "proton"),
        ("proton", "pion"),
        ("muon", "electron"),
    }
    axis = x[3:6] / np.linalg.norm(x[3:6])
    r, f = x[9:11]
    for proposal in proposed:
        u = proposal["path_fraction"]
        for main in (0, 1):
            if (main == 0 and u == 0.0) or (main == 1 and u == 1.0):
                continue
            if main == 0:
                main_axis, bend, ran, fraction = axis, u * x[6:9], r, f * u
            else:
                tangents = trajectory(x[:12], np.array([u, 1.0]))[1]
                main_axis = tangents[0]
                bend = turning(tangents[0], tangents[1])[0]
                ran, fraction = r * (1 - f * u), f * (1 - u) / (1 - f * u)
            for other_particle in ("electron", "muon", "pion", "proton"):
                particles = (particle, other_particle) if main == 0 else (other_particle, particle)
                if particles not in allowed:
                    continue
                m = material(other_particle)
                if other_particle == "electron":
                    energies = (20.0, 80.0)
                else:
                    theta = np.deg2rad(
                        min(proposal["opening_deg"], np.rad2deg(np.arccos(1 / m["index"])) - 0.01)
                    )
                    beta = 1 / (m["index"] * np.cos(theta))
                    energy = m["mass"] * (1 / np.sqrt(1 - beta * beta) - 1)
                    energies = (np.clip(energy, m["energy"][0] + 40.0, 3000.0),)
                for energy in energies:
                    other_range = float(m["range_at"](energy)) / 1000
                    axes = [None, None]
                    bends = [None, None]
                    ranges = [None, None]
                    spans = [None, None]
                    widths = [np.log(np.deg2rad(1.0))] * 2
                    axes[main] = main_axis
                    axes[1 - main] = proposal["axis"]
                    bends[main] = bend
                    bends[1 - main] = np.zeros(3)
                    ranges[main] = ran
                    ranges[1 - main] = other_range
                    spans[main] = fraction
                    spans[1 - main] = min(0.25 / other_range, 0.8)
                    widths[main] = x[11]
                    geometry = np.r_[
                        proposal["vertex_m"],
                        axes[0],
                        axes[1],
                        ranges[0],
                        spans[0],
                        ranges[1],
                        spans[1],
                        widths,
                        bends[0],
                        bends[1],
                    ]
                    yield particles, np.r_[geometry, x[12:-2], x[-2:]], proposal["score"]
