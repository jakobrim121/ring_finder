"""Replace repeated conservative prism rows with exact arc envelopes.

Photon likelihoods, bend/energy constraints, and the final dome-clearance
check stay identical. The geometric feasible region is the same prism with
an exact arc envelope instead of a small conservative chord padding.
"""

import numpy as np
from electron import legs, directions
from curved_legs import local_path, interaction_coordinates
from conservation import margins
from arc import primary, prism


def electron_constraints(path, arc_builder=legs, arc_constraint=prism):
    values, rows = [], []
    for arc, transform in arc_builder(path):
        value, jac = arc_constraint(arc)
        values.append(value)
        rows.append(jac @ transform)
    axes, jac = directions(path)
    chart = np.zeros((len(axes) - 1, len(path)))
    for j in range(len(axes) - 1):
        chart[j, 3 + 3 * j : 6 + 3 * j] = jac[j] @ axes[j + 1]
        chart[j, 6 + 3 * j : 9 + 3 * j] = jac[j + 1] @ axes[j]
    return (
        np.r_[np.concatenate(values), np.einsum("ij,ij->i", axes[:-1], axes[1:]) + 0.99],
        np.vstack([*rows, chart]),
    )


class Envelope:
    def primary_constraints(self, x, particle):
        return primary(x, particle)

    def mapped_paths(self, z):
        return [(z, np.eye(len(z)))]

    def secondary_constraints(self, path):
        return electron_constraints(path)

    def all_constraints(self, x):
        previous = getattr(self, "_envelope_cache", None)
        if previous is not None and np.array_equal(x, previous[0]):
            return previous[1:]
        values, rows = [], []
        if self.ng == 21:
            for leg, particle in enumerate(self.particles):
                path, transform = local_path(x[:21], leg)
                value, jac = self.primary_constraints(path, particle)
                row = np.zeros((len(value), len(x)))
                row[:, :21] = jac @ transform
                values.append(value)
                rows.append(row)
            z, transform = interaction_coordinates(x[:21])
            value, jac = margins(z, self.particles)
            best = int(np.argmax(value))
            row = np.zeros((1, len(x)))
            row[0, :21] = jac[best] @ transform
            values.append(value[best : best + 1])
            rows.append(row)
        else:
            value, jac = self.primary_constraints(x[:12], self.particle)
            row = np.zeros((len(value), len(x)))
            row[:, :12] = jac
            values.append(value)
            rows.append(row)
        for path, transform in self.mapped_paths(x[self.ng : -2]):
            value, jac = self.secondary_constraints(path)
            row = np.zeros((len(value), len(x)))
            row[:, self.ng : -2] = jac @ transform
            values.append(value)
            rows.append(row)
        value, jac = np.concatenate(values), np.vstack(rows)
        self._envelope_cache = (x.copy(), value, jac)
        return value, jac

    def inequality(self, x):
        return self.all_constraints(x)[0]

    def inequality_jacobian(self, x):
        return self.all_constraints(x)[1]
