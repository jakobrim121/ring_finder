"""The same primary tracks, with their analytic knock-on light in background.

All core fractions, geometric constraints, response bounds and model dimensions
are inherited. Knock-on light shares the parent's absolute response; its amount
and angular distribution follow the fitted parent speed, not a free amplitude.
"""

from bare_model import Model as Bare
from chain_sources import paths
from analytic_knockon import render


class Model(Bare):
    delta_scale = 1.0

    def primary_light(self, x):
        parts, jac = super().primary_light(x)
        if self.delta_scale:
            for j, ((path, transform), particle) in enumerate(zip(paths(x), self.particles)):
                value, derivative = render(self.pos, self.normals, path, particle, self.order)
                parts[j, 1] += self.delta_scale * value
                jac += self.delta_scale * derivative @ transform
        return parts, jac
