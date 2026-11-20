"""Find a missed charged section before an electron can absorb its light.

The earlier residual scan started from a primary plus a free electron. That
is useful for rejecting soft background, but can hide a weak second hadron.
This complementary search starts from each fitted primary-only hypothesis.
It retains the old full bank and uses exactly the same final likelihood,
constraints and count rules. Residual scores propose starts only.
"""
import numpy as np
from scipy.optimize import brentq
from chain_sources import paths, interactions
from curved_legs import local_path
from residual_search import axis_grid
from source import trajectory
from material import material, bounded_path
from wctering.receiver.point_source import cap


def residual_scores(shapes, means, residual, nuisance):
    """Project nuisance brightness out of a proposal's local score.

Poisson information is a cheap positive search metric here. The numerator
uses the actual calibrated charge-likelihood derivative. Neither this local
quadratic approximation nor its score is used to select the final fit.
"""
    weight = 1/np.maximum(means, 1e-12)
    information = (nuisance*weight)@nuisance.T
    inverse = np.linalg.pinv(information, rcond=1e-12)
    cross = (shapes*weight)@nuisance.T
    projection = cross@inverse
    score = -shapes@residual+projection@(nuisance@residual)
    fisher = (shapes*shapes)@weight-np.sum(projection*cross, axis=1)
    return score, fisher, projection


def proposals(model, x):
    """Scan the same cone grid as the existing residual search, without soft light."""
    model(x, True)
    primary = model.primary_cache[1]
    means = np.exp(x[-2])*primary+np.exp(x[-1])*model.bg
    total = primary.sum()
    assert np.isfinite(primary).all() and total >= 0
    # A below-threshold particle can emit exactly no Cherenkov light. Its
    # brightness direction is then zero, not primary/0. The pseudoinverse
    # naturally removes that null direction. Nonzero sources are unchanged.
    nuisance = np.array([primary/total if total > 0 else primary, model.bg])
    _, residual = model.obs.score(means, model.ix)
    fractions = np.array([0., .25, .5, .75, 1.])
    points = trajectory(x[:12], fractions)[0]
    grid = axis_grid(); width = np.deg2rad(4.); output = []
    for fraction, point in zip(fractions, points):
        delta = model.pos-point; distance = np.linalg.norm(delta, axis=1)
        rays = delta/np.maximum(distance[:, None], 1e-9)
        exposure = np.array([max(cap(-normal@ray)[0], 1e-30)
                             for normal, ray in zip(model.normals, rays)])
        exposure /= distance*distance+10000.
        candidates = []
        for start in range(0, len(grid), 64):
            axes = grid[start:start+64]
            angle = np.arccos(np.clip(axes@rays.T, -1., 1.))
            for opening in (18., 24., 30., 35., 39., 42.):
                shapes = exposure*np.exp(-.5*((angle-np.deg2rad(opening))/width)**2)
                shapes /= np.maximum(shapes.sum(1)[:, None], 1e-300)
                score, fisher, _ = residual_scores(shapes, means, residual, nuisance)
                statistic = np.maximum(score, 0.)**2/np.maximum(fisher, 1e-30)
                for j in np.flatnonzero((score > 0)&(fisher > 1e-16)):
                    candidates.append((float(statistic[j]), axes[j], opening))
        chosen = []
        for statistic, axis, opening in sorted(candidates, key=lambda v:v[0], reverse=True):
            if any(axis@p['axis'] > np.cos(np.deg2rad(12.)) and
                   abs(opening-p['opening_deg']) < 8. for p in chosen):
                continue
            chosen.append(dict(score=statistic, axis=axis.copy(), opening_deg=opening,
                path_fraction=float(fraction), vertex_m=point/1000))
            if len(chosen) == 4: break
        output.extend(chosen)
    return output


def serial_coordinates(shared):
    """Express an existing common-vertex proposal as the identical serial path."""
    mapped = [local_path(shared[:21], leg)[0] for leg in (0, 1)]
    return np.r_[mapped[0][:3], *[path[3:] for path in mapped], shared[-2:]]


def repair_energy(model, x):
    """Raise incoming energy only when needed for a feasible starting point.

The path length and spatial curve stay fixed. The old residual search simply
discarded these starts. Here the unchanged recoil constraint determines the
minimum permissible incoming range, as in the qualified optical-chain search.
"""
    x = x.copy()
    # The source already canonicalizes machine-roundoff at its closed table
    # boundary. Apply that identical rule before the recoil calculation too:
    # multiplying the largest allowed metre value by 1000 can exceed the
    # millimetre endpoint by one floating-point step.
    for leg, ((path, _), particle) in enumerate(zip(paths(x[:model.ng]), model.particles)):
        safe = bounded_path(path, material(particle)['maximum_range_mm'])
        start = 9+9*leg; x[start:start+2] = safe[9:11]
    length = x[9]*x[10]
    def surplus(ran):
        x[9:11] = ran, length/ran
        return interactions(paths(x[:model.ng]), model.particles)[0][0]
    initial = x[9]
    initial_surplus = surplus(initial)
    assert np.isfinite(initial_surplus), (model.particles, x)
    if initial_surplus >= 1e-7: return x
    # Use the actual closed endpoint. Shrinking it by a relative epsilon can
    # move it BELOW a starting range already at that endpoint. Preserving a
    # full-length path there would then give fraction > 1 and negative residual
    # range. Such a start has no higher-energy repair; reject it directly.
    endpoint = np.r_[x[:9], model.bounds[9][1], 0., x[11]]
    maximum = bounded_path(endpoint, material(model.particles[0])['maximum_range_mm'])[9]
    if maximum <= initial: return None
    maximum_surplus = surplus(maximum)
    assert np.isfinite(maximum_surplus), (model.particles, x)
    if maximum_surplus < 1e-7: return None
    root = brentq(lambda r:surplus(r)-1e-7, initial, maximum, xtol=1e-10)
    surplus(root)
    return x


