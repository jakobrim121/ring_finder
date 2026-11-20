"""Keep optically different modes through a short physical search."""

from collections import defaultdict
import numpy as np

from chain_sources import paths


def separate(proposals, limit):
    """Group starts by spatial curves before ranking their latent masses."""
    modes = []
    for proposal in proposals:
        mapped = [p for p, _ in paths(proposal.coordinates[:-2])]
        feature = (np.array([p[:3] for p in mapped]), np.array([p[3:6] for p in mapped]))
        if any(
            np.linalg.norm(feature[0] - old[0], axis=1).max() < 0.15
            and np.min(np.sum(feature[1] * old[1], axis=1)) > 0.97
            for old in modes
        ):
            continue
        modes.append(feature)
        yield proposal
        if len(modes) >= limit:
            break


def allocate(proposals, coarse_fit, max_modes=3):
    """Give several starts a short fit, then finish different ring counts."""
    stages = defaultdict(list)
    for proposal in separate(proposals, max_modes):
        result = coarse_fit(proposal)
        if result is not None:
            score, count, x = result
            stages[count].append((score, proposal, x))
    finalists = []
    for choices in stages.values():
        finalists.append(max(choices, key=lambda r: r[0]))
    for _, proposal, x in sorted(finalists, key=lambda r: r[0], reverse=True)[:2]:
        yield proposal, x
