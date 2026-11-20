"""Reuse each evaluated charged arc across candidate families in one event."""

from contextlib import contextmanager
from functools import lru_cache, wraps

from wctering.receiver.cache import evaluate


def memoize(function, kind):
    @wraps(function)
    def render(positions, normals, path, *arguments, **options):
        # Source flags and material are scalar and immutable. Preserve all
        # coordinate bits and the integration order in the cache key.
        key = kind, arguments, tuple(sorted(options.items()))

        def compute(p, n, x):
            return function(p, n, x, *arguments, **options)

        # A few callers add the knock-on term to the raw tail in-place.
        # Isolate every returned array from the cache's retained value.
        return tuple(a.copy() for a in evaluate(compute, positions, normals, path, kind=key, order=None))

    return render


@lru_cache(maxsize=1)
def backends():
    import chain_sources
    import coupled_model

    direct, knockon = chain_sources.raw, coupled_model.render
    return (
        chain_sources,
        coupled_model,
        direct,
        knockon,
        memoize(direct, "charged_arc"),
        memoize(knockon, "knockon_arc"),
    )


@contextmanager
def cached_arcs(enabled):
    """Select exact reusable per-arc renders for a worker's search."""
    if not enabled:
        yield
        return
    chain, coupled, direct, knockon, cached_direct, cached_knockon = backends()
    chain.raw, coupled.render = cached_direct, cached_knockon
    try:
        yield
    finally:
        chain.raw, coupled.render = direct, knockon
