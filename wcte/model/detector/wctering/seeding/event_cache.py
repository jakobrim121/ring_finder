"""Event-local memoization of deterministic seed calculations.

Mutable results are copied on both insertion and retrieval. Read-only arrays
can be shared. A new event always starts with a fresh cache."""

from __future__ import annotations

from contextlib import contextmanager

from contextvars import ContextVar

from copy import deepcopy
import pickle


import numpy as np


_EVENT_LOCAL_CACHE = ContextVar("wcte_partitioner_event_cache", default=None)


_CACHE_MISS = object()


def _array_signature(value):
    """Exact dtype/shape/content signature for one numerical argument."""

    array = np.ascontiguousarray(np.asarray(value))
    return array.dtype.str, array.shape, array.tobytes()


def _cached_call(name, function, *arguments):
    cache = _EVENT_LOCAL_CACHE.get()
    if cache is None:
        return function(*arguments)
    key = (str(name), *(_array_signature(value) for value in arguments))
    cached = cache.get(key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return deepcopy(cached)
    result = function(*arguments)
    # Both sides are protected: downstream code can mutate the first result or
    # a cache hit without changing what a later identical call receives.
    cache[key] = deepcopy(result)
    return result


def _cached_readonly_array_call(name, function, *arguments):
    """Cache an internal array whose consumers only read it, without copying it."""

    cache = _EVENT_LOCAL_CACHE.get()
    if cache is None:
        return function(*arguments)
    key = (str(name), *(_array_signature(value) for value in arguments))
    cached = cache.get(key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return cached
    result = np.asarray(function(*arguments))
    result.setflags(write=False)
    cache[key] = result
    return result


@contextmanager
def fresh_event_local_cache():
    """Install an isolated cache even when a worker inherited parent state."""

    token = _EVENT_LOCAL_CACHE.set({})
    try:
        yield
    finally:
        _EVENT_LOCAL_CACHE.reset(token)


class CachedOptimizer:
    """Reuse an identical optical fit only within its owning event.

    Keys include the exact arrays, seeds and optimizer options. Copies protect
    cached results from later changes to family labels or working geometries.
    Debug calls return a live objective closure and are deliberately not cached.
    """

    def __init__(self, function):
        self.function = function
        self.results = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, *args, **kwargs):
        if kwargs.get("_debug", False):
            return self.function(*args, **kwargs)
        key = pickle.dumps((args, sorted(kwargs.items())), protocol=5)
        if key in self.results:
            self.hits += 1
            return deepcopy(self.results[key])
        result = self.function(*args, **kwargs)
        self.results[key] = deepcopy(result)
        self.misses += 1
        return result
