"""Reuse identical optical evaluations within one event, without rounding.

Charge fitting, amplitude profiling and endpoint certification often request
the same source again. Keys contain every parameter bit and the quadrature
order. This cache never stores a fitted answer or uses event identity.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np

_current = ContextVar("event_optical_cache", default=None)


@contextmanager
def event_cache(capacity=64):
    """Bound memory and discard all entries when the event finishes."""
    state = dict(entries=OrderedDict(), capacity=capacity, hits=0, misses=0, geometries={})
    token = _current.set(state)
    try:
        yield state
    finally:
        _current.reset(token)


def evaluate(function, positions, normals, parameters, *, kind, order):
    state = _current.get()
    if state is None:
        return function(positions, normals, parameters)
    values = np.asarray(parameters, dtype=float)
    # Getting ndarray.ctypes allocates a Python wrapper. Resolve the unchanged
    # geometry pointers once per pair of array objects, retaining the objects
    # until this event ends so their identities cannot be recycled.
    identity = (id(positions), id(normals))
    geometry = state["geometries"].get(identity)
    if geometry is None:
        geometry = (positions, normals, positions.ctypes.data, normals.ctypes.data)
        state["geometries"][identity] = geometry
    key = (kind, order, values.tobytes(), geometry[2], geometry[3])
    entries = state["entries"]
    if key in entries:
        state["hits"] += 1
        entries.move_to_end(key)
        return entries[key]
    state["misses"] += 1
    result = function(positions, normals, parameters)
    entries[key] = result
    if len(entries) > state["capacity"]:
        entries.popitem(last=False)
    return result
