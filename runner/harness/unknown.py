# -*- coding: utf-8 -*-
"""UNKNOWN: the third answer. The state a reader needs was not available -- never that a condition was false.

It is a SENTINEL OBJECT, not None. None is already a value with a meaning: `ctx.last_write` is None when the
agent has written no deliverable, which is a fact. A substrate whose adapter declares no deliverable at all
sets it to UNKNOWN instead, so absence is never read where nothing can be known.
"""


class _Unknown(object):
    __slots__ = ()

    def __bool__(self):                       # never quietly usable in an `if`
        raise TypeError("UNKNOWN is not a truth value; handle it explicitly")

    __nonzero__ = __bool__

    def __repr__(self):
        return "UNKNOWN"

    def __copy__(self):                       # a sentinel stays itself through a copy (read-only views copy state)
        return self

    def __deepcopy__(self, memo):
        return self


UNKNOWN = _Unknown()
