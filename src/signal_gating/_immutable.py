"""Recursively immutable, built-in-compatible signal containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn


def _immutable(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError("signal containers are immutable")


class _FrozenDict(dict[Any, Any]):
    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenList(list[Any]):
    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


class _FrozenSet(set[Any]):
    add = _immutable
    clear = _immutable
    difference_update = _immutable
    discard = _immutable
    intersection_update = _immutable
    pop = _immutable
    remove = _immutable
    symmetric_difference_update = _immutable
    update = _immutable
    __ior__ = _immutable
    __iand__ = _immutable
    __ixor__ = _immutable
    __isub__ = _immutable


def deep_freeze(value: Any) -> Any:
    """Copy mutable containers recursively into immutable built-in subclasses."""
    if isinstance(value, Mapping):
        return _FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(deep_freeze(item) for item in value)
    if isinstance(value, set):
        return _FrozenSet(deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_freeze(item) for item in value)
    return value
