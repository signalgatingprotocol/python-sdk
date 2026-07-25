"""Recursively immutable signal containers with safe serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from copy import deepcopy
from types import MappingProxyType
from typing import Any, NoReturn


def _immutable(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError("signal containers are immutable")


class _FrozenDict(Mapping[Any, Any]):
    """Immutable mapping backed by an unaliased read-only proxy."""

    __slots__ = ("_data",)
    _data: Mapping[Any, Any]

    def __init__(self, values: Mapping[Any, Any]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        copied = _FrozenDict(
            {deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()}
        )
        memo[id(self)] = copied
        return copied

    def copy(self) -> dict[Any, Any]:
        return dict(self._data)

    def __or__(self, other: object) -> Any:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self) | dict(other)

    def __ror__(self, other: object) -> Any:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(other) | dict(self)

    __setattr__ = _immutable
    __delattr__ = _immutable
    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenList(Sequence[Any]):
    """Immutable sequence that preserves list equality and serialization shape."""

    __slots__ = ("_values",)
    _values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        object.__setattr__(self, "_values", tuple(values))

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return list(self._values[index])
        return self._values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(list(self._values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenList):
            return bool(self._values == other._values)
        if isinstance(other, list):
            return bool(list(self._values) == other)
        return False

    def __copy__(self) -> _FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenList:
        copied = _FrozenList([deepcopy(value, memo) for value in self])
        memo[id(self)] = copied
        return copied

    def copy(self) -> list[Any]:
        return list(self._values)

    def __add__(self, other: object) -> Any:
        if isinstance(other, _FrozenList):
            return list(self._values) + list(other._values)
        if isinstance(other, list):
            return list(self._values) + other
        return NotImplemented

    def __radd__(self, other: object) -> Any:
        if isinstance(other, _FrozenList):
            return list(other._values) + list(self._values)
        if isinstance(other, list):
            return other + list(self._values)
        return NotImplemented

    def __mul__(self, count: object) -> Any:
        return list(self._values) * count  # type: ignore[operator]

    def __rmul__(self, count: object) -> Any:
        return list(self._values) * count  # type: ignore[operator]

    __setattr__ = _immutable
    __delattr__ = _immutable
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


class _FrozenSet(Set[Any]):
    """Immutable set backed by a frozenset."""

    __slots__ = ("_values",)
    _values: frozenset[Any]

    def __init__(self, values: Set[Any]) -> None:
        object.__setattr__(self, "_values", frozenset(values))

    def __contains__(self, value: object) -> bool:
        return value in self._values

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(set(self._values))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Set) and self._values == frozenset(other)

    def __copy__(self) -> _FrozenSet:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenSet:
        copied = _FrozenSet({deepcopy(value, memo) for value in self})
        memo[id(self)] = copied
        return copied

    def copy(self) -> set[Any]:
        return set(self._values)

    def union(self, *others: Iterable[Any]) -> set[Any]:
        return set(self._values).union(*others)

    def intersection(self, *others: Iterable[Any]) -> set[Any]:
        return set(self._values).intersection(*others)

    def difference(self, *others: Iterable[Any]) -> set[Any]:
        return set(self._values).difference(*others)

    def symmetric_difference(self, other: Iterable[Any]) -> set[Any]:
        return set(self._values).symmetric_difference(other)

    def issubset(self, other: Iterable[Any]) -> bool:
        return set(self._values).issubset(other)

    def issuperset(self, other: Iterable[Any]) -> bool:
        return set(self._values).issuperset(other)

    def isdisjoint(self, other: Iterable[Any]) -> bool:
        return set(self._values).isdisjoint(other)

    def __or__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(self._values) | set(other)

    def __ror__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(other) | set(self._values)

    def __and__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(self._values) & set(other)

    def __rand__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(other) & set(self._values)

    def __sub__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(self._values) - set(other)

    def __rsub__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(other) - set(self._values)

    def __xor__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(self._values) ^ set(other)

    def __rxor__(self, other: object) -> Any:
        if not isinstance(other, Set):
            return NotImplemented
        return set(other) ^ set(self._values)

    __setattr__ = _immutable
    __delattr__ = _immutable
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
    """Copy mutable containers recursively into composition-based wrappers."""
    if isinstance(value, Mapping):
        return _FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList([deep_freeze(item) for item in value])
    if isinstance(value, set):
        return _FrozenSet({deep_freeze(item) for item in value})
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    """Restore frozen wrappers to their original built-in container shapes."""
    if isinstance(value, _FrozenDict):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [deep_thaw(item) for item in value]
    if isinstance(value, _FrozenSet):
        return {deep_thaw(item) for item in value}
    if isinstance(value, tuple):
        return tuple(deep_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_thaw(item) for item in value)
    return value
