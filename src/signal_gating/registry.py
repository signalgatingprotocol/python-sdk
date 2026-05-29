"""Signal wire format and type registry.

A protocol that can only live inside one Python process is a library, not a
protocol. This module is the substrate that lets a signal leave the process it
was born in and come back as itself: a self-describing JSON envelope plus a
registry that maps a wire type name back to the concrete ``Signal`` subclass.

Everything durable builds on this: persisting a dead-letter queue to disk and
replaying it after a restart, shipping signals over a transport, recording a
trajectory you can faithfully reconstruct later. ``model_dump()`` alone is
lossy in the way that matters — it gives you a ``dict``, and there is no way
back to ``TaskSignal``. The registry closes that loop.

Wire envelope (schema version ``WIRE_VERSION``)::

    {"sgp": 1, "type": "TaskSignal", "data": {<every field, JSON-safe>}}

``data`` is the full ``model_dump(mode="json")`` — both the base ``Signal``
envelope fields (``id``, ``trace_id``, ``timestamp``, ...) and the subclass's
domain fields — so a round-trip is faithful, not just structurally similar.

Subclasses register themselves automatically (see ``Signal``), so most callers
never touch this module directly; they just use ``signal.to_json()`` and
``Signal.from_json(...)``. Reach for ``register_signal`` only to assign an
explicit wire name or resolve a name collision between two classes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar, overload

from signal_gating.errors import SignalSerializationError, UnknownSignalType

if TYPE_CHECKING:
    from signal_gating.signal import Signal

logger = logging.getLogger("signal_gating.registry")

WIRE_VERSION = 1
"""Wire envelope schema version. Bumped only on a breaking envelope change."""

_REGISTRY: dict[str, type[Signal]] = {}

S = TypeVar("S", bound="Signal")


def wire_type_of(cls: type[Signal]) -> str:
    """Return the wire type name for a Signal class.

    Defaults to the class's ``__name__``. A class may pin a stable name by
    setting ``__signal_type__`` in its own body (inherited values are ignored,
    so subclasses never silently share a parent's wire name).
    """
    own = cls.__dict__.get("__signal_type__")
    return own if isinstance(own, str) and own else cls.__name__


@overload
def register_signal(cls: type[S], *, name: str | None = ..., override: bool = ...) -> type[S]: ...
@overload
def register_signal(
    cls: None = ..., *, name: str | None = ..., override: bool = ...
) -> Any: ...


def register_signal(
    cls: type[S] | None = None,
    *,
    name: str | None = None,
    override: bool = False,
) -> Any:
    """Register a ``Signal`` subclass under a wire type name.

    Usable as a decorator or a direct call::

        @register_signal(name="task.v2")
        class TaskSignal(Signal):
            task: str

        register_signal(TaskSignal)  # equivalent for the default name

    Subclasses of ``Signal`` register themselves on definition, so this is only
    needed to assign an explicit ``name`` or to re-point a wire name with
    ``override=True``. Registering a *different* class under an existing name
    without ``override`` raises ``SignalSerializationError`` — collisions are
    a bug, not something to paper over silently.
    """

    def _apply(target: type[S]) -> type[S]:
        wire_name = name or wire_type_of(target)
        existing = _REGISTRY.get(wire_name)
        if existing is not None and existing is not target and not override:
            raise SignalSerializationError(
                f"Wire type {wire_name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; pass override=True, "
                f"or set __signal_type__ to disambiguate "
                f"{target.__module__}.{target.__qualname__}."
            )
        _REGISTRY[wire_name] = target
        return target

    if cls is None:
        return _apply
    return _apply(cls)


def _auto_register(cls: type[Signal]) -> None:
    """Lenient registration used by ``Signal.__pydantic_init_subclass__``.

    Must never raise: a failure here would break class definition itself.
    Logs a warning when two genuinely different classes claim the same wire
    name (the later definition wins, mirroring normal Python rebinding).
    """
    wire_name = wire_type_of(cls)
    existing = _REGISTRY.get(wire_name)
    if existing is not None and existing is not cls:
        logger.warning(
            "Signal wire type %r re-registered: %s.%s now shadows %s.%s. "
            "Set __signal_type__ on one of them to keep both addressable.",
            wire_name,
            cls.__module__,
            cls.__qualname__,
            existing.__module__,
            existing.__qualname__,
        )
    _REGISTRY[wire_name] = cls


def lookup_signal(name: str) -> type[Signal] | None:
    """Return the ``Signal`` subclass registered under ``name``, or ``None``."""
    return _REGISTRY.get(name)


def registered_signals() -> dict[str, type[Signal]]:
    """Return a copy of the wire-name → class registry (for introspection)."""
    return dict(_REGISTRY)


def to_wire(signal: Signal) -> dict[str, Any]:
    """Serialize a signal to a self-describing, JSON-safe wire envelope.

    The payload is ``model_dump(mode="json")``, so every value is already a
    JSON primitive — the result can be handed straight to ``json.dumps`` or any
    transport. Domain fields holding non-serializable objects will raise here;
    wire-transportable signals must carry serializable content.
    """
    return {
        "sgp": WIRE_VERSION,
        "type": wire_type_of(type(signal)),
        "data": signal.model_dump(mode="json"),
    }


def from_wire(data: dict[str, Any], *, strict: bool = True) -> Signal:
    """Reconstruct a signal from a wire envelope as its original subclass.

    With ``strict=True`` (default), an unregistered type raises
    ``UnknownSignalType``. With ``strict=False``, the payload is reconstructed
    into a best-effort base ``Signal``: recognised envelope fields are kept and
    any unmapped domain fields are preserved under
    ``metadata["_sgp_unmapped"]``, alongside ``metadata["_sgp_type"]`` recording
    the original wire name — lossless, just untyped.
    """
    if not isinstance(data, dict):
        raise SignalSerializationError(
            f"wire envelope must be a dict, got {type(data).__name__}"
        )
    version = data.get("sgp")
    if version != WIRE_VERSION:
        raise SignalSerializationError(
            f"unsupported wire version {version!r} (expected {WIRE_VERSION})"
        )
    type_name = data.get("type")
    payload = data.get("data")
    if not isinstance(type_name, str) or not isinstance(payload, dict):
        raise SignalSerializationError("malformed wire envelope: missing 'type'/'data'")

    cls = _REGISTRY.get(type_name)
    if cls is None:
        if strict:
            raise UnknownSignalType(type_name)
        return _reconstruct_untyped(payload, type_name)
    try:
        return cls.model_validate(payload)
    except SignalSerializationError:
        raise
    except Exception as e:  # pydantic ValidationError and friends
        raise SignalSerializationError(
            f"failed to reconstruct {type_name!r} from wire data: {e}"
        ) from e


def _reconstruct_untyped(payload: dict[str, Any], type_name: str) -> Signal:
    """Best-effort base ``Signal`` for an unregistered type (strict=False)."""
    from signal_gating.signal import Signal

    envelope_fields = set(Signal.model_fields)
    known = {k: v for k, v in payload.items() if k in envelope_fields}
    unmapped = {k: v for k, v in payload.items() if k not in envelope_fields}
    metadata = dict(known.get("metadata") or {})
    metadata["_sgp_type"] = type_name
    if unmapped:
        metadata["_sgp_unmapped"] = unmapped
    known["metadata"] = metadata
    return Signal.model_validate(known)
