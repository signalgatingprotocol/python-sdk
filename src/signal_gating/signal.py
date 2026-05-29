"""Core signal types: the fundamental unit of the Signal Gating Protocol."""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from signal_gating.registry import _auto_register, from_wire, to_wire, wire_type_of

T = TypeVar("T", bound="Signal")


class Signal(BaseModel):
    """An immutable, typed event that flows through the gating protocol.

    Subclass to create domain-specific signal types:

        class TaskSignal(Signal):
            task: str
            urgency: int = 0

    Signals are immutable by design. Use `evolve()` to create modified copies.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    source: str = ""
    timestamp: float = Field(default_factory=time.time)
    priority: int = 0
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    correlation_id: str = ""
    parent_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    # Optional override for this class's wire type name. Define in a subclass
    # body to pin a stable name independent of refactors (see registry module).
    __signal_type__: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register every Signal subclass under its wire type name on definition."""
        super().__pydantic_init_subclass__(**kwargs)
        _auto_register(cls)

    @classmethod
    def wire_type(cls) -> str:
        """The name this signal class is addressed by on the wire."""
        return wire_type_of(cls)

    def to_wire(self) -> dict[str, Any]:
        """Serialize to a self-describing, JSON-safe wire envelope.

        The envelope carries the wire type name and the full field set, so
        ``Signal.from_wire`` reconstructs the exact subclass — not a bare dict.
        """
        return to_wire(self)

    def to_json(self) -> str:
        """Serialize to a JSON string (the wire envelope, ``json.dumps``-ed)."""
        return json.dumps(self.to_wire(), default=str)

    @classmethod
    def from_wire(cls, data: dict[str, Any], *, strict: bool = True) -> Signal:
        """Reconstruct a signal from a wire envelope as its original subclass.

        Returns the registered subclass instance regardless of which class this
        is called on. With ``strict=False``, an unregistered type yields a
        best-effort base ``Signal`` instead of raising ``UnknownSignalType``.
        """
        return from_wire(data, strict=strict)

    @classmethod
    def from_json(cls, raw: str | bytes, *, strict: bool = True) -> Signal:
        """Reconstruct a signal from a JSON wire string."""
        return from_wire(json.loads(raw), strict=strict)

    def evolve(self: T, **kwargs: Any) -> T:
        """Create a new signal with updated fields, preserving the trace lineage."""
        data = self.model_dump()
        data.update(kwargs)
        if "id" not in kwargs:
            data["id"] = uuid4().hex
        return type(self).model_validate(data)

    def with_source(self: T, source: str) -> T:
        """Tag this signal with its source agent."""
        return self.evolve(source=source)

    def with_metadata(self: T, **kwargs: Any) -> T:
        """Add metadata entries."""
        merged = {**self.metadata, **kwargs}
        return self.evolve(metadata=merged)

    def child(self: T, **kwargs: Any) -> T:
        """Create a child signal that inherits this signal's trace lineage.

        The child preserves the trace_id for correlation and records this
        signal's id as parent_id, enabling full signal lineage trees.

            task = TaskSignal(task="analyze")
            subtask = task.child(task="analyze_section_1", priority=8)
            # subtask.parent_id == task.id
            # subtask.trace_id == task.trace_id
        """
        return self.evolve(parent_id=self.id, **kwargs)

    def __repr__(self) -> str:
        always_hide = {"id", "timestamp", "trace_id"}
        hide_if_default = {
            "source": "", "priority": 0, "correlation_id": "",
            "parent_id": "", "metadata": {},
        }
        fields: dict[str, Any] = {}
        for k, v in self.model_dump().items():
            if k in always_hide:
                continue
            if k in hide_if_default and v == hide_if_default[k]:
                continue
            fields[k] = v
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


# The subclass hook only fires for subclasses; register the base type by hand
# so a plain ``Signal`` round-trips through the wire format like any other.
_auto_register(Signal)
