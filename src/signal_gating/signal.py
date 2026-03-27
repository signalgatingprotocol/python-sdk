"""Core signal types — the fundamental unit of the Signal Gating Protocol."""

from __future__ import annotations

import time
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

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
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

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

    def __repr__(self) -> str:
        fields = {k: v for k, v in self.model_dump().items() if k not in ("id", "timestamp", "trace_id", "metadata") or v}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
