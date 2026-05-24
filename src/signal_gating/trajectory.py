"""Trajectory capture: a verifiable, structured record of every signal hop."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from signal_gating.signal import Signal

# Base Signal envelope fields -- excluded from a Receipt's domain payload because
# they are already represented as typed Receipt fields.
_ENVELOPE_FIELDS = frozenset(
    {
        "id",
        "source",
        "timestamp",
        "priority",
        "trace_id",
        "correlation_id",
        "parent_id",
        "metadata",
    }
)


def _domain_payload(signal: Signal) -> dict[str, Any]:
    return {k: v for k, v in signal.model_dump().items() if k not in _ENVELOPE_FIELDS}


def _digest(core: dict[str, Any]) -> str:
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Receipt:
    """A verifiable, structured record of one signal crossing the mesh."""

    trace_id: str
    signal_id: str
    parent_id: str
    signal_type: str
    source: str
    target: str
    priority: int
    timestamp: float
    payload: dict[str, Any]
    digest: str

    @classmethod
    def from_signal(cls, signal: Signal, source: str, target: str) -> Receipt:
        core: dict[str, Any] = {
            "trace_id": signal.trace_id,
            "signal_id": signal.id,
            "parent_id": signal.parent_id,
            "signal_type": type(signal).__name__,
            "source": source,
            "target": target,
            "priority": signal.priority,
            "timestamp": time.time(),
            "payload": _domain_payload(signal),
        }
        return cls(
            trace_id=core["trace_id"],
            signal_id=core["signal_id"],
            parent_id=core["parent_id"],
            signal_type=core["signal_type"],
            source=core["source"],
            target=core["target"],
            priority=core["priority"],
            timestamp=core["timestamp"],
            payload=core["payload"],
            digest=_digest(core),
        )

    def verify(self) -> bool:
        core = {k: v for k, v in asdict(self).items() if k != "digest"}
        return _digest(core) == self.digest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryRecorder:
    """A mesh interceptor that records a Receipt for every signal hop.

    Attach with ``mesh.intercept(recorder)``. It is a pure observer: it returns
    every signal unchanged and never blocks delivery.

        recorder = TrajectoryRecorder()
        mesh.intercept(recorder)
        ...
        recorder.export_jsonl("runs.jsonl")
    """

    def __init__(self) -> None:
        self._receipts: list[Receipt] = []

    def __call__(self, signal: Signal, source: str, target: str) -> Signal:
        self._receipts.append(Receipt.from_signal(signal, source, target))
        return signal

    @property
    def receipts(self) -> list[Receipt]:
        return list(self._receipts)

    def trajectories(self) -> dict[str, list[Receipt]]:
        """Receipts grouped by trace_id, preserving capture order within each run."""
        grouped: dict[str, list[Receipt]] = {}
        for r in self._receipts:
            grouped.setdefault(r.trace_id, []).append(r)
        return grouped

    def export_jsonl(self, path: str | Path) -> int:
        """Write all receipts as JSON Lines. Returns the number written."""
        out = Path(path)
        with out.open("w", encoding="utf-8") as f:
            for r in self._receipts:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")
        return len(self._receipts)

    def clear(self) -> None:
        self._receipts.clear()
