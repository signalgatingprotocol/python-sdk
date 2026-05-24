# Trajectory / Receipt Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Auto-capture a verifiable, structured `Receipt` for every signal that crosses the mesh, grouped into per-run trajectories, exportable as JSONL.

**Architecture:** `Receipt` (frozen dataclass, content-hashed) + `TrajectoryRecorder` (a callable attached via the existing public `mesh.intercept`). New module `signal_gating/trajectory.py` imports only `Signal` — zero coupling to `mesh.py`/`agent.py`. Stdlib only (`json`, `hashlib`, `dataclasses`).

**Tech Stack:** Python 3.10+, pydantic v2 (`Signal`), pytest + pytest-asyncio (auto mode), ruff, mypy strict.

**Spec:** `specs/2026-05-24-trajectory-receipts-design.md`

---

## Conventions
- **Commits:** per CLAUDE.md, commit only when the user authorizes. Branch `feat/trajectories` (off `main`). **Never** a `claude/` prefix or agent attribution.
- **Working dir:** `/Users/p/code/github/signalgatingprotocol/python-sdk`. Tests via `pytest` (asyncio auto mode).
- **mypy:** `src/` only; test files needn't be strict.

## File structure
| File | Change |
| --- | --- |
| `src/signal_gating/trajectory.py` | New: `Receipt`, `TrajectoryRecorder` (+ private helpers). |
| `src/signal_gating/__init__.py` | Export `Receipt`, `TrajectoryRecorder`. |
| `tests/test_trajectory.py` | New: unit (Receipt) + mesh-integration (recorder) tests. |
| `README.md` | New "Trajectories" subsection. |

---

## Task 1: `Receipt`

**Files:** Create `src/signal_gating/trajectory.py`; Test `tests/test_trajectory.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory.py`:
```python
from signal_gating import Signal
from signal_gating.trajectory import Receipt


class Ping(Signal):
    n: int = 0


def test_from_signal_extracts_envelope_and_domain_payload():
    sig = Ping(n=7, priority=3)
    r = Receipt.from_signal(sig, source="a", target="b")
    assert r.trace_id == sig.trace_id
    assert r.signal_id == sig.id
    assert r.parent_id == sig.parent_id
    assert r.signal_type == "Ping"
    assert r.source == "a" and r.target == "b"
    assert r.priority == 3
    assert r.payload == {"n": 7}            # domain fields only; no envelope keys
    assert "trace_id" not in r.payload and "source" not in r.payload


def test_digest_verifies_and_detects_tampering():
    r = Receipt.from_signal(Ping(n=1), source="a", target="b")
    assert r.verify() is True
    tampered = Receipt(**{**r.to_dict(), "payload": {"n": 999}})
    assert tampered.verify() is False


def test_to_dict_is_json_serializable():
    import json
    r = Receipt.from_signal(Ping(n=1), source="a", target="b")
    d = r.to_dict()
    assert json.loads(json.dumps(d, default=str))["signal_type"] == "Ping"
    assert d["digest"] == r.digest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'signal_gating.trajectory'`.

- [ ] **Step 3: Implement `Receipt`**

Create `src/signal_gating/trajectory.py`:
```python
"""Trajectory capture: a verifiable, structured record of every signal hop."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from signal_gating.signal import Signal

# Base Signal envelope fields — excluded from a Receipt's domain payload because
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add verifiable Receipt record"
```

---

## Task 2: `TrajectoryRecorder`

**Files:** Modify `src/signal_gating/trajectory.py`; Test `tests/test_trajectory.py`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_trajectory.py`:
```python
import asyncio

import pytest

from signal_gating import Agent, Mesh
from signal_gating.trajectory import TrajectoryRecorder


async def _run_relay_mesh(recorder):
    """seed -> a -> b -> c; relays thread lineage via child()."""
    a, b, c = Agent("a"), Agent("b"), Agent("c")
    seen: list[Ping] = []
    done = asyncio.Event()

    @a.on(Ping)
    async def a_relay(sig: Ping) -> None:
        await a.emit(sig.child(n=sig.n + 1))

    @b.on(Ping)
    async def b_relay(sig: Ping) -> None:
        await b.emit(sig.child(n=sig.n + 1))

    @c.on(Ping)
    async def c_sink(sig: Ping) -> None:
        seen.append(sig)
        done.set()

    mesh = Mesh([a, b, c])
    mesh.intercept(recorder)
    mesh.connect(a, b)
    mesh.connect(b, c)

    seed = Ping(n=0)
    async with mesh:
        await mesh.inject(a, seed)
        await asyncio.wait_for(done.wait(), timeout=3.0)
    return seed, seen


async def test_recorder_captures_each_hop():
    recorder = TrajectoryRecorder()
    seed, seen = await _run_relay_mesh(recorder)
    assert len(seen) == 1
    rs = recorder.receipts
    assert len(rs) == 2                       # a->b and b->c (seed inject is not a hop)
    assert (rs[0].source, rs[0].target) == ("a", "b")
    assert (rs[1].source, rs[1].target) == ("b", "c")
    assert rs[0].payload == {"n": 1}
    assert rs[1].payload == {"n": 2}
    assert all(r.signal_type == "Ping" for r in rs)


async def test_trajectories_group_by_trace_and_chain_lineage():
    recorder = TrajectoryRecorder()
    seed, _ = await _run_relay_mesh(recorder)
    traj = recorder.trajectories()
    assert list(traj.keys()) == [seed.trace_id]      # one run
    run = traj[seed.trace_id]
    assert len(run) == 2
    assert run[0].parent_id == seed.id               # first hop descends from the seed
    assert run[1].parent_id == run[0].signal_id      # lineage chains hop-to-hop


async def test_all_receipts_verify():
    recorder = TrajectoryRecorder()
    await _run_relay_mesh(recorder)
    assert all(r.verify() for r in recorder.receipts)


async def test_export_jsonl_round_trips(tmp_path):
    import json
    recorder = TrajectoryRecorder()
    await _run_relay_mesh(recorder)
    out = tmp_path / "runs.jsonl"
    n = recorder.export_jsonl(out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert n == len(lines) == 2
    first = json.loads(lines[0])
    assert first["source"] == "a" and first["payload"] == {"n": 1}


async def test_recorder_is_pure_observer():
    """Attaching a recorder must not drop signals."""
    recorder = TrajectoryRecorder()
    _, seen = await _run_relay_mesh(recorder)
    assert len(seen) == 1                            # delivery unaffected


def test_clear_empties_receipts():
    recorder = TrajectoryRecorder()
    recorder._receipts.append(Receipt.from_signal(Ping(n=1), "a", "b"))  # type: ignore[attr-defined]
    recorder.clear()
    assert recorder.receipts == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory.py -q -k "recorder or trajectories or verify or export or observer or clear"`
Expected: FAIL — `ImportError: cannot import name 'TrajectoryRecorder'`.

- [ ] **Step 3: Implement `TrajectoryRecorder`**

Append to `src/signal_gating/trajectory.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add TrajectoryRecorder mesh interceptor + JSONL export"
```

---

## Task 3: Exports + README + verification gate

**Files:** Modify `src/signal_gating/__init__.py`, `README.md`.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_trajectory.py`:
```python
def test_exports():
    import signal_gating
    assert hasattr(signal_gating, "Receipt")
    assert hasattr(signal_gating, "TrajectoryRecorder")
    assert "Receipt" in signal_gating.__all__
    assert "TrajectoryRecorder" in signal_gating.__all__
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_trajectory.py -q -k exports`
Expected: FAIL — `AssertionError`.

- [ ] **Step 3: Add the exports**

In `src/signal_gating/__init__.py`, add an import line (after the existing `from signal_gating.tracing import ...` line, keeping import grouping):
```python
from signal_gating.trajectory import Receipt, TrajectoryRecorder
```
Add `"Receipt"` and `"TrajectoryRecorder"` to `__all__`, keeping it alphabetical (`"Receipt"` after `"PriorityChannel"`/before `"Signal"`; `"TrajectoryRecorder"` after `"ToolSpec"`/before `"Tracer"`).

- [ ] **Step 4: Add the README section**

In `README.md`, add a `### Trajectories` subsection at the end of the "Core Primitives" area, immediately before `## Architecture`:
````markdown
### Trajectories

Capture a verifiable, structured record of every signal that crosses the mesh,
exportable as JSONL for audit, learning, or training. Attach a recorder with one
line; it is a pure observer (never blocks):

```python
from signal_gating import TrajectoryRecorder

recorder = TrajectoryRecorder()
mesh.intercept(recorder)

async with mesh:
    await mesh.inject(planner, Topic(text="..."))

recorder.trajectories()              # {trace_id: [Receipt, ...]}, grouped per run
recorder.export_jsonl("runs.jsonl")  # one Receipt per line
```

Each `Receipt` carries the signal's lineage (`trace_id` / `parent_id`), routing
(`source` -> `target`), typed domain `payload`, and a `digest` (sha256) so the
record is tamper-evident: `receipt.verify()`.
````

- [ ] **Step 5: Run tests + full gate**

Run: `pytest tests/test_trajectory.py -q` → all pass.
Run: `ruff check .` → clean.
Run: `mypy src/` → `Success: no issues found`.
Run: `pytest -q 2>&1 | tail -3` → full suite green.
Run: `python3 -c "import sys, signal_gating; assert 'openai' not in sys.modules; print('clean')"` → `clean`.

- [ ] **Step 6: Commit** (when authorized)

```bash
git add src/signal_gating/__init__.py README.md tests/test_trajectory.py
git commit -m "feat(trajectory): export Receipt/TrajectoryRecorder and document"
```

---

## Done criteria (maps to spec success criteria)

1. `mesh.intercept(TrajectoryRecorder())` captures a `Receipt` per hop; `trajectories()` groups by run; `export_jsonl` writes valid JSONL (Tasks 2–3 tests).
2. `Receipt.digest` verifies and detects tampering (Task 1 test).
3. `trajectory.py` imports only `Signal`; base modules unchanged; `ruff` + `mypy --strict` + full `pytest` green; `import signal_gating` pulls no `openai` (Task 3 Step 5).
