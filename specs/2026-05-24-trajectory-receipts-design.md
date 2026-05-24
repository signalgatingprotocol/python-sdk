# Trajectory / Receipt Substrate

- **Date:** 2026-05-24
- **Status:** Design approved, building
- **Repo:** `signalgatingprotocol/python-sdk` (branch `feat/trajectories`)
- **Scope:** The foundational Hermes-harness concept: a verifiable, structured, exportable record of every agent execution. The substrate that persistent memory, skill distillation, RL/fine-tuning, and evals all build on. v1 captures agent-to-agent signal hops.

## Problem

Hermes-style autonomous agents learn from their own runs. That requires a record of what happened. SGP has lineage (`trace_id`/`parent_id`) and a `Tracer` (gate/dispatch spans for latency observability), but **no exportable execution record**: there is no way to say "here is the structured trajectory of this multi-agent run" for audit, learning, or training. This builds that — the org roadmap's "Receipts."

## Goal

Auto-capture, with one line, a structured `Receipt` for every signal that crosses the mesh, group them into per-run trajectories, and export them as JSONL. Because SGP signals are typed with lineage, these trajectories are structured (typed payloads, parent/child chains, routing) rather than flat chat logs.

## Non-goals (v1)

- No change to base `Signal`, `Gate`, `Agent`, `Mesh`. The recorder attaches through the existing public `mesh.intercept`.
- No storage/DB dependency in the core. Capture is in-memory; export is stdlib JSONL. Persistence is opt-in, later.
- v1 captures **agent-to-agent hops** (what an interceptor sees). The initial injected seed and gate-*dropped* signals are v2 (via `Tracer` integration — the Tracer already records those spans).
- Per-receipt hash now; a chained tamper-evident ledger is v2.
- Not memory, not skill distillation, not RL plumbing — those consume this substrate later.

## `Receipt`

A frozen, slots dataclass — a structured, verifiable record of one signal hop:

```python
@dataclass(frozen=True, slots=True)
class Receipt:
    trace_id: str        # the run
    signal_id: str       # this signal's id
    parent_id: str       # the signal that caused it ("" if none)
    signal_type: str     # type(signal).__name__
    source: str          # emitting agent name
    target: str          # receiving agent name
    priority: int
    timestamp: float     # wall-clock (time.time) at capture
    payload: dict        # the signal's domain fields (see below)
    digest: str          # sha256 hex over the canonical record — verifiable
```

- **`payload`** = `signal.model_dump()` minus the base `Signal` envelope fields (`id`, `source`, `timestamp`, `priority`, `trace_id`, `correlation_id`, `parent_id`, `metadata`). So for `TaskSignal(task="build")` → `{"task": "build"}`. The envelope is already represented by the typed Receipt fields; `payload` holds only the domain content.
- **`digest`** = `sha256(canonical_json).hexdigest()` where canonical JSON is the receipt's fields *except* `digest`, with sorted keys. Content-addressed and tamper-evident. A `Receipt.verify()` recomputes and compares.
- `to_dict()` returns a JSON-serializable dict (all fields, including `digest`).

## `TrajectoryRecorder`

A mesh interceptor that records a Receipt per hop. Non-invasive: it is just a callable attached via `mesh.intercept`.

```python
class TrajectoryRecorder:
    def __init__(self) -> None: ...

    def __call__(self, signal: Signal, source: str, target: str) -> Signal:
        # build a Receipt from signal + routing, store it, return signal unchanged
        ...
        return signal

    @property
    def receipts(self) -> list[Receipt]: ...           # capture order

    def trajectories(self) -> dict[str, list[Receipt]]: # grouped by trace_id, capture order
        ...

    def export_jsonl(self, path: str | Path) -> int:    # one Receipt dict per line; returns count
        ...

    def clear(self) -> None: ...
```

- Interceptors are called `(signal, source_name, target_name) -> signal` on every agent-to-agent hop (verified in `mesh.py`); returning the signal unchanged makes the recorder a pure observer (never blocks).
- `source`/`target` are agent names (strings), exactly what the interceptor receives.
- **Lineage note:** the recorder records whatever `trace_id`/`parent_id` each signal carries. Coherent per-run grouping therefore requires agents to *thread lineage* — `LLMAgent` does this by default (`_default_build` sets `trace_id`/`parent_id`); plain agents thread it by emitting `signal.child(...)` rather than a fresh `Signal()`. The recorder does not (and should not) invent lineage.

## Usage

```python
from signal_gating import TrajectoryRecorder

recorder = TrajectoryRecorder()
mesh.intercept(recorder)                 # capture on; zero base-class changes

async with mesh:
    await mesh.inject(planner, Topic(text="..."))
    await asyncio.sleep(0.1)

for trace_id, receipts in recorder.trajectories().items():
    print(trace_id, [r.signal_type for r in receipts])
recorder.export_jsonl("runs.jsonl")
```

## Architecture / taste

- **north-star:** uses only the public `mesh.intercept`; `mesh.py`/`agent.py`/`signal.py` untouched. Lives in a new `signal_gating/trajectory.py`. Core stays dependency-light (stdlib `json`, `hashlib`, `dataclasses`).
- **high-taste:** one concept (`Receipt`), one line to enable (`mesh.intercept(recorder)`), one line to export (`export_jsonl`). No configuration for the common path. `digest` makes "verifiable" concrete.

## Testing (deterministic, no LLM)

`tests/test_trajectory.py`:
- **capture:** a 2-hop mesh (A→B→C) run; `recorder.receipts` has the expected hops with correct `source`/`target`/`signal_type`/`payload`.
- **grouping + lineage:** the test's relay agents emit `signal.child(...)` so lineage threads; `trajectories()` groups all hops of one run under the seed's `trace_id`, and each hop's `parent_id` chains to the prior signal's `id`.
- **digest verifiable:** `r.verify()` is `True`; mutating a field and recomputing fails (tamper-evidence).
- **payload:** domain fields only (no `trace_id`/`source`/etc. duplicated inside `payload`).
- **export round-trip:** `export_jsonl(tmp)` writes N lines; each line is valid JSON that reconstructs the receipt dict; returns N.
- **observer:** the recorder never drops signals (delivery is unaffected when an interceptor is attached).

## Files

| File | Change |
| --- | --- |
| `src/signal_gating/trajectory.py` | New: `Receipt`, `TrajectoryRecorder`. |
| `src/signal_gating/__init__.py` | Export `Receipt`, `TrajectoryRecorder`; add to `__all__`. |
| `tests/test_trajectory.py` | New: deterministic tests above. |
| `README.md` | New "Trajectories" subsection under Core Primitives. |

## Success criteria

1. `mesh.intercept(TrajectoryRecorder())` captures a structured `Receipt` per hop; `trajectories()` groups by run; `export_jsonl` writes valid JSONL. Proven by tests, no LLM.
2. `Receipt.digest` is verifiable (recompute matches; tamper detected).
3. `import signal_gating` pulls no new third-party dependency; `ruff` + `mypy --strict` + full `pytest` green; base modules unchanged.
