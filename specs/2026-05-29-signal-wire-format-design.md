# Signal Wire Format & Type Registry

- **Date:** 2026-05-29
- **Status:** Built
- **Repo:** `signalgatingprotocol/python-sdk` (branch `claude/long-term-improvement-fetsD`)
- **Scope:** A self-describing serialization format for signals plus a type registry that reconstructs a serialized signal as its original subclass. The substrate beneath persistence, durable replay, and any future cross-process or cross-language transport.

## Problem

SGP calls itself a *protocol*, but every signal it has ever carried has lived and died inside a single Python process. There is no faithful serialization. `Signal.model_dump()` produces a `dict`, and there is no path back from that `dict` to `TaskSignal` — the type is gone. `Receipt` records a *lossy* audit projection (domain payload minus envelope), not a reconstructable signal.

The consequences are concrete and limiting:

- The dead-letter queue can collect failed signals but cannot survive a restart — `replay` only works within the lifetime of one process.
- `TrajectoryRecorder` can export receipts but cannot reconstruct the signals they describe.
- A distributed mesh (agents in different processes / hosts) is impossible: there is no wire contract.
- Nothing can persist a signal to a queue, log, or database and read it back as itself.

Every one of these wants the same missing primitive: **serialize a signal, move it across a boundary, get the same typed signal back.**

## Goal

Faithful, self-describing serialization with zero boilerplate for users:

```python
sig = TaskSignal(task="build", priority=5)
restored = Signal.from_json(sig.to_json())
assert type(restored) is TaskSignal and restored == sig
```

And make it immediately pay off via durable dead-letter recovery (persist → restart → reload → replay).

## Non-goals (v1)

- No transport. No sockets, brokers, or RPC. This is the *format and registry* a transport would stand on, nothing more.
- No schema evolution / migration engine. Versioning hooks exist (`WIRE_VERSION`, `__signal_type__`), but field migration is the caller's job for now.
- No change to `Receipt`'s lossy audit projection — it serves a different purpose (tamper-evident audit, not reconstruction).
- No new dependency. Stdlib `json` + the pydantic the SDK already requires.

## Wire envelope

```json
{"sgp": 1, "type": "TaskSignal", "data": {<every field, JSON-safe>}}
```

- `sgp` — `WIRE_VERSION`, the envelope schema version. Mismatch raises `SignalSerializationError`.
- `type` — the wire type name (see registry). Default is the class `__name__`.
- `data` — `model_dump(mode="json")`: the **full** field set (base `Signal` envelope *and* subclass domain fields), already reduced to JSON primitives. The round-trip is faithful (`id`, `trace_id`, `timestamp`, lineage, metadata all preserved), not merely structural.

## Registry

A module-level `name -> type[Signal]` map in `registry.py`.

- **Auto-registration.** `Signal.__pydantic_init_subclass__` registers every subclass on definition. Users write nothing. The base `Signal` and the tool-protocol signals register on import.
- **Wire name.** Defaults to `cls.__name__`. A class pins a stable name with `__signal_type__ = "task.v2"` in its own body — read from `cls.__dict__` only, so a subclass never silently inherits a parent's pinned name.
- **Collisions.** Two different classes claiming one wire name is a real possibility (it happens across test modules). Auto-registration is lenient: last definition wins, with a logged warning naming both classes. Explicit `register_signal(cls, name=...)` is strict: it raises `SignalSerializationError` on a genuine collision unless `override=True`. Re-registering the *same* class is idempotent.

This mirrors the existing `Receipt.signal_type` convention (bare `__name__`) while giving anyone who needs stability or cross-language addressing an explicit pin.

## API

On `Signal`:

- `signal.to_wire() -> dict` / `Signal.from_wire(dict, *, strict=True) -> Signal`
- `signal.to_json() -> str` / `Signal.from_json(str | bytes, *, strict=True) -> Signal`
- `Signal.wire_type() -> str`

Module-level (re-exported from the package): `register_signal`, `lookup_signal`, `registered_signals`, `to_wire`, `from_wire`, `WIRE_VERSION`.

Errors (in `errors.py`, under `SignalGatingError`): `SignalSerializationError`, and `UnknownSignalType` (its subclass).

### Unknown types

`from_wire` with `strict=True` (default) raises `UnknownSignalType` when no class is registered — honest failure that tells you to import/register the class. With `strict=False` it returns a best-effort base `Signal`: recognized envelope fields are kept, unmapped domain fields are preserved under `metadata["_sgp_unmapped"]`, and the original wire name under `metadata["_sgp_type"]`. Lossless, just untyped — useful for tolerant readers (log scrapers, dashboards) that don't have every type imported.

## Durable dead-letter recovery

The first consumer, proving the substrate. `DeadLetterQueue` gains:

- `to_jsonl(path) -> int` — write `{"entry": <failure context>, "signal": <wire>}` per line.
- `load_jsonl(path, *, strict=True) -> int` — reconstruct each signal as its original type, append (honoring `max_size`).

Persist on shutdown, reload on restart, `replay` into an inbox — failed work survives a crash and dispatches to the same handlers, because the types come back intact.

## Why this is the right foundation

It is the smallest change that turns "library" into "protocol." Every deferred ambition — distribution, persistent memory, replayable trajectories, cross-language clients — reduces to *move bytes across a boundary and rehydrate the type*. That is exactly, and only, what this provides. Transport, schema migration, and a chained ledger build on top; none of them are possible without it.
