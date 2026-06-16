"""Script: a script-held workflow runtime with checkpointed, resumable steps.

Unrelated to mesh.workflow(), which is a one-shot step chain; a Script is a
user-authored coroutine that owns the loop and the branching, with the runtime
contributing bounded fan-out, an agent budget, and resume-as-cache.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from signal_gating.agent import Agent
from signal_gating.errors import BudgetExceeded, SignalSerializationError
from signal_gating.mesh import Mesh
from signal_gating.registry import from_wire
from signal_gating.signal import Signal
from signal_gating.trajectory import _digest, domain_payload


def step_key(
    script: str,
    phase: str,
    signal: Signal,
    *,
    occurrence: int,
    target: str | None = None,
) -> str:
    """Content-addressed step identity: stable across runs, volatile-field-free.

    ``target`` is set only by ctx.run (a caller-named agent); fan_out/spawn
    omit it because worker assignment is incidental to a step's identity.
    """
    return _digest(
        {
            "script": script,
            "phase": phase,
            "target": target,
            "type": signal.wire_type(),
            "payload": domain_payload(signal),
            "occurrence": occurrence,
        }
    )


class CheckpointStore:
    """Append-only JSONL of completed step results; per-record sha256 digests.

    ``path=None`` is a per-run in-memory store: no persistence, no resume.
    Duplicate keys on load: last record wins.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._results: dict[str, Signal] = {}
        if self._path is not None and self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    record = json.loads(line)
                    core = {k: record[k] for k in ("key", "script", "phase", "result")}
                    if record["digest"] != _digest(core):
                        raise SignalSerializationError(
                            f"checkpoint record {n} failed digest verification"
                        )
                    self._results[record["key"]] = from_wire(record["result"])

    def __len__(self) -> int:
        return len(self._results)

    def get(self, key: str) -> Signal | None:
        return self._results.get(key)

    def put(self, key: str, script: str, phase: str, result: Signal) -> None:
        self._results[key] = result
        if self._path is None:
            return
        core = {"key": key, "script": script, "phase": phase, "result": result.to_wire()}
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**core, "digest": _digest(core)}, sort_keys=True) + "\n")


class ScriptContext:
    def __init__(self, script: Script, args: Any) -> None:
        self.args = args
        self._script = script
        self._phase = ""
        self._occurrences: dict[str, int] = {}
        self._spawn_counter = itertools.count()

    @asynccontextmanager
    async def phase(self, name: str) -> AsyncIterator[ScriptContext]:
        previous, self._phase = self._phase, name
        try:
            yield self
        finally:
            self._phase = previous

    async def run(
        self,
        target: Agent | str,
        signal: Signal,
        *,
        timeout: float = 30.0,
        key: str | None = None,
    ) -> Signal:
        """One checkpointed mesh.request. Failure — including a dead-lettered
        target handler that never replies — surfaces as asyncio.TimeoutError;
        the script decides what that means. The runtime never retries."""
        name = target if isinstance(target, str) else target.name
        resolved = key if key is not None else self._key(signal, target=name)
        return await self._execute(resolved, name, signal, timeout)

    async def fan_out(
        self,
        targets: Sequence[Agent | str],
        signals: Sequence[Signal],
        *,
        timeout: float = 30.0,
    ) -> list[Signal]:
        """Round-robin signals across targets under the concurrency semaphore;
        results in input order. One target is serial by construction."""
        if not targets:
            raise ValueError("fan_out requires at least one target")
        names = [t if isinstance(t, str) else t.name for t in targets]
        steps = [
            (self._key(signal), names[i % len(names)], signal)
            for i, signal in enumerate(signals)
        ]
        return list(
            await asyncio.gather(
                *(self._execute(k, name, s, timeout) for k, name, s in steps)
            )
        )

    async def spawn(
        self,
        factory: Callable[[], Agent],
        signal: Signal,
        *,
        timeout: float = 30.0,
    ) -> Signal:
        """Ephemeral agent: build, uniquely rename, add, start, one checkpointed
        request, then stop and mesh.remove() — topology as found."""
        key = self._key(signal)
        cached = self._script._store.get(key)
        if cached is not None:
            return cached
        agent = factory()
        agent.name = f"{agent.name}#{next(self._spawn_counter)}"
        mesh = self._script._mesh
        mesh.add(agent)
        try:
            await agent.start()
            return await self._execute(key, agent.name, signal, timeout)
        finally:
            await mesh.remove(agent)

    def _key(self, signal: Signal, *, target: str | None = None) -> str:
        content = step_key(
            self._script.name, self._phase, signal, occurrence=0, target=target
        )
        occurrence = self._occurrences.get(content, 0)
        self._occurrences[content] = occurrence + 1
        return step_key(
            self._script.name,
            self._phase,
            signal,
            occurrence=occurrence,
            target=target,
        )

    async def _execute(
        self, key: str, target: str, signal: Signal, timeout: float
    ) -> Signal:
        cached = self._script._store.get(key)
        if cached is not None:
            return cached
        self._script._spent += 1
        if self._script._spent > self._script._budget:
            raise BudgetExceeded(self._script._budget, key)
        async with self._script._semaphore:
            result = await self._script._mesh.request(target, signal, timeout=timeout)
        self._script._store.put(key, self._script.name, self._phase, result)
        return result


class Script:
    def __init__(
        self,
        name: str,
        mesh: Mesh,
        fn: Callable[[ScriptContext], Awaitable[Any]],
        *,
        max_concurrency: int = 16,
        budget: int = 1000,
        store: CheckpointStore | None = None,
    ) -> None:
        self.name = name
        self._mesh = mesh
        self._fn = fn
        self._budget = budget
        self._spent = 0
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._store = store if store is not None else CheckpointStore()

    async def run(self, args: Any = None) -> Any:
        self._spent = 0
        return await self._fn(ScriptContext(self, args))
