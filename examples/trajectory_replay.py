"""Trajectory replay example: record a multi-agent run, persist it, replay it.

A TrajectoryRecorder captures verifiable Receipts for signal-carrying mesh
events. Because each receipt carries the full signal wire envelope, a trajectory
isn't just an audit log you can read — persisted signal events can be
reconstructed exactly in a fresh process, ready to inspect or re-deliver through
a compatible mesh. This is SGP signal replay, not Claude Agent SDK session
resume.

Run it twice with the same path to watch a "restart" recover a prior run:

    python examples/trajectory_replay.py /tmp/sgp-trajectory.jsonl
"""

import asyncio
import sys
from pathlib import Path

from signal_gating import Agent, Mesh, Signal, TrajectoryRecorder, TrajectoryReplayRunner


class Step(Signal):
    name: str
    n: int = 0


async def main(path: Path) -> None:
    # If a prior run was persisted, reload it and reconstruct the typed signals.
    if path.exists():
        reloaded = TrajectoryRecorder()
        count = reloaded.load_jsonl(path)
        print(f"Reloaded {count} receipt(s) from {path}")
        print(f"  all verify: {all(r.verify() for r in reloaded.receipts)}")
        signals = reloaded.replay()  # reconstructed as their original Step type
        for sig in signals:
            assert isinstance(sig, Step)  # not a dict, not a base Signal
            print(f"  replayed {type(sig).__name__}(name={sig.name!r}, n={sig.n})")

        replay_worker = Agent("a")
        replay_seen = asyncio.Event()

        @replay_worker.on(Step)
        async def replay_sink(sig: Step) -> None:
            print(f"  delivered to replay mesh: {sig.name!r}, n={sig.n}")
            replay_seen.set()

        replay_mesh = Mesh([replay_worker])
        runner = TrajectoryReplayRunner.from_recorder(reloaded)
        async with replay_mesh:
            result = await runner.replay_into(replay_mesh, strict_targets=False)
            if result.delivered:
                await asyncio.wait_for(replay_seen.wait(), timeout=3.0)
        print(f"  delivery replay: delivered={result.delivered}, skipped={result.skipped}")
        path.unlink()
        return

    # First run: a 3-stage pipeline a -> b -> c, threading lineage with child().
    a, b, c = Agent("a"), Agent("b"), Agent("c")
    done = asyncio.Event()

    @a.on(Step)
    async def a_relay(sig: Step) -> None:
        await a.emit(sig.child(name="b", n=sig.n + 1))

    @b.on(Step)
    async def b_relay(sig: Step) -> None:
        await b.emit(sig.child(name="c", n=sig.n + 1))

    @c.on(Step)
    async def c_sink(sig: Step) -> None:
        done.set()

    recorder = TrajectoryRecorder()
    mesh = Mesh([a, b, c])
    mesh.record(recorder)  # one line: capture direct orchestration + routes
    mesh.connect(a, b)
    mesh.connect(b, c)

    async with mesh:
        await mesh.inject(a, Step(name="a", n=0))
        await asyncio.wait_for(done.wait(), timeout=3.0)

    for trace_id, receipts in recorder.trajectories().items():
        events = " -> ".join(f"{r.action}({r.source}:{r.target})" for r in receipts)
        print(f"Captured run {trace_id[:8]}: {events}")

    written = recorder.export_jsonl(path)
    print(f"Persisted {written} receipt(s) to {path}")
    print("Run again with the same path to reload and replay them.")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/sgp-trajectory.jsonl")
    asyncio.run(main(out))
