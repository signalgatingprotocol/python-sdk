"""Durable recovery example: persist failed signals, then replay after a restart.

Signals serialize to a self-describing wire envelope and reconstruct as their
original subclass. That makes the dead-letter queue survivable: persist it on
shutdown, reload it in a fresh process, and replay the failed work — the signals
come back as their real types and dispatch to the same handlers.

Run it twice with the same path to watch a "restart" recover prior failures:

    python examples/durable_recovery.py /tmp/sgp-dlq.jsonl
"""

import asyncio
import logging
import sys
from pathlib import Path

from signal_gating import Agent, DeadLetterQueue, Mesh, Signal

# The failures in this example are deliberate; quiet the agent's error logs so
# the recovery narrative reads cleanly.
logging.getLogger("signal_gating.agent").setLevel(logging.CRITICAL)


class TaskSignal(Signal):
    task: str
    attempt: int = 1


async def main(dlq_path: Path) -> None:
    worker = Agent("worker")
    succeeded: list[str] = []

    @worker.on(TaskSignal)
    async def handle(signal: TaskSignal) -> None:
        # First time around, every task "fails" and lands in the dead-letter queue.
        if signal.attempt == 1:
            raise RuntimeError(f"transient failure on {signal.task!r}")
        succeeded.append(signal.task)
        print(f"  processed {signal.task!r} on attempt {signal.attempt}")

    # On restart, recover any signals persisted by a previous run and replay them
    # with a bumped attempt so they go down the success path this time.
    if dlq_path.exists():
        recovered = DeadLetterQueue()
        n = recovered.load_jsonl(dlq_path)
        print(f"Recovered {n} signal(s) from {dlq_path}")
        for sig in recovered.drain():
            assert isinstance(sig, TaskSignal)  # reconstructed as its real type
            await worker.inbox.send(sig.evolve(attempt=2))
        dlq_path.unlink()

    mesh = Mesh([worker])
    async with mesh:
        # Fresh work — these fail on attempt 1 and get dead-lettered.
        for task in ("index", "embed", "summarize"):
            await worker.inbox.send(TaskSignal(task=task))
        await asyncio.sleep(0.1)

    if worker.dead_letters.count:
        written = worker.dead_letters.to_jsonl(dlq_path)
        print(f"Persisted {written} failed signal(s) to {dlq_path}")
        print("Run again with the same path to recover and replay them.")

    if succeeded:
        print(f"Succeeded this run: {succeeded}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/sgp-dlq.jsonl")
    asyncio.run(main(path))
