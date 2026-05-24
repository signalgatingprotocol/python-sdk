"""Multi-agent example: agents communicating through a mesh with gates."""

import asyncio

from signal_gating import Agent, Gate, Mesh, Signal


class TaskSignal(Signal):
    task: str
    assigned_to: str = ""


class ResultSignal(Signal):
    task: str
    output: str


async def main():
    # Create agents
    coordinator = Agent("coordinator")
    analyst = Agent("analyst", gates=[Gate.by_priority(3)])
    reporter = Agent("reporter")

    # Analyst processes tasks and emits results
    @analyst.on(TaskSignal)
    async def analyze(signal: TaskSignal):
        print(f"  [analyst] Analyzing: {signal.task}")
        result = ResultSignal(
            task=signal.task,
            output=f"Analysis of '{signal.task}' complete",
            priority=signal.priority,
        )
        await analyst.emit(result)

    # Reporter receives results
    @reporter.on(ResultSignal)
    async def report(signal: ResultSignal):
        print(f"  [reporter] Report: {signal.output}")

    # Build mesh
    mesh = Mesh([coordinator, analyst, reporter])
    mesh.connect(coordinator, analyst)
    mesh.connect(analyst, reporter)

    async with mesh:
        print("Emitting tasks...")
        await coordinator.emit(TaskSignal(task="market trends", priority=8))
        await coordinator.emit(TaskSignal(task="minor update", priority=1))  # filtered by gate
        await coordinator.emit(TaskSignal(task="risk assessment", priority=9))
        await asyncio.sleep(0.1)

    print("\nAgent stats:")
    for agent in [coordinator, analyst, reporter]:
        print(f"  {agent.name}: {agent.stats}")


if __name__ == "__main__":
    asyncio.run(main())
