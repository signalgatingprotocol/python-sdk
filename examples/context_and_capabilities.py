"""Agent-native patterns: AgentContext, capability discovery, and interceptors."""

import asyncio

from signal_gating import Agent, AgentContext, Mesh, Signal


class TaskSignal(Signal):
    task: str


class ResultSignal(Signal):
    result: str


async def main():
    # Create specialized agents
    analyst = Agent("analyst")
    summarizer = Agent("summarizer")
    coordinator = Agent("coordinator")

    # Handlers use AgentContext (no closure needed)
    @analyst.on(TaskSignal)
    async def analyze(signal: TaskSignal, ctx: AgentContext):
        ctx.state["tasks_done"] = ctx.state.get("tasks_done", 0) + 1
        await ctx.emit(ResultSignal(result=f"Analysis of '{signal.task}' complete"))

    @summarizer.on(TaskSignal)
    async def summarize(signal: TaskSignal, ctx: AgentContext):
        ctx.state["tasks_done"] = ctx.state.get("tasks_done", 0) + 1
        await ctx.emit(ResultSignal(result=f"Summary of '{signal.task}' ready"))

    results: list[str] = []

    @coordinator.on(ResultSignal)
    async def collect(signal: ResultSignal):
        results.append(signal.result)
        print(f"  [coordinator] Received: {signal.result}")

    # Build mesh with interceptor
    mesh = Mesh([coordinator, analyst, summarizer])

    # Declare capabilities for discovery
    mesh.declare_capabilities(analyst, "analysis", "research")
    mesh.declare_capabilities(summarizer, "summarization", "research")

    # Interceptor: log all signal flow
    def audit(signal: Signal, source: str, target: str) -> Signal:
        print(f"  [audit] {source} -> {target}: {type(signal).__name__}")
        return signal

    mesh.intercept(audit)

    # Connect topology
    mesh.connect(coordinator, analyst)
    mesh.connect(coordinator, summarizer)
    mesh.connect(analyst, coordinator)
    mesh.connect(summarizer, coordinator)

    # Discover agents by capability
    researchers = mesh.find_capable("research")
    print(f"Agents capable of 'research': {[a.name for a in researchers]}")
    print(f"Analyst capabilities: {mesh.agent_capabilities(analyst)}")

    async with mesh:
        await coordinator.emit(TaskSignal(task="market trends", priority=5))
        await mesh.wait_idle()

    # Graceful shutdown already happened via __aexit__
    print(f"\nResults collected: {len(results)}")
    for r in results:
        print(f"  - {r}")

    print("\nAgent states:")
    for agent in [analyst, summarizer]:
        print(f"  {agent.name}: {agent.state}")


if __name__ == "__main__":
    asyncio.run(main())
