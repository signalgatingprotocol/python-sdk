"""Mesh — the agent network topology that ties everything together."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any
from uuid import uuid4

from signal_gating.agent import Agent
from signal_gating.errors import MeshError
from signal_gating.gate import Gate
from signal_gating.signal import Signal
from signal_gating.tracing import Tracer

logger = logging.getLogger("signal_gating.mesh")

Interceptor = Callable[[Signal, str, str], Signal | None | Awaitable[Signal | None]]


class Edge:
    """A directional connection between two agents, optionally gated."""

    def __init__(self, source: Agent, target: Agent, gate: Gate | None = None):
        self.source = source
        self.target = target
        self.gate = gate


class Mesh:
    """A network of agents connected by gated edges.

    The mesh manages agent lifecycles, signal routing, and observability.
    When a tracer is attached, all signal flow is automatically traced.

        mesh = Mesh()
        mesh.add(planner)
        mesh.add(worker)
        mesh.connect(planner, worker, gate=priority_gate)

        async with mesh:
            await planner.emit(TaskSignal(task="build"))

        # Inspect traces
        for span in mesh.tracer.get_trace(trace_id):
            print(f"{span.agent} -> {span.gate}: {span.action}")
    """

    def __init__(
        self,
        agents: list[Agent] | None = None,
        tracer: Tracer | None = None,
    ):
        self._agents: dict[str, Agent] = {}
        self._edges: list[Edge] = []
        self._running = False
        self._interceptors: list[Interceptor] = []
        self._capabilities: dict[str, set[str]] = {}
        self.tracer = tracer or Tracer()
        for agent in agents or []:
            self.add(agent)

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def add(self, agent: Agent) -> None:
        """Add an agent to the mesh."""
        if agent.name in self._agents:
            raise MeshError(f"Agent '{agent.name}' already exists in mesh")
        agent._tracer = self.tracer  # noqa: SLF001
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise MeshError(f"Agent '{name}' not found in mesh")
        return self._agents[name]

    def connect(
        self,
        source: Agent | str,
        target: Agent | str,
        gate: Gate | None = None,
    ) -> None:
        """Connect two agents with an optional gate on the edge.

        Signals emitted by `source` will be delivered to `target`'s inbox,
        after passing through the optional gate.
        """
        src = self._resolve(source)
        tgt = self._resolve(target)

        edge = Edge(src, tgt, gate)
        self._edges.append(edge)
        tracer = self.tracer
        mesh = self

        async def route(signal: Signal) -> None:
            # Apply mesh-level interceptors
            intercepted = await mesh._apply_interceptors(signal, src.name, tgt.name)
            if intercepted is None:
                tracer.record(
                    trace_id=signal.trace_id,
                    signal_id=signal.id,
                    agent=f"{src.name}->{tgt.name}",
                    gate="interceptor",
                    action="intercepted",
                )
                return
            signal = intercepted

            if gate is not None:
                result = await gate.process(signal)
                if result is None:
                    tracer.record(
                        trace_id=signal.trace_id,
                        signal_id=signal.id,
                        agent=f"{src.name}->{tgt.name}",
                        gate=gate.name,
                        action="edge_rejected",
                    )
                    return
                signal = result
            tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent=src.name,
                gate="route",
                action="routed",
                target=tgt.name,
            )
            await tgt.inbox.send(signal)

        src._add_output(route)

    def broadcast_connect(
        self,
        source: Agent | str,
        targets: list[Agent | str],
        gate: Gate | None = None,
    ) -> None:
        """Connect one source to multiple targets (fan-out)."""
        for target in targets:
            self.connect(source, target, gate)

    def converge_connect(
        self,
        sources: list[Agent | str],
        target: Agent | str,
        gate: Gate | None = None,
    ) -> None:
        """Connect multiple sources to one target (fan-in)."""
        for source in sources:
            self.connect(source, target, gate)

    def route(
        self,
        source: Agent | str,
        routes: list[tuple[Callable[[Signal], bool], Agent | str]],
        default: Agent | str | None = None,
    ) -> None:
        """Content-based routing: route signals to different targets based on predicates.

        First matching predicate wins. If no predicate matches and a default is
        specified, the signal goes there. Otherwise it is dropped.

        This is the agent-native way to build intelligent signal routing —
        signals go where they need to based on their content, not just topology.

            mesh.route(coordinator, [
                (lambda s: s.priority >= 8, critical_handler),
                (lambda s: isinstance(s, AnalysisTask), analyst),
            ], default=general_worker)
        """
        src = self._resolve(source)
        resolved: list[tuple[Callable[[Signal], bool], Agent]] = [
            (pred, self._resolve(tgt)) for pred, tgt in routes
        ]
        resolved_default = self._resolve(default) if default else None
        tracer = self.tracer

        async def content_route(signal: Signal) -> None:
            for predicate, target in resolved:
                if predicate(signal):
                    tracer.record(
                        trace_id=signal.trace_id,
                        signal_id=signal.id,
                        agent=src.name,
                        gate="content_route",
                        action="routed",
                        target=target.name,
                    )
                    await target.inbox.send(signal)
                    return
            if resolved_default:
                tracer.record(
                    trace_id=signal.trace_id,
                    signal_id=signal.id,
                    agent=src.name,
                    gate="content_route",
                    action="default_routed",
                    target=resolved_default.name,
                )
                await resolved_default.inbox.send(signal)

        src._add_output(content_route)

    def load_balance(
        self,
        source: Agent | str,
        targets: list[Agent | str],
        gate: Gate | None = None,
    ) -> None:
        """Round-robin load balancing across multiple target agents.

        Distributes signals evenly across targets. Essential for scaling
        agent workloads horizontally:

            mesh.load_balance(dispatcher, [worker1, worker2, worker3])
        """
        src = self._resolve(source)
        resolved = [self._resolve(t) for t in targets]
        if not resolved:
            raise MeshError("load_balance requires at least one target")
        counter = [0]
        tracer = self.tracer

        async def balanced_route(signal: Signal) -> None:
            if gate is not None:
                result = await gate.process(signal)
                if result is None:
                    return
                signal = result
            target = resolved[counter[0] % len(resolved)]
            counter[0] += 1
            tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent=src.name,
                gate="load_balance",
                action="routed",
                target=target.name,
            )
            await target.inbox.send(signal)

        src._add_output(balanced_route)

    async def start(self) -> None:
        """Start all agents in the mesh."""
        self._running = True
        for agent in self._agents.values():
            await agent.start()
        # Yield control so agent tasks can start running
        await asyncio.sleep(0)
        logger.info(
            "Mesh started with %d agents, %d edges",
            len(self._agents),
            len(self._edges),
        )

    async def inject(self, target: Agent | str, signal: Signal) -> None:
        """Inject a signal directly into an agent's inbox."""
        agent = self._resolve(target)
        await agent.inbox.send(signal)

    def health(self) -> dict[str, Any]:
        """Aggregate health status of all agents in the mesh."""
        agent_health = {a.name: a.health() for a in self._agents.values()}
        all_healthy = all(h["healthy"] for h in agent_health.values())
        return {
            "healthy": all_healthy,
            "running": self._running,
            "agents": agent_health,
            "total_agents": len(self._agents),
            "total_edges": len(self._edges),
        }

    def topology(self) -> dict[str, Any]:
        """Return the mesh topology as a dict for inspection."""
        return {
            "agents": [
                {
                    "name": a.name,
                    "gates": len(a.gates),
                    "handlers": sum(len(h) for h in a._handlers.values()),
                }
                for a in self._agents.values()
            ],
            "edges": [
                {
                    "source": e.source.name,
                    "target": e.target.name,
                    "gate": e.gate.name if e.gate else None,
                }
                for e in self._edges
            ],
        }

    # --- Interceptors ---

    def intercept(self, fn: Interceptor) -> None:
        """Add a mesh-level interceptor for all signal routing.

        Interceptors see every signal that flows through mesh edges.
        They receive (signal, source_name, target_name) and return:
        - The signal (possibly modified) to allow routing
        - None to block the signal

        Use for cross-cutting concerns: auth, logging, metrics, rate limiting.

            def log_all(signal, source, target):
                print(f"{source} -> {target}: {signal}")
                return signal

            mesh.intercept(log_all)
        """
        self._interceptors.append(fn)

    async def _apply_interceptors(
        self, signal: Signal, source: str, target: str
    ) -> Signal | None:
        """Apply all interceptors to a signal. Returns None if any blocks it."""
        current: Signal | None = signal
        for interceptor in self._interceptors:
            if current is None:
                return None
            result = interceptor(current, source, target)
            if isawaitable(result):
                current = await result
            else:
                current = result
        return current

    # --- Capability Discovery ---

    def declare_capabilities(self, agent: Agent | str, *capabilities: str) -> None:
        """Declare what an agent can do. Enables discovery by capability.

            mesh.declare_capabilities(analyst, "analysis", "summarization")
            mesh.declare_capabilities(coder, "code_generation", "debugging")

            # Later, find agents by what they can do
            analysts = mesh.find_capable("analysis")
        """
        name = agent if isinstance(agent, str) else agent.name
        if name not in self._agents:
            raise MeshError(f"Agent '{name}' not in mesh")
        self._capabilities.setdefault(name, set()).update(capabilities)

    def find_capable(self, capability: str) -> list[Agent]:
        """Find all agents that have declared a given capability."""
        return [
            self._agents[name]
            for name, caps in self._capabilities.items()
            if capability in caps
        ]

    def agent_capabilities(self, agent: Agent | str) -> set[str]:
        """Get all declared capabilities for an agent."""
        name = agent if isinstance(agent, str) else agent.name
        return set(self._capabilities.get(name, set()))

    # --- Scatter / Gather ---

    async def scatter(
        self,
        signal: Signal,
        targets: list[Agent | str],
        timeout: float = 30.0,
    ) -> list[Signal]:
        """Scatter a signal to multiple agents and gather all correlated responses.

        This is THE fundamental multi-agent coordination pattern:
        send work to N agents in parallel, wait for all to respond.

            responses = await mesh.scatter(
                TaskSignal(task="analyze"),
                [analyst1, analyst2, analyst3],
                timeout=10.0,
            )

        Each target agent must `reply()` to the signal for gather to complete.
        Returns responses in the same order as targets.
        """
        cid = uuid4().hex
        tagged = signal.evolve(correlation_id=cid)
        resolved = [self._resolve(t) for t in targets]

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[Signal]] = []

        # Create a temporary collector agent to gather responses
        for target in resolved:
            future: asyncio.Future[Signal] = loop.create_future()
            futures.append(future)
            target_cid = f"{cid}:{target.name}"
            response_signal = tagged.evolve(correlation_id=target_cid)
            await target.inbox.send(response_signal)

        # Wait for all responses or timeout — agents reply via correlation_id
        # We register pending requests on each target
        per_target_futures: list[asyncio.Future[Signal]] = []
        for target in resolved:
            target_future: asyncio.Future[Signal] = loop.create_future()
            target_cid = f"{cid}:{target.name}"
            target._pending_requests[target_cid] = target_future  # noqa: SLF001
            per_target_futures.append(target_future)
            await target.inbox.send(tagged.evolve(correlation_id=target_cid))

        try:
            done, _ = await asyncio.wait(per_target_futures, timeout=timeout)
            results: list[Signal] = []
            for f in per_target_futures:
                if f.done() and not f.cancelled() and f.exception() is None:
                    results.append(f.result())
                else:
                    results.append(tagged)  # Placeholder for timed-out targets
            return results
        finally:
            for target in resolved:
                target_cid = f"{cid}:{target.name}"
                target._pending_requests.pop(target_cid, None)  # noqa: SLF001

    # --- Graceful Shutdown ---

    async def stop(self, drain: bool = False) -> None:
        """Stop all agents gracefully.

        Args:
            drain: If True, wait for agents to process all pending signals
                   before shutting down. If False, stop immediately.
        """
        self._running = False
        if drain:
            # Wait for all inboxes to empty
            for _ in range(100):  # Max 10 seconds
                all_empty = all(a.inbox.pending == 0 for a in self._agents.values())
                if all_empty:
                    break
                await asyncio.sleep(0.1)
        await asyncio.gather(*(agent.stop() for agent in self._agents.values()))
        logger.info("Mesh stopped")

    def _resolve(self, agent: Agent | str) -> Agent:
        if isinstance(agent, str):
            return self.get(agent)
        if agent.name not in self._agents:
            raise MeshError(f"Agent '{agent.name}' not in mesh — add it first")
        return agent

    async def __aenter__(self) -> Mesh:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    def __repr__(self) -> str:
        return f"Mesh(agents={len(self._agents)}, edges={len(self._edges)})"
