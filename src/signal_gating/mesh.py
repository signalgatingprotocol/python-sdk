"""Mesh — the agent network topology that ties everything together."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from signal_gating.agent import Agent
from signal_gating.errors import MeshError
from signal_gating.gate import Gate
from signal_gating.signal import Signal
from signal_gating.tracing import Tracer

if TYPE_CHECKING:
    from signal_gating.pool import AgentPool

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
        self._pools: dict[str, AgentPool] = {}
        self._topics: dict[str, list[Agent]] = {}
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
        agent.set_tracer(self.tracer)
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise MeshError(f"Agent '{name}' not found in mesh")
        return self._agents[name]

    async def remove(self, agent: Agent | str) -> None:
        """Remove an agent from the mesh, stopping it and cleaning up edges.

        Dynamic topology is essential for agent systems — agents join and leave
        at runtime based on demand, failures, or scaling decisions.

        This removes all edges, routing functions, and capabilities — the agent
        is fully severed from the mesh.
        """
        resolved = self._resolve(agent)
        if resolved.running:
            await resolved.stop()
        # Remove all outbox functions that route TO this agent (from other agents)
        for other in self._agents.values():
            if other.name != resolved.name:
                other._remove_outputs(target=resolved.name)
        # Clear the removed agent's own outbox
        resolved._outbox.clear()
        # Remove all edges involving this agent
        self._edges = [
            e for e in self._edges
            if e.source.name != resolved.name and e.target.name != resolved.name
        ]
        # Remove capabilities
        self._capabilities.pop(resolved.name, None)
        del self._agents[resolved.name]

    def disconnect(self, source: Agent | str, target: Agent | str) -> int:
        """Remove all edges between source and target. Returns count removed.

        Enables runtime topology rewiring — agents can be reconnected
        to different targets without restarting the mesh.

        This removes both the edge records AND the actual routing functions,
        so signals genuinely stop flowing between the disconnected agents.
        """
        src = self._resolve(source)
        tgt_name = target if isinstance(target, str) else target.name
        before = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (e.source.name == src.name and e.target.name == tgt_name)
        ]
        # Remove the actual routing functions from the source agent's outbox
        src._remove_outputs(target=tgt_name)
        return before - len(self._edges)

    def pipe(
        self,
        *agents: Agent | str,
        gate: Gate | None = None,
    ) -> None:
        """Connect agents in a linear chain — the most common topology pattern.

        Instead of writing N separate connect() calls:

            mesh.connect(fetcher, parser)
            mesh.connect(parser, validator)
            mesh.connect(validator, storer)

        Write one pipe():

            mesh.pipe(fetcher, parser, validator, storer)

        An optional gate is applied to every edge in the chain.
        All agents are auto-added to the mesh if not already present.

            mesh.pipe(fetcher, parser, validator, storer, gate=Gate.by_priority(3))
        """
        if len(agents) < 2:
            raise MeshError("pipe() requires at least 2 agents")
        resolved: list[Agent] = []
        for a in agents:
            agent = a if isinstance(a, Agent) else self._agents.get(a)
            if agent is None:
                raise MeshError(f"Agent '{a}' not found in mesh — add it first")
            if agent.name not in self._agents:
                self.add(agent)
            resolved.append(agent)
        for i in range(len(resolved) - 1):
            self.connect(resolved[i], resolved[i + 1], gate)

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

        content_route._mesh_tag = "content_route"  # type: ignore[attr-defined]
        content_route._mesh_source = src.name  # type: ignore[attr-defined]
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
        index = itertools.count()
        tracer = self.tracer

        async def balanced_route(signal: Signal) -> None:
            if gate is not None:
                result = await gate.process(signal)
                if result is None:
                    return
                signal = result
            target = resolved[next(index) % len(resolved)]
            tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent=src.name,
                gate="load_balance",
                action="routed",
                target=target.name,
            )
            await target.inbox.send(signal)

        balanced_route._mesh_tag = "load_balance"  # type: ignore[attr-defined]
        balanced_route._mesh_source = src.name  # type: ignore[attr-defined]
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

    # --- Agent Pools ---

    def add_pool(self, pool: AgentPool) -> None:
        """Add an agent pool to the mesh.

        All pool workers are added as mesh agents. The pool object is tracked
        so that ``connect(source, pool)`` and ``connect(pool, target)`` work
        with automatic load balancing.

            pool = AgentPool("workers", size=3)
            mesh.add_pool(pool)
            mesh.connect(coordinator, pool)  # load-balanced to workers
            mesh.connect(pool, collector)    # all workers emit to collector
        """
        if pool.name in self._pools:
            raise MeshError(f"Pool '{pool.name}' already exists in mesh")
        self._pools[pool.name] = pool
        for worker in pool.workers:
            self.add(worker)

    def get_pool(self, name: str) -> AgentPool:
        """Get a pool by name."""
        if name not in self._pools:
            raise MeshError(f"Pool '{name}' not found in mesh")
        return self._pools[name]

    def _resolve_pool_or_agent(
        self, ref: Agent | str | AgentPool,
    ) -> Agent | AgentPool:
        """Resolve a reference that could be an Agent, pool name, or AgentPool."""
        from signal_gating.pool import AgentPool

        if isinstance(ref, AgentPool):
            return ref
        if isinstance(ref, str):
            if ref in self._pools:
                return self._pools[ref]
            return self.get(ref)
        return ref

    def connect(
        self,
        source: Agent | str | AgentPool,
        target: Agent | str | AgentPool,
        gate: Gate | None = None,
    ) -> None:
        """Connect two agents with an optional gate on the edge.

        Supports AgentPool as source or target:
        - Pool as target: load-balanced across pool workers
        - Pool as source: all workers connect to target
        """
        from signal_gating.pool import AgentPool

        resolved_src = self._resolve_pool_or_agent(source)
        resolved_tgt = self._resolve_pool_or_agent(target)

        if isinstance(resolved_src, AgentPool) and isinstance(resolved_tgt, AgentPool):
            # Pool-to-pool: all source workers load-balance to target workers
            for worker in resolved_src.workers:
                self.load_balance(worker, resolved_tgt.workers, gate)
            return

        if isinstance(resolved_tgt, AgentPool):
            # Load-balance across pool workers
            src_agent = resolved_src if isinstance(resolved_src, Agent) else self.get(str(source))
            self.load_balance(src_agent, resolved_tgt.workers, gate)
            return

        if isinstance(resolved_src, AgentPool):
            # All pool workers connect to target
            tgt_agent = resolved_tgt if isinstance(resolved_tgt, Agent) else self.get(str(target))
            for worker in resolved_src.workers:
                self._connect_agents(worker, tgt_agent, gate)
            return

        # Standard agent-to-agent
        src_agent = resolved_src if isinstance(resolved_src, Agent) else self._resolve(source)
        tgt_agent = resolved_tgt if isinstance(resolved_tgt, Agent) else self._resolve(target)
        self._connect_agents(src_agent, tgt_agent, gate)

    def _connect_agents(
        self,
        src: Agent,
        tgt: Agent,
        gate: Gate | None = None,
    ) -> None:
        """Internal: wire two agents together."""
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

        # Tag route function so disconnect/remove can find and remove it
        route._mesh_target = tgt.name  # type: ignore[attr-defined]
        route._mesh_source = src.name  # type: ignore[attr-defined]
        route._mesh_tag = "connect"  # type: ignore[attr-defined]
        src._add_output(route)

    # --- Pub/Sub Topics ---

    def create_topic(self, name: str) -> None:
        """Create a named topic for publish/subscribe communication.

        Topics decouple producers from consumers. Any agent can publish to a
        topic, and all subscribers receive the signal. This is the agent-native
        broadcast pattern — essential when the number and identity of consumers
        is dynamic.

            mesh.create_topic("events")
            mesh.subscribe(logger_agent, "events")
            mesh.subscribe(metrics_agent, "events")

            # Any agent can publish
            await mesh.publish("events", AlertSignal(message="CPU spike"))
        """
        if name in self._topics:
            raise MeshError(f"Topic '{name}' already exists")
        self._topics[name] = []

    def subscribe(self, agent: Agent | str, topic: str) -> None:
        """Subscribe an agent to a topic. It will receive all published signals."""
        if topic not in self._topics:
            raise MeshError(f"Topic '{topic}' does not exist — create it first")
        resolved = self._resolve(agent)
        if resolved not in self._topics[topic]:
            self._topics[topic].append(resolved)

    def unsubscribe(self, agent: Agent | str, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        if topic not in self._topics:
            return
        resolved = self._resolve(agent)
        self._topics[topic] = [a for a in self._topics[topic] if a.name != resolved.name]

    async def publish(self, topic: str, signal: Signal) -> int:
        """Publish a signal to all subscribers of a topic.

        Returns the number of subscribers that received the signal.
        This is fire-and-forget — the signal is sent to each subscriber's
        inbox without waiting for processing.

            count = await mesh.publish("events", AlertSignal(message="alert"))
        """
        if topic not in self._topics:
            raise MeshError(f"Topic '{topic}' does not exist")
        subscribers = self._topics[topic]
        for subscriber in subscribers:
            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent="mesh",
                gate=f"topic:{topic}",
                action="published",
                target=subscriber.name,
            )
            await subscriber.inbox.send(signal)
        return len(subscribers)

    def list_topics(self) -> dict[str, list[str]]:
        """List all topics and their subscriber names."""
        return {
            topic: [a.name for a in subscribers]
            for topic, subscribers in self._topics.items()
        }

    def delete_topic(self, name: str) -> None:
        """Delete a topic and remove all subscriptions."""
        if name not in self._topics:
            raise MeshError(f"Topic '{name}' does not exist")
        del self._topics[name]

    # --- Request / Response ---

    async def request(
        self,
        target: Agent | str,
        signal: Signal,
        timeout: float = 30.0,
    ) -> Signal:
        """Send a signal to a specific agent and wait for a correlated response.

        This is the workflow building block — the agent-native RPC pattern.
        Unlike Agent.request() which emits through the outbox (to all connected
        agents), Mesh.request() injects directly into a specific agent's inbox
        and captures the response.

        The target agent must call ``reply()`` or emit a signal with the
        matching correlation_id for the request to resolve.

            response = await mesh.request(analyst, TaskSignal(task="analyze"))
            print(response.metadata)

        Chain requests to build multi-step workflows:

            r1 = await mesh.request(fetcher, FetchSignal(url="..."))
            r2 = await mesh.request(parser, ParseSignal(data=r1))
            r3 = await mesh.request(storer, StoreSignal(parsed=r2))
        """
        resolved = self._resolve(target)
        cid = uuid4().hex
        tagged = signal.evolve(correlation_id=cid)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Signal] = loop.create_future()

        # Capture the reply from the target's outbox
        async def capture(sig: Signal) -> None:
            if sig.correlation_id == cid and not future.done():
                future.set_result(sig)

        resolved._outbox.append(capture)
        try:
            await resolved.inbox.send(tagged)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            try:
                resolved._outbox.remove(capture)
            except ValueError:
                pass

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
        resolved = [self._resolve(t) for t in targets]

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[Signal]] = []
        capture_fns: list[tuple[Agent, Any]] = []

        for target in resolved:
            target_cid = f"{cid}:{target.name}"
            future: asyncio.Future[Signal] = loop.create_future()
            futures.append(future)

            # Capture replies from this target's outbox by correlation_id.
            # This avoids the previous bug where registering on _pending_requests
            # caused the signal to be intercepted before reaching handlers.
            def _make_capture(
                f: asyncio.Future[Signal], tcid: str
            ) -> Any:
                async def capture(sig: Signal) -> None:
                    if sig.correlation_id == tcid and not f.done():
                        f.set_result(sig)

                return capture

            capture = _make_capture(future, target_cid)
            target._outbox.append(capture)  # noqa: SLF001
            capture_fns.append((target, capture))

        # Send signals to all targets
        for target in resolved:
            target_cid = f"{cid}:{target.name}"
            await target.inbox.send(
                signal.evolve(correlation_id=target_cid)
            )

        try:
            done, _ = await asyncio.wait(futures, timeout=timeout)
            results: list[Signal] = []
            for f in futures:
                if f.done() and not f.cancelled() and f.exception() is None:
                    results.append(f.result())
                else:
                    results.append(signal)  # Placeholder for timed-out targets
            return results
        finally:
            for target, capture in capture_fns:
                try:
                    target._outbox.remove(capture)  # noqa: SLF001
                except ValueError:
                    pass

    # --- Workflow Orchestration ---

    async def workflow(
        self,
        signal: Signal,
        steps: list[Agent | str],
        timeout: float = 60.0,
        step_timeout: float = 30.0,
    ) -> Signal:
        """Execute a sequential multi-agent workflow — THE agent orchestration primitive.

        Sends a signal through a chain of agents where each agent's response
        becomes the next agent's input. This is the fundamental pattern for
        building complex agent pipelines: fetch → parse → analyze → store.

        Each step uses the mesh request/response pattern, so target agents
        must call ``reply()`` or emit a correlated response.

        Args:
            signal: The initial signal to send to the first step.
            steps: Ordered list of agents to process the signal through.
            timeout: Maximum total time for the entire workflow.
            step_timeout: Maximum time for each individual step.

        Returns:
            The final response signal from the last agent in the chain.

        Example::

            result = await mesh.workflow(
                TaskSignal(task="analyze quarterly revenue"),
                steps=[data_fetcher, analyzer, summarizer, formatter],
                timeout=60.0,
                step_timeout=15.0,
            )
            # result is the formatter's response, carrying the full
            # lineage of transformations through all four agents

        Chain workflows for complex orchestration::

            # Parallel preparation, then sequential processing
            prep = await mesh.scatter(SetupSignal(), [db_agent, cache_agent])
            result = await mesh.workflow(
                AnalyzeSignal(context=prep),
                steps=[analyzer, reviewer, publisher],
            )
        """
        if not steps:
            raise MeshError("workflow requires at least one step")

        async def _run() -> Signal:
            current = signal
            for i, step in enumerate(steps):
                resolved = self._resolve(step)
                self.tracer.record(
                    trace_id=signal.trace_id,
                    signal_id=current.id,
                    agent="mesh",
                    gate="workflow",
                    action="step_start",
                    step=i,
                    target=resolved.name,
                )
                current = await self.request(resolved, current, timeout=step_timeout)
                self.tracer.record(
                    trace_id=signal.trace_id,
                    signal_id=current.id,
                    agent="mesh",
                    gate="workflow",
                    action="step_complete",
                    step=i,
                    target=resolved.name,
                )
            return current

        return await asyncio.wait_for(_run(), timeout=timeout)

    async def race(
        self,
        signal: Signal,
        targets: list[Agent | str],
        timeout: float = 30.0,
    ) -> Signal:
        """Race a signal across multiple agents — first response wins.

        Sends the same signal to N agents in parallel and returns the
        first response received. All other pending responses are discarded.
        This is THE competitive execution primitive for agent systems.

        Use cases:
        - Try multiple strategies in parallel, take the fastest
        - Query multiple data sources, return first available
        - Run fast/cheap and slow/thorough agents, take whichever finishes first
        - Speculative execution with automatic cancellation

        Args:
            signal: The signal to send to all targets.
            targets: Agents that will race to respond.
            timeout: Maximum time to wait for the first response.

        Returns:
            The first response signal received from any target.

        Example::

            result = await mesh.race(
                AnalyzeSignal(data=data),
                [cache_lookup, fast_analyzer, deep_analyzer],
                timeout=5.0,
            )
            # result comes from whichever agent responds first
        """
        if not targets:
            raise MeshError("race requires at least one target")

        cid = uuid4().hex
        resolved = [self._resolve(t) for t in targets]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Signal] = loop.create_future()
        capture_fns: list[tuple[Agent, Any]] = []

        for target in resolved:
            target_cid = f"{cid}:{target.name}"

            def _make_capture(
                f: asyncio.Future[Signal], tcid: str,
            ) -> Any:
                async def capture(sig: Signal) -> None:
                    if sig.correlation_id == tcid and not f.done():
                        f.set_result(sig)

                return capture

            capture = _make_capture(future, target_cid)
            target._outbox.append(capture)
            capture_fns.append((target, capture))

        # Send to all targets in parallel
        for target in resolved:
            target_cid = f"{cid}:{target.name}"
            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent="mesh",
                gate="race",
                action="sent",
                target=target.name,
            )
            await target.inbox.send(signal.evolve(correlation_id=target_cid))

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=result.id,
                agent="mesh",
                gate="race",
                action="winner",
            )
            return result
        finally:
            for target, capture in capture_fns:
                try:
                    target._outbox.remove(capture)
                except ValueError:
                    pass

    # --- Graceful Shutdown ---

    async def stop(self, drain: bool = False, drain_timeout: float = 10.0) -> None:
        """Stop all agents gracefully.

        Args:
            drain: If True, wait for agents to process all pending signals
                   before shutting down. If False, stop immediately.
            drain_timeout: Maximum seconds to wait for drain (default 10s).
        """
        self._running = False
        if drain:
            # Wait for all inboxes to empty with adaptive polling
            deadline = asyncio.get_running_loop().time() + drain_timeout
            interval = 0.01  # Start fast, back off
            while asyncio.get_running_loop().time() < deadline:
                if all(a.inbox.pending == 0 for a in self._agents.values()):
                    break
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, 0.2)  # Adaptive: 10ms -> 200ms
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

    def visualize(self) -> str:
        """Text-based topology visualization for introspection and debugging.

        Returns a human-readable string showing all agents, their connections,
        topics, and capabilities. Agents can use this to understand their own
        network topology at runtime.

            print(mesh.visualize())
            # Mesh (3 agents, 2 edges)
            # ├── fetcher [2 handlers]
            # │   └──> parser (via priority_filter)
            # ├── parser [1 handler]
            # │   └──> storer
            # └── storer [1 handler]
        """
        lines: list[str] = []
        agents = list(self._agents.values())
        lines.append(f"Mesh ({len(agents)} agents, {len(self._edges)} edges)")

        for i, agent in enumerate(agents):
            is_last_agent = i == len(agents) - 1
            prefix = "└── " if is_last_agent else "├── "
            child_prefix = "    " if is_last_agent else "│   "

            n_handlers = sum(len(h) for h in agent._handlers.values())
            status = "running" if agent.running else "stopped"
            line = f"{prefix}{agent.name} [{n_handlers} handlers, {status}]"

            caps = self._capabilities.get(agent.name, set())
            if caps:
                line += f" caps={{{', '.join(sorted(caps))}}}"
            lines.append(line)

            # Show outgoing edges
            outgoing = [e for e in self._edges if e.source.name == agent.name]
            for j, edge in enumerate(outgoing):
                is_last_edge = j == len(outgoing) - 1
                edge_prefix = child_prefix + ("└──> " if is_last_edge else "├──> ")
                gate_info = f" (via {edge.gate.name})" if edge.gate else ""
                lines.append(f"{edge_prefix}{edge.target.name}{gate_info}")

        # Show topics
        if self._topics:
            lines.append("")
            lines.append("Topics:")
            for topic, subscribers in self._topics.items():
                sub_names = [a.name for a in subscribers]
                lines.append(f"  {topic}: [{', '.join(sub_names)}]")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Mesh(agents={len(self._agents)}, edges={len(self._edges)})"
