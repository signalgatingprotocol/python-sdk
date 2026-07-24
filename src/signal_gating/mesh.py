"""Mesh: the agent network topology that ties everything together."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from signal_gating.agent import Agent, ToolCallSignal, ToolResultSignal, ToolSpec
from signal_gating.errors import AgentError, ChannelClosed, MeshError
from signal_gating.gate import Gate
from signal_gating.signal import Signal
from signal_gating.tracing import Tracer

if TYPE_CHECKING:
    from signal_gating.pool import AgentPool

logger = logging.getLogger("signal_gating.mesh")

Interceptor = Callable[[Signal, str, str], Signal | None | Awaitable[Signal | None]]
MeshEventSink = Callable[["MeshEvent"], None | Awaitable[None]]
_DeliveryFn = Callable[[Signal], None | Awaitable[None]]


class Edge:
    """A directional connection between two agents, optionally gated."""

    def __init__(self, source: Agent, target: Agent, gate: Gate | None = None):
        self.source = source
        self.target = target
        self.gate = gate


@dataclass
class MeshEvent:
    """A structured mesh execution event for receipts, replay, and audit sinks."""

    action: str
    signal: Signal
    source: str
    target: str = ""
    event_kind: str = "mesh"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _DeliveryOutcome:
    """Internal result of one mesh-mediated delivery attempt."""

    signal: Signal
    delivered: bool
    blocked_stage: str = ""
    blocked_name: str = ""


@dataclass(frozen=True)
class _PoolConnection:
    """A logical connection whose source or target is an agent pool."""

    source: Agent | AgentPool
    target: Agent | AgentPool
    gate: Gate | None = None


class _PublishDestinationClosedError(Exception):
    """Internal marker for a publish destination closing during inbox send."""


@dataclass
class RouteFn:
    """A mesh output route, carrying the send function plus routing metadata.

    Metadata lets the mesh find and tear down routes by source, target, or
    tag (e.g. disconnecting a specific edge or removing all load-balanced
    routes from an agent).
    """

    fn: Callable[[Signal], Coroutine[Any, Any, None]]
    source: str
    target: str | None = None
    tag: str = ""
    # Called with a removed agent's name; mutates captured route state and
    # returns True when the route has no remaining targets and must be dropped.
    prune: Callable[[str], bool] | None = None

    async def __call__(self, signal: Signal) -> None:
        await self.fn(signal)


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
        self._event_sinks: list[MeshEventSink] = []
        self._event_sink_errors = 0
        self._capabilities: dict[str, set[str]] = {}
        self._pools: dict[str, AgentPool] = {}
        self._pool_connections: list[_PoolConnection] = []
        self._pool_scale_lock = asyncio.Lock()
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
        if agent.name in self._pools:
            raise MeshError(
                f"Agent name '{agent.name}' conflicts with an existing pool"
            )
        agent.set_tracer(self.tracer)
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise MeshError(f"Agent '{name}' not found in mesh")
        return self._agents[name]

    async def remove(self, agent: Agent | str) -> None:
        """Remove an agent from the mesh, stopping it and cleaning up edges.

        This removes all edges, routing functions, pool memberships, and
        capabilities; the agent is fully severed from the mesh. Routes built
        by ``route()``/``load_balance()`` are pruned of the removed agent and
        dropped entirely once they have no remaining targets.

        In-flight request hazard: clearing the removed agent's ``_outbox``
        drops any live ``mesh.request()`` capture functions, so a requester
        awaiting a reply from that agent times out rather than erroring.
        """
        resolved = self._resolve(agent)
        for pool in self._pools.values():
            if resolved.name in pool.worker_names:
                descriptor = "final worker" if pool.size == 1 else "worker"
                raise MeshError(
                    f"Cannot remove {descriptor} '{resolved.name}' directly "
                    f"from attached pool '{pool.name}'; use await "
                    f"mesh.scale_pool('{pool.name}', size)"
                )
        if resolved.running:
            await resolved.stop()
        # Remove all outbox functions that route TO this agent (from other agents)
        for other in self._agents.values():
            if other.name != resolved.name:
                other._remove_outputs(target=resolved.name)
        # Prune target=None routing closures (route()/load_balance()) that
        # hold a direct reference to this agent; drop routes left empty.
        for other in self._agents.values():
            other._outbox = [
                fn
                for fn in other._outbox
                if not (
                    isinstance(fn, RouteFn)
                    and fn.prune is not None
                    and fn.prune(resolved.name)
                )
            ]
        # Purge pool membership so pools stop selecting the dead worker
        for pool in self._pools.values():
            pool._discard_worker(resolved.name)
        # Clear the removed agent's own outbox
        resolved._outbox.clear()
        # Remove all edges involving this agent
        self._edges = [
            e for e in self._edges
            if e.source.name != resolved.name and e.target.name != resolved.name
        ]
        # Remove capabilities
        self._capabilities.pop(resolved.name, None)
        # Remove topic subscriptions so publish() never targets the dead
        # agent's closed inbox (and the agent object can be collected).
        for topic, subscribers in self._topics.items():
            self._topics[topic] = [a for a in subscribers if a.name != resolved.name]
        # Drop logical pool policies with this individual agent as an endpoint.
        # Policies whose endpoint is the owning pool remain valid for its
        # surviving and future workers.
        self._pool_connections = [
            connection
            for connection in self._pool_connections
            if connection.source is not resolved and connection.target is not resolved
        ]
        del self._agents[resolved.name]

    def disconnect(self, source: Agent | str, target: Agent | str) -> int:
        """Remove all edges between source and target. Returns count removed.

        Enables runtime topology rewiring. Agents can be reconnected
        to different targets without restarting the mesh.

        This removes both the edge records AND the actual routing functions,
        so signals genuinely stop flowing between the disconnected agents.
        """
        src = self._resolve(source)
        tgt_name = self._resolve(target).name
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
        """Connect agents in a linear chain, the most common topology pattern.

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
                raise MeshError(f"Agent '{a}' not found in mesh; add it first")
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

        Signals are routed based on their content, not just topology.

            mesh.route(coordinator, [
                (lambda s: s.priority >= 8, critical_handler),
                (lambda s: isinstance(s, AnalysisTask), analyst),
            ], default=general_worker)
        """
        src = self._resolve(source)
        resolved: list[tuple[Callable[[Signal], bool], Agent]] = [
            (pred, self._resolve(tgt)) for pred, tgt in routes
        ]
        default_box: list[Agent | None] = [self._resolve(default) if default else None]

        async def content_route(signal: Signal) -> None:
            for predicate, target in resolved:
                if predicate(signal):
                    await self._deliver(
                        signal,
                        source=src.name,
                        target=target.name,
                        send=target.inbox.send,
                        action="routed",
                        event_kind="signal",
                        trace_gate="content_route",
                        trace_action="routed",
                    )
                    return
            resolved_default = default_box[0]
            if resolved_default is not None:
                await self._deliver(
                    signal,
                    source=src.name,
                    target=resolved_default.name,
                    send=resolved_default.inbox.send,
                    action="routed",
                    event_kind="signal",
                    trace_gate="content_route",
                    trace_action="default_routed",
                )

        def prune(name: str) -> bool:
            resolved[:] = [(p, t) for p, t in resolved if t.name != name]
            if default_box[0] is not None and default_box[0].name == name:
                default_box[0] = None
            return not resolved and default_box[0] is None

        src._add_output(
            RouteFn(fn=content_route, source=src.name, tag="content_route", prune=prune)
        )

    def load_balance(
        self,
        source: Agent | str,
        targets: Sequence[Agent | str],
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
        async def balanced_route(signal: Signal) -> None:
            target = resolved[next(index) % len(resolved)]
            await self._deliver(
                signal,
                source=src.name,
                target=target.name,
                send=target.inbox.send,
                action="routed",
                event_kind="signal",
                gate=gate,
                trace_gate="load_balance",
                trace_action="routed",
            )

        def prune(name: str) -> bool:
            resolved[:] = [t for t in resolved if t.name != name]
            return not resolved

        src._add_output(
            RouteFn(fn=balanced_route, source=src.name, tag="load_balance", prune=prune)
        )

    async def start(self) -> None:
        """Start all agents in the mesh."""
        self._running = True
        try:
            for agent in self._agents.values():
                await agent.start()
        except BaseException:
            # A failing on_start hook must not leave already-started agents
            # running with no handle to stop them (and __aexit__ won't fire
            # because __aenter__ raised). Tear down what we started.
            await self.stop()
            raise
        # Yield control so agent tasks can start running
        await asyncio.sleep(0)
        logger.info(
            "Mesh started with %d agents, %d edges",
            len(self._agents),
            len(self._edges),
        )

    async def inject(self, target: Agent | str, signal: Signal) -> None:
        """Inject a signal through the mesh trust boundary into an agent."""
        agent = self._resolve(target)
        outcome = await self._deliver(
            signal,
            source="mesh",
            target=agent.name,
            send=agent.inbox.send,
            action="inject",
            trace_gate="inject",
            trace_action="inject",
        )
        self._raise_if_blocked(outcome, source="mesh", target=agent.name, action="inject")

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

        Interceptors see every mesh-mediated delivery, including direct
        orchestration sends and their correlated responses. Raw writes to an
        agent's inbox are outside the mesh boundary and are not intercepted.
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

    def record(self, sink: MeshEventSink | Any) -> None:
        """Record structured mesh execution events into a sink.

        This is the stable hook for action-specific delivery events and
        orchestration control events from ``inject()``, ``request()``,
        ``scatter()``, ``race()``, ``publish()``, ``workflow()``, and
        ``call_tool()``. Generic interceptors mediate every mesh-owned delivery
        but do not observe non-delivery control events. If ``sink`` exposes
        ``record_event()``, that method is registered; otherwise the object
        itself is treated as the event sink.
        """
        event_sink = getattr(sink, "record_event", sink)
        self._event_sinks.append(event_sink)

    def record_events(self, sink: MeshEventSink | Any) -> None:
        """Alias for :meth:`record` with a more explicit name."""
        self.record(sink)

    @property
    def event_sink_errors(self) -> int:
        """Number of event sink failures suppressed by the mesh."""
        return self._event_sink_errors

    async def _record_event(
        self,
        action: str,
        signal: Signal,
        *,
        source: str,
        target: str = "",
        event_kind: str = "mesh",
        **metadata: Any,
    ) -> None:
        if not self._event_sinks:
            return
        event = MeshEvent(
            action=action,
            signal=signal,
            source=source,
            target=target,
            event_kind=event_kind,
            metadata=metadata,
        )
        for sink in tuple(self._event_sinks):
            try:
                result = sink(event)
                if isawaitable(result):
                    await result
            except Exception:
                self._event_sink_errors += 1
                logger.warning("Mesh event sink failed", exc_info=True)

    async def _deliver(
        self,
        signal: Signal,
        *,
        source: str,
        target: str,
        send: _DeliveryFn,
        action: str,
        event_kind: str = "mesh",
        gate: Gate | None = None,
        trace_gate: str = "delivery",
        trace_action: str | None = None,
        **metadata: Any,
    ) -> _DeliveryOutcome:
        """Apply the mesh trust boundary and perform exactly one delivery.

        Every mesh-mediated signal path uses this function, including replies
        captured by request/scatter/race. Successful events are emitted only
        after the destination accepts the signal.
        """
        current = signal
        for index, interceptor in enumerate(tuple(self._interceptors)):
            result = interceptor(current, source, target)
            if isawaitable(result):
                result = await result
            if result is None:
                name = getattr(interceptor, "__name__", type(interceptor).__name__)
                await self._record_event(
                    "intercepted",
                    current,
                    source=source,
                    target=target,
                    event_kind=event_kind,
                    delivery_action=action,
                    blocked_stage="interceptor",
                    blocked_name=name,
                    interceptor_index=index,
                    signal_type=type(current).__name__,
                    signal_id=current.id,
                    **metadata,
                )
                self.tracer.record(
                    trace_id=current.trace_id,
                    signal_id=current.id,
                    agent=f"{source}->{target}",
                    gate="interceptor",
                    action="intercepted",
                    delivery_action=action,
                    blocked_name=name,
                )
                return _DeliveryOutcome(
                    current,
                    delivered=False,
                    blocked_stage="interceptor",
                    blocked_name=name,
                )
            if not isinstance(result, Signal):
                name = getattr(interceptor, "__name__", type(interceptor).__name__)
                raise MeshError(
                    f"Interceptor {name!r} returned {type(result).__name__}; "
                    "interceptors must return Signal or None"
                )
            current = result

        if gate is not None:
            gated = await gate.process(current)
            if gated is None:
                await self._record_event(
                    "edge_rejected",
                    current,
                    source=source,
                    target=target,
                    event_kind=event_kind,
                    delivery_action=action,
                    blocked_stage="gate",
                    blocked_name=gate.name,
                    gate=gate.name,
                    signal_type=type(current).__name__,
                    signal_id=current.id,
                    **metadata,
                )
                self.tracer.record(
                    trace_id=current.trace_id,
                    signal_id=current.id,
                    agent=f"{source}->{target}",
                    gate=gate.name,
                    action="edge_rejected",
                    delivery_action=action,
                )
                return _DeliveryOutcome(
                    current,
                    delivered=False,
                    blocked_stage="gate",
                    blocked_name=gate.name,
                )
            current = gated

        sent = send(current)
        if isawaitable(sent):
            await sent

        self.tracer.record(
            trace_id=current.trace_id,
            signal_id=current.id,
            agent=source,
            gate=trace_gate,
            action=trace_action or action,
            target=target,
            **metadata,
        )
        await self._record_event(
            action,
            current,
            source=source,
            target=target,
            event_kind=event_kind,
            **metadata,
        )
        return _DeliveryOutcome(current, delivered=True)

    async def _deliver_replay(
        self,
        target: Agent | str,
        signal: Signal,
        *,
        original_action: str,
        original_source: str,
        original_signal_id: str,
    ) -> None:
        """Deliver one replayed signal through the mesh trust boundary."""
        resolved = self._resolve(target)
        outcome = await self._deliver(
            signal,
            source="mesh",
            target=resolved.name,
            send=resolved.inbox.send,
            action="replay_delivered",
            trace_gate="replay",
            original_action=original_action,
            original_source=original_source,
            original_signal_id=original_signal_id,
        )
        self._raise_if_blocked(
            outcome,
            source="mesh",
            target=resolved.name,
            action="replay_delivered",
        )
        return None

    @staticmethod
    def _delivery_error(
        outcome: _DeliveryOutcome,
        *,
        source: str,
        target: str,
        action: str,
    ) -> MeshError:
        return MeshError(
            f"Mesh delivery {action!r} blocked at {outcome.blocked_stage} "
            f"{outcome.blocked_name!r}: {source!r} -> {target!r}, "
            f"signal {type(outcome.signal).__name__} {outcome.signal.id!r}"
        )

    @classmethod
    def _raise_if_blocked(
        cls,
        outcome: _DeliveryOutcome,
        *,
        source: str,
        target: str,
        action: str,
    ) -> None:
        if not outcome.delivered:
            raise cls._delivery_error(
                outcome, source=source, target=target, action=action
            )

    # --- Capability Discovery ---

    def declare_capabilities(self, agent: Agent | str, *capabilities: str) -> None:
        """Declare what an agent can do. Enables discovery by capability.

            mesh.declare_capabilities(analyst, "analysis", "summarization")
            mesh.declare_capabilities(coder, "code_generation", "debugging")

            # Later, find agents by what they can do
            analysts = mesh.find_capable("analysis")
        """
        resolved = self._resolve(agent)
        self._capabilities.setdefault(resolved.name, set()).update(capabilities)

    def find_capable(self, capability: str) -> list[Agent]:
        """Find all agents that have declared a given capability."""
        return [
            self._agents[name]
            for name, caps in self._capabilities.items()
            if capability in caps
        ]

    def agent_capabilities(self, agent: Agent | str) -> set[str]:
        """Get all declared capabilities for an agent."""
        resolved = self._resolve(agent)
        return set(self._capabilities.get(resolved.name, set()))

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
        if pool.name in self._agents:
            raise MeshError(
                f"Pool name '{pool.name}' conflicts with an existing agent"
            )
        worker_names = [worker.name for worker in pool.workers]
        seen_worker_names: set[str] = set()
        duplicate_worker_names: list[str] = []
        for name in worker_names:
            if name in seen_worker_names and name not in duplicate_worker_names:
                duplicate_worker_names.append(name)
            seen_worker_names.add(name)
        if duplicate_worker_names:
            joined = ", ".join(repr(name) for name in duplicate_worker_names)
            raise MeshError(f"Pool '{pool.name}' has duplicate worker names: {joined}")
        agent_conflicts = [name for name in worker_names if name in self._agents]
        if agent_conflicts:
            joined = ", ".join(repr(name) for name in agent_conflicts)
            raise MeshError(f"Pool workers conflict with existing agents: {joined}")
        pool_namespace = {*self._pools, pool.name}
        pool_conflicts = [name for name in worker_names if name in pool_namespace]
        if pool_conflicts:
            joined = ", ".join(repr(name) for name in pool_conflicts)
            raise MeshError(f"Pool workers conflict with pool names: {joined}")
        pool._claim(self)
        for worker in pool.workers:
            self.add(worker)
        self._pools[pool.name] = pool

    def get_pool(self, name: str) -> AgentPool:
        """Get a pool by name."""
        if name not in self._pools:
            raise MeshError(f"Pool '{name}' not found in mesh")
        return self._pools[name]

    async def scale_pool(
        self,
        pool: AgentPool | str,
        size: int,
    ) -> list[Agent]:
        """Resize an attached pool while preserving mesh-owned topology.

        New workers are registered and wired before they become visible to
        pool routing. When the mesh is running they are also started before
        membership is committed. Removed workers leave pool selection before
        they are stopped and fully removed from the mesh.
        """
        if size < 1:
            raise ValueError("Pool size must be at least 1")

        resolved = self._resolve_pool_or_agent(pool)
        from signal_gating.pool import AgentPool

        if not isinstance(resolved, AgentPool):
            raise MeshError(f"Pool '{resolved.name}' not found in mesh")

        async with self._pool_scale_lock:
            current_size = resolved.size
            if size == current_size:
                return []

            if size > current_size:
                new_workers = resolved._prepare_workers(size - current_size)
                initial_connections = [
                    connection
                    for connection in self._pool_connections
                    if connection.source is resolved
                ]
                initial_connection_ids = {
                    id(connection) for connection in initial_connections
                }
                for worker in new_workers:
                    self.add(worker)
                    for connection in initial_connections:
                        self._wire_pool_source_worker(connection, worker)
                if self._running:
                    for worker in new_workers:
                        await worker.start()
                    # Agent.start() schedules its supervised loop; yield so
                    # newly added capacity is running before this call returns.
                    await asyncio.sleep(0)
                # A start hook or another task may connect the source pool
                # while worker startup yields. Apply those newly recorded
                # policies before committing staged workers to membership.
                for connection in self._pool_connections:
                    if (
                        connection.source is resolved
                        and id(connection) not in initial_connection_ids
                    ):
                        for worker in new_workers:
                            self._wire_pool_source_worker(connection, worker)
                resolved._commit_workers(new_workers)
                logger.info(
                    "Pool '%s' scaled up to %d workers (+%d)",
                    resolved.name,
                    resolved.size,
                    len(new_workers),
                )
                return new_workers

            removed = resolved._take_workers(current_size - size)
            for worker in removed:
                await self.remove(worker)
            logger.info(
                "Pool '%s' scaled down to %d workers (-%d)",
                resolved.name,
                resolved.size,
                len(removed),
            )
            return removed

    def _resolve_pool_or_agent(
        self, ref: Agent | str | AgentPool,
    ) -> Agent | AgentPool:
        """Resolve a reference that could be an Agent, pool name, or AgentPool."""
        from signal_gating.pool import AgentPool

        if isinstance(ref, AgentPool):
            registered = self._pools.get(ref.name)
            if registered is None:
                raise MeshError(f"Pool '{ref.name}' not in mesh; add it first")
            if registered is not ref:
                raise MeshError(
                    f"Pool '{ref.name}' is not the registered instance; "
                    "use mesh.get_pool() or the object originally added"
                )
            return registered
        if isinstance(ref, str):
            if ref in self._pools:
                return self._pools[ref]
            return self.get(ref)
        return self._resolve(ref)

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

        if isinstance(resolved_src, AgentPool) or isinstance(resolved_tgt, AgentPool):
            connection = _PoolConnection(resolved_src, resolved_tgt, gate)
            self._pool_connections.append(connection)
            if isinstance(resolved_src, AgentPool):
                for worker in resolved_src.workers:
                    self._wire_pool_source_worker(connection, worker)
            else:
                assert isinstance(resolved_tgt, AgentPool)
                self._connect_to_pool(resolved_src, resolved_tgt, gate)
            return

        # Standard agent-to-agent
        src_agent = resolved_src if isinstance(resolved_src, Agent) else self._resolve(source)
        tgt_agent = resolved_tgt if isinstance(resolved_tgt, Agent) else self._resolve(target)
        self._connect_agents(src_agent, tgt_agent, gate)

    def _wire_pool_source_worker(
        self,
        connection: _PoolConnection,
        worker: Agent,
    ) -> None:
        """Apply one logical source-pool policy to a current worker."""
        from signal_gating.pool import AgentPool

        if isinstance(connection.target, AgentPool):
            self._connect_to_pool(worker, connection.target, connection.gate)
        else:
            self._connect_agents(worker, connection.target, connection.gate)

    def _connect_to_pool(
        self,
        src: Agent,
        pool: AgentPool,
        gate: Gate | None = None,
    ) -> None:
        """Route each signal to a worker selected from current membership."""
        mesh = self

        async def route(signal: Signal) -> None:
            target = pool.select_worker(signal)
            await mesh._deliver(
                signal,
                source=src.name,
                target=target.name,
                send=target.inbox.send,
                action="routed",
                event_kind="signal",
                gate=gate,
                trace_gate="load_balance",
                trace_action="routed",
            )

        src._add_output(
            RouteFn(
                fn=route,
                source=src.name,
                target=pool.name,
                tag="pool_connect",
            )
        )

    def _connect_agents(
        self,
        src: Agent,
        tgt: Agent,
        gate: Gate | None = None,
    ) -> None:
        """Internal: wire two agents together."""
        edge = Edge(src, tgt, gate)
        self._edges.append(edge)
        mesh = self

        async def route(signal: Signal) -> None:
            await mesh._deliver(
                signal,
                source=src.name,
                target=tgt.name,
                send=tgt.inbox.send,
                action="routed",
                event_kind="signal",
                gate=gate,
                trace_gate="route",
                trace_action="routed",
            )

        src._add_output(
            RouteFn(fn=route, source=src.name, target=tgt.name, tag="connect")
        )

    # --- Pub/Sub Topics ---

    def create_topic(self, name: str) -> None:
        """Create a named topic for publish/subscribe communication.

        Topics decouple producers from consumers. Any agent can publish to a
        topic, and all subscribers receive the signal. This is the agent-native
        broadcast pattern, essential when the number and identity of consumers
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
            raise MeshError(f"Topic '{topic}' does not exist; create it first")
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

        Returns the number of subscribers that accepted the signal after
        interception. Blocked or closed subscribers do not abort delivery to
        the remaining subscribers.
        This is fire-and-forget. The signal is sent to each subscriber's
        inbox without waiting for processing.

            count = await mesh.publish("events", AlertSignal(message="alert"))
        """
        if topic not in self._topics:
            raise MeshError(f"Topic '{topic}' does not exist")
        # Snapshot: subscribe/unsubscribe during the awaits below must not
        # mutate the list we are iterating.
        subscribers = list(self._topics[topic])
        delivered = 0
        for subscriber in subscribers:
            if subscriber.inbox.closed:
                # A stopped subscriber must not abort the broadcast for the
                # rest; deliver to everyone reachable and report the count.
                logger.warning(
                    "publish('%s'): skipping subscriber '%s' (inbox closed)",
                    topic, subscriber.name,
                )
                continue

            async def send_to_subscriber(delivered_signal: Signal) -> None:
                try:
                    await subscriber.inbox.send(delivered_signal)
                except ChannelClosed as error:
                    raise _PublishDestinationClosedError from error

            try:
                outcome = await self._deliver(
                    signal,
                    source="mesh",
                    target=subscriber.name,
                    send=send_to_subscriber,
                    action="published",
                    trace_gate=f"topic:{topic}",
                    trace_action="published",
                    topic=topic,
                )
            except _PublishDestinationClosedError:
                logger.warning(
                    "publish('%s'): skipping subscriber '%s' (inbox closed)",
                    topic,
                    subscriber.name,
                )
                continue
            if outcome.delivered:
                delivered += 1
        return delivered

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

        Unlike Agent.request() which emits through the outbox (to all connected
        agents), Mesh.request() injects directly into a specific agent's inbox
        and captures the response.

        The target agent must call ``reply()`` or emit a signal with the
        matching correlation_id for the request to resolve. Both the request
        and correlated response cross the interceptor boundary; a blocked
        delivery raises ``MeshError`` immediately.

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
                try:
                    outcome = await self._deliver(
                        sig,
                        source=resolved.name,
                        target="mesh",
                        send=future.set_result,
                        action="response_received",
                        trace_gate="request",
                        correlation_id=cid,
                    )
                    if not outcome.delivered and not future.done():
                        future.set_exception(
                            self._delivery_error(
                                outcome,
                                source=resolved.name,
                                target="mesh",
                                action="response_received",
                            )
                        )
                except Exception as error:
                    if not future.done():
                        future.set_exception(error)

        resolved._outbox.append(capture)
        try:
            outcome = await self._deliver(
                tagged,
                source="mesh",
                target=resolved.name,
                send=resolved.inbox.send,
                action="request_sent",
                trace_gate="request",
                correlation_id=cid,
            )
            self._raise_if_blocked(
                outcome,
                source="mesh",
                target=resolved.name,
                action="request_sent",
            )
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

        Send work to N agents in parallel, then wait for all to respond.

            responses = await mesh.scatter(
                TaskSignal(task="analyze"),
                [analyst1, analyst2, analyst3],
                timeout=10.0,
            )

        Each target agent must `reply()` to the signal for gather to complete.
        Returns responses in the same order as targets. Scatter is all-or-nothing:
        if any target misses the deadline, no partial result is returned.
        Each resolved target must appear only once; duplicate object/name
        references raise ``MeshError`` before any work is sent.

        Raises:
            asyncio.TimeoutError: If one or more targets do not reply before
                ``timeout``. The error identifies every missing target.
            MeshError: If the same resolved agent appears more than once.
        """
        if not targets:
            return []
        cid = uuid4().hex
        resolved = [self._resolve(t) for t in targets]
        seen_targets: set[str] = set()
        duplicate_targets: list[str] = []
        for target in resolved:
            if target.name in seen_targets:
                if target.name not in duplicate_targets:
                    duplicate_targets.append(target.name)
            else:
                seen_targets.add(target.name)
        if duplicate_targets:
            duplicates = ", ".join(repr(name) for name in duplicate_targets)
            noun = "agent" if len(duplicate_targets) == 1 else "agents"
            raise MeshError(
                f"scatter targets must be unique; duplicate {noun}: {duplicates}. "
                "Pass each agent once."
            )

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[Signal]] = []
        capture_fns: list[tuple[Agent, Any]] = []

        try:
            for target in resolved:
                target_cid = f"{cid}:{target.name}"
                future: asyncio.Future[Signal] = loop.create_future()
                futures.append(future)

                # Capture replies from this target's outbox by correlation_id.
                # This avoids the previous bug where registering on _pending_requests
                # caused the signal to be intercepted before reaching handlers.
                def _make_capture(
                    f: asyncio.Future[Signal], tcid: str, tname: str
                ) -> Any:
                    async def capture(sig: Signal) -> None:
                        if sig.correlation_id == tcid and not f.done():
                            try:
                                outcome = await self._deliver(
                                    sig,
                                    source=tname,
                                    target="mesh",
                                    send=f.set_result,
                                    action="scatter_response",
                                    trace_gate="scatter",
                                    correlation_id=tcid,
                                )
                                if not outcome.delivered and not f.done():
                                    f.set_exception(
                                        self._delivery_error(
                                            outcome,
                                            source=tname,
                                            target="mesh",
                                            action="scatter_response",
                                        )
                                    )
                            except Exception as error:
                                if not f.done():
                                    f.set_exception(error)

                    return capture

                capture = _make_capture(future, target_cid, target.name)
                target._outbox.append(capture)  # noqa: SLF001
                capture_fns.append((target, capture))

            # Send signals to all targets.
            for target in resolved:
                target_cid = f"{cid}:{target.name}"
                scatter_signal = signal.evolve(correlation_id=target_cid)
                outcome = await self._deliver(
                    scatter_signal,
                    source="mesh",
                    target=target.name,
                    send=target.inbox.send,
                    action="scatter_sent",
                    trace_gate="scatter",
                    correlation_id=target_cid,
                )
                self._raise_if_blocked(
                    outcome,
                    source="mesh",
                    target=target.name,
                    action="scatter_sent",
                )

            done, pending = await asyncio.wait(
                futures,
                timeout=timeout,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            failed = next(
                (future for future in done if not future.cancelled() and future.exception()),
                None,
            )
            if failed is not None:
                for future in pending:
                    future.cancel()
                error = failed.exception()
                assert error is not None
                raise error
            if pending:
                missing_targets = [
                    target.name
                    for target, future in zip(resolved, futures, strict=True)
                    if future in pending
                ]
                for future in pending:
                    future.cancel()
                await self._record_event(
                    "scatter_timeout",
                    signal,
                    source="mesh",
                    missing_targets=missing_targets,
                    received_count=len(futures) - len(pending),
                    expected_count=len(futures),
                    timeout=timeout,
                )
                missing = ", ".join(repr(name) for name in missing_targets)
                raise asyncio.TimeoutError(
                    f"Scatter timed out after {timeout:g}s waiting for agents: {missing}"
                )
            return [future.result() for future in futures]
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
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
        """Execute a sequential multi-agent workflow, THE agent orchestration primitive.

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
                await self._record_event(
                    "workflow_step_start",
                    current,
                    source="mesh",
                    target=resolved.name,
                    step=i,
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
                await self._record_event(
                    "workflow_step_complete",
                    current,
                    source=resolved.name,
                    target="mesh",
                    step=i,
                )
            return current

        return await asyncio.wait_for(_run(), timeout=timeout)

    async def race(
        self,
        signal: Signal,
        targets: list[Agent | str],
        timeout: float = 30.0,
    ) -> Signal:
        """Race a signal across multiple agents. First response wins.

        Sends the same signal to N agents in parallel and returns the
        first allowed response received. Blocked sends and responses are
        ignored while another candidate remains; if every candidate is
        blocked, ``MeshError`` is raised immediately. All other pending
        responses are discarded.
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
        target_cids = [
            f"{cid}:{index}:{target.name}" for index, target in enumerate(resolved)
        ]
        remaining: set[str] = set(target_cids)
        blocked_errors: list[MeshError] = []

        def block_candidate(
            candidate_id: str,
            outcome: _DeliveryOutcome,
            *,
            source: str,
            target: str,
            action: str,
        ) -> None:
            if candidate_id not in remaining:
                return
            remaining.remove(candidate_id)
            blocked_errors.append(
                self._delivery_error(
                    outcome,
                    source=source,
                    target=target,
                    action=action,
                )
            )
            if not remaining and not future.done():
                future.set_exception(blocked_errors[0])

        for target, target_cid in zip(resolved, target_cids, strict=True):

            def _make_capture(
                f: asyncio.Future[Signal], tcid: str,
                tname: str,
            ) -> Any:
                async def capture(sig: Signal) -> None:
                    if sig.correlation_id == tcid and not f.done():
                        try:
                            outcome = await self._deliver(
                                sig,
                                source=tname,
                                target="mesh",
                                send=f.set_result,
                                action="race_response",
                                trace_gate="race",
                                correlation_id=tcid,
                            )
                            if not outcome.delivered:
                                block_candidate(
                                    tcid,
                                    outcome,
                                    source=tname,
                                    target="mesh",
                                    action="race_response",
                                )
                        except Exception as error:
                            if not f.done():
                                f.set_exception(error)
                            return

                return capture

            capture = _make_capture(future, target_cid, target.name)
            target._outbox.append(capture)
            capture_fns.append((target, capture))

        async def run_race() -> Signal:
            for target, target_cid in zip(resolved, target_cids, strict=True):
                race_signal = signal.evolve(correlation_id=target_cid)
                outcome = await self._deliver(
                    race_signal,
                    source="mesh",
                    target=target.name,
                    send=target.inbox.send,
                    action="race_sent",
                    trace_gate="race",
                    trace_action="sent",
                    correlation_id=target_cid,
                )
                if not outcome.delivered:
                    block_candidate(
                        target_cid,
                        outcome,
                        source="mesh",
                        target=target.name,
                        action="race_sent",
                    )

            result = await asyncio.wait_for(future, timeout=timeout)
            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=result.id,
                agent="mesh",
                gate="race",
                action="winner",
            )
            await self._record_event(
                "race_winner",
                result,
                source="mesh",
                target="",
            )
            return result

        try:
            return await run_race()
        finally:
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()
            for target, capture in capture_fns:
                try:
                    target._outbox.remove(capture)
                except ValueError:
                    pass

    # --- Tool Calling ---

    async def call_tool(
        self,
        target: Agent | str,
        tool_name: str,
        timeout: float = 30.0,
        **arguments: Any,
    ) -> Any:
        """Call a tool on a target agent and return the result.

        Sends a ToolCallSignal to the target agent, waits for the ToolResultSignal,
        and returns the result value. If the tool errors, raises AgentError.

        Args:
            target: The agent that exposes the tool.
            tool_name: Name of the tool to invoke.
            timeout: Maximum time to wait for the result.
            **arguments: Keyword arguments passed to the tool function.

        Returns:
            The tool function's return value.

        Raises:
            AgentError: If the tool returns an error.
            asyncio.TimeoutError: If the call exceeds the timeout.

        Example::

            result = await mesh.call_tool(
                analyst, "analyze",
                data="quarterly revenue", depth=2,
            )
        """
        return await self._call_tool(
            target,
            tool_name,
            arguments,
            timeout=timeout,
        )

    async def _call_tool(
        self,
        target: Agent | str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = 30.0,
        expected_binding_id: str = "",
    ) -> Any:
        """Call a tool with an optional binding identity check."""
        resolved = self._resolve(target)
        signal = ToolCallSignal(
            tool_name=tool_name,
            arguments=arguments,
            expected_binding_id=expected_binding_id,
        )
        await self._record_event(
            "tool_call_start",
            signal,
            source="mesh",
            target=resolved.name,
            tool_name=tool_name,
            argument_names=sorted(arguments),
        )
        response = await self.request(resolved, signal, timeout=timeout)
        if isinstance(response, ToolResultSignal):
            await self._record_event(
                "tool_call_complete",
                response,
                source=resolved.name,
                target="mesh",
                tool_name=tool_name,
                error=bool(response.error),
            )
            if response.error:
                raise AgentError(resolved.name, f"Tool '{tool_name}' failed: {response.error}")
            return response.result
        await self._record_event(
            "tool_call_complete",
            response,
            source=resolved.name,
            target="mesh",
            tool_name=tool_name,
            error=False,
        )
        return response

    def discover_tools(
        self,
        agent: Agent | str | None = None,
    ) -> dict[str, list[ToolSpec]]:
        """Discover tools across the mesh, the agent-native service registry.

        Without arguments, returns all tools from all agents. With an agent
        specified, returns only that agent's tools.

        This is how LLM-based agents discover what capabilities are available
        in the mesh at runtime. Feed this to an LLM's tool selection to enable
        dynamic, self-organizing multi-agent systems.

            # Discover all available tools
            all_tools = mesh.discover_tools()
            for agent_name, tools in all_tools.items():
                for tool in tools:
                    print(f"{agent_name}.{tool.name}: {tool.description}")

            # Discover tools for a specific agent
            tools = mesh.discover_tools(analyst)
        """
        if agent is not None:
            resolved = self._resolve(agent)
            return {resolved.name: resolved.list_tools()}
        return {
            a.name: a.list_tools()
            for a in self._agents.values()
            if a.list_tools()
        }

    # --- Map/Reduce ---

    async def map_reduce(
        self,
        signal: Signal,
        mappers: list[Agent | str],
        reducer: Agent | str,
        timeout: float = 60.0,
        mapper_timeout: float = 30.0,
    ) -> Signal:
        """Parallel map across agents, then reduce through a single agent.

        Distributes work across N specialized agents in parallel, then
        synthesizes their outputs through a reducer. Useful when multiple
        agents analyze the same data from different angles and a synthesizer
        combines their insights.

        Each mapper receives the signal and must ``reply()`` with its result.
        The reducer receives a signal carrying all mapper responses in its
        metadata (``metadata["responses"]``) and must ``reply()`` with the
        final combined result.

        Args:
            signal: The signal to distribute to all mappers.
            mappers: Agents that process the signal in parallel.
            reducer: Agent that combines all mapper responses.
            timeout: Maximum total time for the entire operation.
            mapper_timeout: Maximum time for each mapper.

        Returns:
            The reducer's response signal.

        Example::

            result = await mesh.map_reduce(
                AnalyzeSignal(data="quarterly revenue report"),
                mappers=[trend_analyst, risk_analyst, sentiment_analyst],
                reducer=synthesizer,
                timeout=30.0,
            )
            # synthesizer receives all three analyses and combines them
        """
        if not mappers:
            raise MeshError("map_reduce requires at least one mapper")

        async def _run() -> Signal:
            # Map phase: scatter to all mappers
            responses = await self.scatter(
                signal, mappers, timeout=mapper_timeout
            )

            # Build the reduce signal with all responses
            reduce_signal = signal.evolve(
                metadata={
                    **signal.metadata,
                    "responses": [r.model_dump() for r in responses],
                    "mapper_count": len(responses),
                },
            )

            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent="mesh",
                gate="map_reduce",
                action="reduce_start",
                mapper_count=len(responses),
            )
            await self._record_event(
                "map_reduce_reduce_start",
                reduce_signal,
                source="mesh",
                target=self._resolve(reducer).name,
                mapper_count=len(responses),
            )

            # Reduce phase: send combined result to reducer
            result = await self.request(
                reducer, reduce_signal, timeout=mapper_timeout
            )

            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=result.id,
                agent="mesh",
                gate="map_reduce",
                action="reduce_complete",
            )
            await self._record_event(
                "map_reduce_reduce_complete",
                result,
                source=self._resolve(reducer).name,
                target="mesh",
            )
            return result

        return await asyncio.wait_for(_run(), timeout=timeout)

    # --- Branching Workflow ---

    async def branch_workflow(
        self,
        signal: Signal,
        router: Callable[[Signal], str],
        branches: dict[str, list[Agent | str]],
        timeout: float = 60.0,
        step_timeout: float = 30.0,
    ) -> Signal:
        """Conditional workflow: route signals through different agent chains.

        This is the agent-native decision tree. A router function inspects the
        signal and returns a branch key. The signal then flows through the
        corresponding agent chain. This enables workflows that adapt their
        processing path based on signal content, not just static topology.

        Args:
            signal: The signal to route.
            router: Function that inspects the signal and returns a branch key.
            branches: Dict mapping branch keys to ordered lists of agents.
            timeout: Maximum total time for the entire workflow.
            step_timeout: Maximum time for each individual step.

        Returns:
            The final response signal from the chosen branch.

        Raises:
            MeshError: If the router returns a key not in branches.

        Example::

            result = await mesh.branch_workflow(
                TaskSignal(task="analyze", priority=9),
                router=lambda s: "critical" if s.priority >= 8 else "normal",
                branches={
                    "critical": [validator, deep_analyzer, reviewer],
                    "normal": [quick_analyzer],
                },
            )
        """
        if not branches:
            raise MeshError("branch_workflow requires at least one branch")

        async def _run() -> Signal:
            branch_key = router(signal)
            if branch_key not in branches:
                raise MeshError(
                    f"Router returned unknown branch {branch_key!r}. "
                    f"Available: {list(branches.keys())}"
                )

            steps = branches[branch_key]
            self.tracer.record(
                trace_id=signal.trace_id,
                signal_id=signal.id,
                agent="mesh",
                gate="branch_workflow",
                action="branch_selected",
                branch=branch_key,
                steps=len(steps),
            )
            await self._record_event(
                "branch_selected",
                signal,
                source="mesh",
                target="",
                branch=branch_key,
                steps=len(steps),
            )

            return await self.workflow(
                signal, steps, timeout=timeout, step_timeout=step_timeout
            )

        return await asyncio.wait_for(_run(), timeout=timeout)

    # --- Graceful Shutdown ---

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """Wait until every agent has handled its queued and in-flight work.

        This includes signals that handlers emit to downstream agents while
        the mesh is draining. Raises ``asyncio.TimeoutError`` with the active
        agent states when the mesh does not become idle before ``timeout``.
        """
        if timeout < 0:
            raise ValueError("timeout must be >= 0")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        interval = 0.01

        while True:
            active = [
                agent
                for agent in self._agents.values()
                if agent.inbox.pending > 0 or agent.busy
            ]
            if not active:
                return

            remaining = deadline - loop.time()
            if remaining <= 0:
                states = ", ".join(
                    f"{agent.name}(pending={agent.inbox.pending}, busy={agent.busy})"
                    for agent in active
                )
                raise asyncio.TimeoutError(
                    f"Mesh did not become idle within {timeout:g}s; active agents: {states}"
                )

            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * 1.5, 0.2)

    async def stop(self, drain: bool = False, drain_timeout: float = 10.0) -> None:
        """Stop all agents gracefully.

        Args:
            drain: If True, wait for agents to process all pending signals
                   before shutting down. If False, stop immediately.
            drain_timeout: Maximum seconds to wait for drain (default 10s).
        """
        self._running = False
        if drain:
            try:
                await self.wait_idle(timeout=drain_timeout)
            except asyncio.TimeoutError as error:
                logger.warning("%s; forcing shutdown", error)
        await asyncio.gather(*(agent.stop() for agent in self._agents.values()))
        logger.info("Mesh stopped")

    def _resolve(self, agent: Agent | str) -> Agent:
        if isinstance(agent, str):
            return self.get(agent)
        registered = self._agents.get(agent.name)
        if registered is None:
            raise MeshError(f"Agent '{agent.name}' not in mesh; add it first")
        if registered is not agent:
            raise MeshError(
                f"Agent '{agent.name}' is not the registered instance; "
                "use mesh.get() or the object originally added"
            )
        return registered

    async def __aenter__(self) -> Mesh:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: Any,
    ) -> None:
        # A clean context exit is the safe common path: finish work, including
        # downstream emissions, before closing inboxes. On an exceptional exit,
        # stop promptly instead of continuing potentially unwanted side effects.
        await self.stop(drain=exc_type is None)

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
