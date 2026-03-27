"""Mesh — the agent network topology that ties everything together."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from signal_gating.agent import Agent
from signal_gating.errors import MeshError
from signal_gating.gate import Gate
from signal_gating.signal import Signal

logger = logging.getLogger("signal_gating.mesh")


class Edge:
    """A directional connection between two agents, optionally gated."""

    def __init__(self, source: Agent, target: Agent, gate: Gate | None = None):
        self.source = source
        self.target = target
        self.gate = gate


class Mesh:
    """A network of agents connected by gated edges.

    The mesh manages agent lifecycles and signal routing:

        mesh = Mesh()
        mesh.add(planner)
        mesh.add(worker)
        mesh.connect(planner, worker, gate=priority_gate)

        async with mesh:
            await planner.emit(TaskSignal(task="build"))
    """

    def __init__(self, agents: list[Agent] | None = None):
        self._agents: dict[str, Agent] = {}
        self._edges: list[Edge] = []
        self._running = False
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

        async def route(signal: Signal) -> None:
            if gate is not None:
                result = await gate.process(signal)
                if result is None:
                    return
                signal = result
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

    async def start(self) -> None:
        """Start all agents in the mesh."""
        self._running = True
        for agent in self._agents.values():
            await agent.start()
        # Yield control so agent tasks can start running
        await asyncio.sleep(0)
        logger.info(f"Mesh started with {len(self._agents)} agents, {len(self._edges)} edges")

    async def stop(self) -> None:
        """Stop all agents gracefully."""
        self._running = False
        await asyncio.gather(*(agent.stop() for agent in self._agents.values()))
        logger.info("Mesh stopped")

    async def inject(self, target: Agent | str, signal: Signal) -> None:
        """Inject a signal directly into an agent's inbox."""
        agent = self._resolve(target)
        await agent.inbox.send(signal)

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
