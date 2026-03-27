"""Signal Gating Protocol — agent-native signal orchestration.

The Signal Gating Protocol provides composable primitives for building
autonomous multi-agent systems with controlled, observable signal flow.

Core primitives:
    Signal   — Typed, immutable events that flow through the system
    Gate     — Composable predicates that control signal flow
    Channel  — Async typed conduits for signal transport
    Agent    — Autonomous signal processors with lifecycle management
    Pipeline — Ordered gate chains for building processing flows
    Mesh     — Agent network topology with gated connections

Quick start:
    from signal_gating import Signal, Gate, Agent, Mesh

    class TaskSignal(Signal):
        task: str

    planner = Agent("planner")
    worker = Agent("worker", gates=[Gate.by_priority(3)])

    @worker.on(TaskSignal)
    async def handle(signal: TaskSignal):
        print(f"Working on: {signal.task}")

    mesh = Mesh([planner, worker])
    mesh.connect(planner, worker)

    async with mesh:
        await planner.emit(TaskSignal(task="build", priority=5))
"""

from signal_gating.agent import Agent
from signal_gating.channel import Channel
from signal_gating.errors import (
    AgentError,
    ChannelClosed,
    ChannelFull,
    GateRejected,
    MeshError,
    SignalGatingError,
    SignalValidationError,
)
from signal_gating.gate import Gate
from signal_gating.mesh import Mesh
from signal_gating.pipeline import Pipeline
from signal_gating.signal import Signal
from signal_gating.tracing import Span, Tracer

__all__ = [
    "Agent",
    "AgentError",
    "Channel",
    "ChannelClosed",
    "ChannelFull",
    "Gate",
    "GateRejected",
    "Mesh",
    "MeshError",
    "Pipeline",
    "Signal",
    "SignalGatingError",
    "SignalValidationError",
    "Span",
    "Tracer",
]

__version__ = "0.1.0"
