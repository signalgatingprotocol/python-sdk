"""Signal Gating Protocol: agent-native signal orchestration.

The Signal Gating Protocol provides composable primitives for building
autonomous multi-agent systems with controlled, observable signal flow.

Core primitives:
    Signal   : Typed, immutable events that flow through the system
    Gate     : Composable predicates that control signal flow
    Channel  : Async typed conduits for signal transport
    Agent    : Autonomous signal processors with lifecycle management
    Pipeline : Ordered gate chains for building processing flows
    Mesh     : Agent network topology with gated connections

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

from signal_gating.agent import (
    Agent,
    AgentContext,
    DeadLetterQueue,
    ErrorHook,
    ToolCallSignal,
    ToolResultSignal,
    ToolSpec,
)
from signal_gating.channel import Channel, PriorityChannel
from signal_gating.errors import (
    AgentError,
    ChannelClosed,
    ChannelFull,
    CircuitOpenError,
    GateRejected,
    MeshError,
    SignalGatingError,
    SignalSerializationError,
    SignalValidationError,
    UnknownSignalType,
)
from signal_gating.gate import Gate
from signal_gating.llm import LLMAgent, MeshToolProvider, Message, ToolProvider
from signal_gating.mesh import Edge, Mesh, MeshEvent, MeshEventSink
from signal_gating.pipeline import Pipeline
from signal_gating.pool import AgentPool
from signal_gating.registry import (
    WIRE_VERSION,
    from_wire,
    lookup_signal,
    register_signal,
    registered_signals,
    to_wire,
)
from signal_gating.signal import Signal
from signal_gating.tracing import OpenTelemetrySpanExporter, Span, SpanSink, Tracer
from signal_gating.trajectory import (
    Receipt,
    ReplayDelivery,
    ReplayResult,
    TrajectoryRecorder,
    TrajectoryReplayRunner,
)

__all__ = [
    "WIRE_VERSION",
    "Agent",
    "AgentContext",
    "AgentError",
    "AgentPool",
    "Channel",
    "ChannelClosed",
    "ChannelFull",
    "CircuitOpenError",
    "DeadLetterQueue",
    "Edge",
    "ErrorHook",
    "Gate",
    "GateRejected",
    "LLMAgent",
    "Mesh",
    "MeshEvent",
    "MeshEventSink",
    "MeshError",
    "MeshToolProvider",
    "Message",
    "OpenTelemetrySpanExporter",
    "Pipeline",
    "PriorityChannel",
    "Receipt",
    "ReplayDelivery",
    "ReplayResult",
    "Signal",
    "SignalGatingError",
    "SignalSerializationError",
    "SignalValidationError",
    "Span",
    "SpanSink",
    "ToolCallSignal",
    "ToolProvider",
    "ToolResultSignal",
    "ToolSpec",
    "Tracer",
    "TrajectoryRecorder",
    "TrajectoryReplayRunner",
    "UnknownSignalType",
    "from_wire",
    "lookup_signal",
    "register_signal",
    "registered_signals",
    "to_wire",
]

__version__ = "0.1.0"
