"""Exception hierarchy for the Signal Gating Protocol."""

from __future__ import annotations


class SignalGatingError(Exception):
    """Base exception for all Signal Gating Protocol errors."""


class GateRejected(SignalGatingError):
    """A gate rejected a signal."""

    def __init__(self, gate_name: str, signal_id: str, reason: str = ""):
        self.gate_name = gate_name
        self.signal_id = signal_id
        self.reason = reason
        msg = f"Gate '{gate_name}' rejected signal '{signal_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ChannelClosed(SignalGatingError):
    """Attempted to send/receive on a closed channel."""

    def __init__(self, message: str = ""):
        super().__init__(
            message
            or "Channel is closed; create a new channel or check channel.closed before sending."
        )


class ChannelFull(SignalGatingError):
    """Channel buffer is full and would block."""

    def __init__(self, message: str = ""):
        super().__init__(
            message
            or "Channel buffer is full; use send_wait() or increase buffer_size."
        )


class AgentError(SignalGatingError):
    """An agent encountered an error during signal processing."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}': {message}")


class MeshError(SignalGatingError):
    """Error in mesh topology or operation."""


class SignalValidationError(SignalGatingError):
    """Signal payload failed validation."""


class CircuitOpenError(SignalGatingError):
    """Circuit breaker is open; calls are being rejected."""

    def __init__(self, gate_name: str, until: float):
        self.gate_name = gate_name
        self.until = until
        super().__init__(f"Circuit '{gate_name}' is open until {until:.1f}")


class SignalSerializationError(SignalGatingError):
    """A signal could not be serialized to, or reconstructed from, its wire form."""


class UnknownSignalType(SignalSerializationError):
    """A wire envelope references a signal type that is not in the registry.

    Raised by ``Signal.from_wire`` / ``from_json`` in strict mode when no class
    is registered under the envelope's wire name. Import (or explicitly
    register) the signal class so it self-registers, or pass ``strict=False``
    to reconstruct a best-effort base ``Signal`` instead.
    """

    def __init__(self, type_name: str):
        self.type_name = type_name
        super().__init__(
            f"Unknown signal type {type_name!r}: no class is registered under this "
            "wire name. Import or register the class, or use strict=False."
        )


class TaskRejected(SignalGatingError):
    """A TaskBoard gate refused a task transition.

    Gate combinators collapse names, so gate_name is the outermost gate's
    name — the algebra cannot know which leaf rejected.
    """

    def __init__(self, task_id: str, gate_name: str = "") -> None:
        self.task_id = task_id
        self.gate_name = gate_name
        super().__init__(f"task {task_id!r} rejected by gate {gate_name!r}")


class TeamError(SignalGatingError):
    """Team protocol misuse (duplicate enrollment, bad assign, reuse after dissolve)."""


class BudgetExceeded(SignalGatingError):
    """A Script run exceeded its agent budget."""

    def __init__(self, budget: int, key: str) -> None:
        self.budget = budget
        self.key = key
        super().__init__(f"script budget of {budget} steps exceeded at step {key!r}")
