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


class ChannelFull(SignalGatingError):
    """Channel buffer is full and would block."""


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
    """Circuit breaker is open — calls are being rejected."""

    def __init__(self, gate_name: str, until: float):
        self.gate_name = gate_name
        self.until = until
        super().__init__(f"Circuit '{gate_name}' is open until {until:.1f}")
