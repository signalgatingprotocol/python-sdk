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
