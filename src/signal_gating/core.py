"""The stable Signal Gating Protocol core.

This module is the compatibility-focused surface for production integrations.
Broader orchestration helpers remain available from their owning modules and
the package root while the project is alpha.
"""

from signal_gating.agent import Agent
from signal_gating.gate import Gate
from signal_gating.mesh import Mesh, MeshEvent
from signal_gating.signal import Signal
from signal_gating.trajectory import Receipt, TrajectoryRecorder

__all__ = [
    "Agent",
    "Gate",
    "Mesh",
    "MeshEvent",
    "Receipt",
    "Signal",
    "TrajectoryRecorder",
]
