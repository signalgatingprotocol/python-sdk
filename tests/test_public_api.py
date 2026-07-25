from signal_gating import (
    Agent,
    Gate,
    Mesh,
    MeshEvent,
    Receipt,
    Signal,
    TrajectoryRecorder,
    core,
)


def test_stable_core_exports_exact_contract() -> None:
    assert core.__all__ == [
        "Agent",
        "Gate",
        "Mesh",
        "MeshEvent",
        "Receipt",
        "Signal",
        "TrajectoryRecorder",
    ]
    assert core.Agent is Agent
    assert core.Gate is Gate
    assert core.Mesh is Mesh
    assert core.MeshEvent is MeshEvent
    assert core.Receipt is Receipt
    assert core.Signal is Signal
    assert core.TrajectoryRecorder is TrajectoryRecorder
