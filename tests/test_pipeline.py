"""Tests for Pipeline gate composition."""

from signal_gating import Gate, Pipeline, Signal


async def test_empty_pipeline():
    p = Pipeline()
    s = Signal(priority=5)
    result = await p.process(s)
    assert result is s


async def test_pipeline_process():
    p = Pipeline([
        Gate.by_priority(3),
        Gate.transform(lambda s: s.evolve(priority=s.priority + 10)),
    ])
    result = await p.process(Signal(priority=5))
    assert result is not None
    assert result.priority == 15


async def test_pipeline_rejection():
    p = Pipeline([
        Gate.by_priority(10),
        Gate.transform(lambda s: s.evolve(priority=99)),
    ])
    result = await p.process(Signal(priority=3))
    assert result is None


async def test_pipeline_add():
    p = Pipeline()
    p.add(Gate.by_priority(1))
    p.add(Gate.transform(lambda s: s.evolve(priority=0)))
    assert len(p) == 2


async def test_pipeline_to_gate():
    p = Pipeline([
        Gate.by_priority(3),
        Gate.transform(lambda s: s.evolve(priority=s.priority * 2)),
    ])
    gate = p.to_gate()
    result = await gate.process(Signal(priority=5))
    assert result is not None
    assert result.priority == 10


async def test_pipeline_repr():
    p = Pipeline([Gate.passthrough(), Gate.block()])
    r = repr(p)
    assert "passthrough" in r
    assert "block" in r
