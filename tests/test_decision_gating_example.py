"""Contracts for the public decision-model integration example."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "decision_gating.py"
spec = importlib.util.spec_from_file_location("decision_gating_example", EXAMPLE)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


async def test_fixture_routes_real_sdk_work_and_traces_drops() -> None:
    result = await example.run_scenario(example.fixture_judge)
    assert result["signals_in"] == 5
    assert result["handler_calls"] == 4
    assert result["model_calls"] == 4
    assert result["trace_drops"] == 1
    assert result["missed_required_cases"] == []
    assert result["routes"][-1]["reason"] == "unknown_preserved"


async def test_protected_work_never_calls_the_judge() -> None:
    async def judge(alert):
        pytest.fail("critical priority must bypass the model")

    alert = example.WorkAlert(case="critical", summary="Outage", priority=9)
    routes = []
    assert await example.decision_gate(judge, routes).process(alert) is alert
    assert routes[0].reason == "priority_bypass"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, "0.99", True, None])
async def test_invalid_judgments_preserve_work(score) -> None:
    async def judge(alert):
        return example.Decision(score, "test-model")

    alert = example.WorkAlert(case="unknown", summary="Unclear impact")
    routes = []
    assert await example.decision_gate(judge, routes).process(alert) is alert
    assert routes[0].reason == "unknown_preserved"


async def test_timeout_preserves_work_and_cancels_judgment() -> None:
    cancelled = asyncio.Event()

    async def judge(alert):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    alert = example.WorkAlert(case="slow", summary="Unknown impact")
    gate = example.decision_gate(judge, [], timeout=0.01)
    assert await gate.process(alert) is alert
    assert cancelled.is_set()


async def test_caller_cancellation_does_not_admit_work() -> None:
    started = asyncio.Event()
    routes = []

    async def judge(alert):
        started.set()
        await asyncio.Event().wait()

    alert = example.WorkAlert(case="cancelled", summary="Unknown impact")
    task = asyncio.create_task(example.decision_gate(judge, routes).process(alert))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert routes == []


@pytest.mark.parametrize("score,admitted", [(0.949, True), (0.95, False), (1.0, False)])
async def test_threshold_boundary(score, admitted) -> None:
    async def judge(alert):
        return example.Decision(score, "test-model")

    alert = example.WorkAlert(case="boundary", summary="Informational")
    result = await example.decision_gate(judge, []).process(alert)
    assert (result is not None) is admitted


def test_official_noul_schema_is_parsed() -> None:
    result = example.parse_jev({
        "model": "jev-1.13.0",
        "answers": {"is_noise": {"type": "noul", "noul": 0.98}},
    })
    assert result.noise_probability == 0.98
    assert result.model == "jev-1.13.0"


@pytest.mark.parametrize("payload", [
    {}, [], {"model": "jev", "answers": {"is_noise": {"type": "choice", "noul": 1}}},
    {"answers": {"is_noise": {"type": "noul", "noul": 1}}},
])
def test_unknown_schema_is_rejected(payload) -> None:
    with pytest.raises(ValueError):
        example.parse_jev(payload)
