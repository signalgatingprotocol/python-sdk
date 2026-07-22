"""Closed-loop acceptance tests for model/harness-agnostic focused improvement."""

from __future__ import annotations

import asyncio
import stat
import time
from dataclasses import dataclass, replace

import pytest

from signal_gating import (
    AcceptancePolicy,
    Assessment,
    EvaluationCase,
    EvaluationError,
    EvaluationReport,
    EvaluationSuite,
    ImprovementEvent,
    ImprovementHistory,
    ImprovementLoop,
    Objective,
    RetentionPolicy,
)
from signal_gating.errors import SignalSerializationError


@dataclass(frozen=True)
class Candidate:
    name: str
    reasoning: float
    reliability: float


CASES = (
    EvaluationCase("easy", "2 + 2", weight=1.0),
    EvaluationCase("hard", "prove it", weight=2.0),
)

OBJECTIVES = (
    Objective("reasoning", target=0.8, weight=2.0),
    Objective("reliability", target=0.8, weight=1.0),
)


def harness(candidate: Candidate, case: EvaluationCase[str]) -> dict[str, float]:
    hard_penalty = 0.1 if case.id == "hard" else 0.0
    return {
        "reasoning": candidate.reasoning - hard_penalty,
        "reliability": candidate.reliability,
    }


def evaluator(
    case: EvaluationCase[str], output: dict[str, float]
) -> Assessment:
    return Assessment(
        scores=output,
        evidence={
            name: f"{case.id}:{name}={score:.2f}" for name, score in output.items()
        },
    )


def identify(candidate: Candidate) -> str:
    return candidate.name


async def test_suite_aggregates_weighted_scores_and_selects_focus():
    suite = EvaluationSuite(CASES, OBJECTIVES, harness, evaluator)
    report = await suite.evaluate(Candidate("base", 0.6, 0.9), candidate_id="base")

    assert report.dimension_scores["reasoning"] == pytest.approx((0.6 + 2 * 0.5) / 3)
    assert report.dimension_scores["reliability"] == pytest.approx(0.9)
    assert report.target_met is False
    focus = report.focus()
    assert focus.dimension == "reasoning"
    assert focus.case_ids == ("hard", "easy")
    assert focus.evidence[0].startswith("hard:reasoning=")


async def test_suite_reduces_repeated_samples_by_median_in_stable_order():
    calls: list[str] = []
    values = iter([0.1, 0.9, 0.5, 0.8, 0.4, 0.6])

    def noisy_harness(candidate: str, case: EvaluationCase[str]) -> float:
        calls.append(case.id)
        return next(values)

    def score(case: EvaluationCase[str], output: float) -> Assessment:
        return Assessment(scores={"quality": output}, evidence={"quality": case.id})

    suite = EvaluationSuite(
        (EvaluationCase("a", "a"), EvaluationCase("b", "b")),
        (Objective("quality", target=1.0),),
        noisy_harness,
        score,
        samples=3,
        max_concurrency=1,
    )
    report = await suite.evaluate("candidate", candidate_id="candidate")

    assert calls == ["a", "a", "a", "b", "b", "b"]
    assert report.cases[0].scores["quality"] == 0.5
    assert report.cases[1].scores["quality"] == 0.6


async def test_suite_discards_raw_outputs_by_default_and_retains_only_on_opt_in():
    payload = b"large-model-output" * 1000

    def output_harness(candidate: str, case: EvaluationCase[str]) -> bytes:
        return payload

    def output_evaluator(case: EvaluationCase[str], output: bytes) -> Assessment:
        return Assessment(scores={"quality": 1.0}, evidence={"quality": "passed"})

    cases = (EvaluationCase("case", "input"),)
    objectives = (Objective("quality", target=1.0),)
    discarded = await EvaluationSuite(
        cases, objectives, output_harness, output_evaluator
    ).evaluate("candidate", candidate_id="candidate")
    assert discarded.cases[0].outputs == ()

    retained = await EvaluationSuite(
        cases,
        objectives,
        output_harness,
        output_evaluator,
        retain_outputs=True,
    ).evaluate("candidate", candidate_id="candidate")

    assert retained.cases[0].outputs == (payload,)


async def test_suite_creates_only_bounded_worker_tasks():
    concurrency = 3
    started = 0
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def blocking_harness(candidate: str, case: EvaluationCase[int]) -> float:
        nonlocal started
        started += 1
        if started == concurrency:
            saturated.set()
        await release.wait()
        return 1.0

    def score(case: EvaluationCase[int], output: float) -> Assessment:
        return Assessment(scores={"quality": output}, evidence={"quality": case.id})

    suite = EvaluationSuite(
        tuple(EvaluationCase(str(index), index) for index in range(200)),
        (Objective("quality", target=1.0),),
        blocking_harness,
        score,
        max_concurrency=concurrency,
    )
    existing = set(asyncio.all_tasks())
    evaluation = asyncio.create_task(suite.evaluate("candidate", candidate_id="candidate"))
    await asyncio.wait_for(saturated.wait(), timeout=1.0)
    try:
        spawned = [
            task
            for task in asyncio.all_tasks()
            if task not in existing and not task.done()
        ]
        # One evaluate task plus exactly `max_concurrency` workers. The suite
        # must not create one task per case/sample.
        assert len(spawned) <= concurrency + 1
    finally:
        release.set()
        await evaluation


async def test_suite_cancels_sibling_workers_after_failure():
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def failing_harness(candidate: str, case: EvaluationCase[str]) -> float:
        if case.id == "boom":
            await sibling_started.wait()
            raise RuntimeError("boom")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    def score(case: EvaluationCase[str], output: float) -> Assessment:
        return Assessment(scores={"quality": output}, evidence={"quality": case.id})

    suite = EvaluationSuite(
        (EvaluationCase("boom", "boom"), EvaluationCase("blocked", "blocked")),
        (Objective("quality", target=1.0),),
        failing_harness,
        score,
        max_concurrency=2,
    )

    with pytest.raises(EvaluationError, match="RuntimeError: boom"):
        await suite.evaluate("candidate", candidate_id="candidate")
    assert sibling_cancelled.is_set()


async def test_suite_fails_closed_on_missing_score_or_evidence():
    def bad_evaluator(case: EvaluationCase[str], output: object) -> Assessment:
        return Assessment(scores={"reasoning": 0.5}, evidence={"reasoning": "seen"})

    suite = EvaluationSuite(CASES, OBJECTIVES, harness, bad_evaluator)
    with pytest.raises(EvaluationError, match="exact objective set"):
        await suite.evaluate(Candidate("base", 0.5, 0.5), candidate_id="base")


async def test_loop_reaches_target_through_repeated_focused_improvement():
    async def improve(context):
        current = context.incumbent
        if context.focus.dimension == "reasoning":
            return replace(
                current,
                name=f"candidate-{context.iteration}",
                reasoning=current.reasoning + 0.2,
            )
        return replace(
            current,
            name=f"candidate-{context.iteration}",
            reliability=current.reliability + 0.2,
        )

    loop = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
    )
    result = await loop.run(Candidate("base", 0.4, 0.5), max_iterations=6)

    assert result.stop_reason == "target_met"
    assert result.report.target_met
    assert result.incumbent.reasoning == pytest.approx(1.0)
    assert result.incumbent.reliability == pytest.approx(0.9)
    assert all(record.accepted for record in result.records)
    assert [record.focus_dimension for record in result.records] == [
        "reasoning",
        "reasoning",
        "reliability",
        "reasoning",
        "reliability",
    ]


async def test_loop_rejects_case_regression_then_improves_from_incumbent():
    objectives = (
        Objective("reasoning", target=0.8),
        Objective("reliability", target=0.8, regression_tolerance=0.05),
    )

    async def improve(context):
        current = context.incumbent
        if context.iteration == 1:
            return Candidate("unsafe", current.reasoning + 0.3, current.reliability - 0.2)
        assert current.name == "base"  # rejected candidate never becomes incumbent
        assert context.history[-1].accepted is False
        return Candidate("safe", current.reasoning + 0.3, current.reliability + 0.1)

    loop = ImprovementLoop(
        EvaluationSuite(CASES, objectives, harness, evaluator),
        improve,
        identify=identify,
        policy=AcceptancePolicy(min_progress_delta=0.01, min_focus_delta=0.01),
    )
    result = await loop.run(Candidate("base", 0.6, 0.8), max_iterations=2)

    assert [record.accepted for record in result.records] == [False, True]
    assert result.records[0].regressions
    assert result.incumbent.name == "safe"


async def test_case_guardrail_catches_regression_hidden_by_better_average():
    cases = (EvaluationCase("a", "a"), EvaluationCase("b", "b"))
    objectives = (Objective("quality", target=0.9),)
    candidates = {
        "base": {"a": 0.4, "b": 0.8},
        "candidate": {"a": 0.8, "b": 0.6},
    }

    def case_harness(candidate: str, case: EvaluationCase[str]) -> float:
        return candidates[candidate][case.id]

    def case_evaluator(case: EvaluationCase[str], output: float) -> Assessment:
        return Assessment(scores={"quality": output}, evidence={"quality": case.id})

    def improve(context):
        return "candidate"

    loop = ImprovementLoop(
        EvaluationSuite(cases, objectives, case_harness, case_evaluator),
        improve,
        identify=lambda candidate: candidate,
    )
    result = await loop.run("base", max_iterations=1)

    record = result.records[0]
    assert record.candidate_progress > record.baseline_progress
    assert record.accepted is False
    assert [(item.scope, item.case_id) for item in record.regressions] == [("case", "b")]
    assert record.candidate_evidence["b"]["quality"] == ("b",)


async def test_policy_refuses_reports_with_different_objective_contracts():
    suite = EvaluationSuite(CASES, OBJECTIVES, harness, evaluator)
    baseline = await suite.evaluate(Candidate("base", 0.5, 0.5), candidate_id="base")
    incompatible = EvaluationReport(
        candidate_id="candidate",
        objectives=(
            Objective("reasoning", target=0.9, weight=2.0),
            OBJECTIVES[1],
        ),
        cases=baseline.cases,
        dimension_scores=baseline.dimension_scores,
    )

    with pytest.raises(EvaluationError, match="different objective contracts"):
        AcceptancePolicy().compare(baseline, incompatible, baseline.focus())


async def test_history_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "improvements.jsonl"
    history = ImprovementHistory(path)

    async def improve(context):
        return Candidate("next", 0.9, 0.9)

    loop = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
        history=history,
    )
    result = await loop.run(Candidate("base", 0.6, 0.8), max_iterations=1)

    reloaded = ImprovementHistory(path)
    assert reloaded.records == result.records
    assert reloaded.head_digest == history.head_digest
    path.write_text(path.read_text().replace('"accepted": true', '"accepted": false'))
    with pytest.raises(SignalSerializationError, match="history chain broken"):
        ImprovementHistory(path)


async def test_reloaded_history_informs_next_run_without_polluting_run_records(tmp_path):
    path = tmp_path / "improvements.jsonl"

    def first_improvement(context):
        return Candidate("first", 0.9, 0.9)

    first = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        first_improvement,
        identify=identify,
        history=ImprovementHistory(path),
    )
    await first.run(Candidate("base", 0.5, 0.8), max_iterations=1)

    reloaded = ImprovementHistory(path)

    def next_improvement(context):
        assert len(context.history) == 1
        assert context.history[0].candidate_id == "first"
        return Candidate("second", 1.0, 1.0)

    second = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        next_improvement,
        identify=identify,
        history=reloaded,
    )
    result = await second.run(Candidate("new-base", 0.6, 0.8), max_iterations=1)

    assert len(result.records) == 1
    assert result.records[0].candidate_id == "second"
    assert len(reloaded.records) == 2


async def test_history_prunes_stale_records_and_rebuilds_valid_chain(tmp_path):
    path = tmp_path / "improvements.jsonl"

    def improve(context):
        return Candidate("seed", 0.9, 0.9)

    seed = await ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
    ).run(Candidate("base", 0.6, 0.8), max_iterations=1)
    now = time.time()
    history = ImprovementHistory(path, retention=None)
    history.append(replace(seed.records[0], candidate_id="old", created_at=now - 3600))
    history.append(replace(seed.records[0], candidate_id="recent", created_at=now))

    compacted = ImprovementHistory(
        path,
        retention=RetentionPolicy(
            max_age_seconds=60,
            max_records=None,
            max_bytes=None,
        ),
    )

    assert [record.candidate_id for record in compacted.records] == ["recent"]
    assert ImprovementHistory(path, retention=None).records == compacted.records


async def test_history_enforces_count_and_exact_file_size_caps(tmp_path):
    def improve(context):
        return Candidate("seed", 0.9, 0.9)

    seed = await ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
    ).run(Candidate("base", 0.6, 0.8), max_iterations=1)
    record = seed.records[0]

    count_path = tmp_path / "count.jsonl"
    counted = ImprovementHistory(
        count_path,
        retention=RetentionPolicy(
            max_age_seconds=None,
            max_records=2,
            max_bytes=None,
        ),
    )
    for index in range(4):
        counted.append(replace(record, candidate_id=f"count-{index}"))
    assert [item.candidate_id for item in counted.records] == ["count-2", "count-3"]

    sample_path = tmp_path / "sample.jsonl"
    sample = ImprovementHistory(sample_path, retention=None)
    large = replace(
        record,
        candidate_id="size-0",
        focus_evidence=("x" * 1000,),
        candidate_evidence={"case": {"quality": ("y" * 1000,)}},
    )
    sample.append(large)
    one_record_bytes = sample_path.stat().st_size

    size_path = tmp_path / "size.jsonl"
    sized = ImprovementHistory(
        size_path,
        retention=RetentionPolicy(
            max_age_seconds=None,
            max_records=None,
            max_bytes=one_record_bytes + 64,
        ),
    )
    for index in range(3):
        sized.append(replace(large, candidate_id=f"size-{index}"))

    assert size_path.stat().st_size <= one_record_bytes + 64
    assert [item.candidate_id for item in sized.records] == ["size-2"]
    assert stat.S_IMODE(size_path.stat().st_mode) == 0o600
    assert ImprovementHistory(size_path, retention=None).records == sized.records


async def test_run_result_keeps_current_records_when_durable_history_compacts(tmp_path):
    objectives = (
        Objective("reasoning", target=2.0),
        Objective("reliability", target=2.0),
    )

    def improve(context):
        assert len(context.history) == context.iteration - 1
        current = context.incumbent
        return Candidate(
            f"candidate-{context.iteration}",
            current.reasoning + 0.1,
            current.reliability + 0.1,
        )

    history = ImprovementHistory(
        tmp_path / "bounded.jsonl",
        retention=RetentionPolicy(
            max_age_seconds=None,
            max_records=1,
            max_bytes=None,
        ),
    )
    result = await ImprovementLoop(
        EvaluationSuite(CASES, objectives, harness, evaluator),
        improve,
        identify=identify,
        history=history,
    ).run(Candidate("base", 0.4, 0.5), max_iterations=3)

    assert len(result.records) == 3
    assert len(history.records) == 1
    assert history.records[0].candidate_id == "candidate-3"


async def test_loop_emits_complete_observable_feedback_cycle():
    events: list[ImprovementEvent] = []

    async def observe(event: ImprovementEvent) -> None:
        events.append(event)

    async def improve(context):
        return Candidate("next", 1.0, 1.0)

    loop = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
        observer=observe,
    )
    result = await loop.run(Candidate("base", 0.6, 0.8), max_iterations=1)

    assert result.stop_reason == "target_met"
    assert [event.action for event in events] == [
        "baseline_evaluated",
        "focus_selected",
        "candidate_evaluated",
        "candidate_accepted",
        "target_met",
    ]
    assert len({event.trace_id for event in events}) == 1


async def test_observer_failure_is_counted_without_stopping_improvement():
    def broken_observer(event: ImprovementEvent) -> None:
        raise RuntimeError(event.action)

    def improve(context):
        return Candidate("next", 1.0, 1.0)

    loop = ImprovementLoop(
        EvaluationSuite(CASES, OBJECTIVES, harness, evaluator),
        improve,
        identify=identify,
        observer=broken_observer,
    )
    result = await loop.run(Candidate("base", 0.6, 0.8), max_iterations=1)

    assert result.stop_reason == "target_met"
    assert loop.observer_errors == 5
