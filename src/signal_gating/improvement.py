"""Evidence-driven evaluation and focused improvement for arbitrary AI systems.

The candidate is caller-owned: it may be a prompt, model ID, tool policy, mesh,
or compound harness configuration. SGP evaluates and compares candidates; it
does not assume how a candidate is represented or improved.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import statistics
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Generic, TypeVar, cast
from uuid import uuid4

from signal_gating.errors import EvaluationError, SignalSerializationError
from signal_gating.signal import Signal
from signal_gating.trajectory import _digest

CandidateT = TypeVar("CandidateT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class EvaluationCase(Generic[InputT]):
    """One stable, weighted input in an evaluation suite."""

    id: str
    input: InputT
    weight: float = 1.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evaluation case id must not be empty")
        weight = _finite("evaluation case weight", self.weight)
        if weight <= 0:
            raise ValueError("evaluation case weight must be > 0")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class Objective:
    """A named capability target and its regression budget."""

    name: str
    target: float
    weight: float = 1.0
    regression_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("objective name must not be empty")
        target = _finite("objective target", self.target)
        weight = _finite("objective weight", self.weight)
        tolerance = _finite("objective regression_tolerance", self.regression_tolerance)
        if target <= 0:
            raise ValueError("objective target must be > 0")
        if weight <= 0:
            raise ValueError("objective weight must be > 0")
        if tolerance < 0:
            raise ValueError("objective regression_tolerance must be >= 0")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "regression_tolerance", tolerance)


@dataclass(frozen=True)
class Assessment:
    """Evaluator scores and concrete evidence for one output sample."""

    scores: Mapping[str, float]
    evidence: Mapping[str, str]

    def __post_init__(self) -> None:
        scores = {name: _finite(f"score {name!r}", value) for name, value in self.scores.items()}
        evidence = {str(name): str(value) for name, value in self.evidence.items()}
        object.__setattr__(self, "scores", MappingProxyType(scores))
        object.__setattr__(self, "evidence", MappingProxyType(evidence))


@dataclass(frozen=True)
class CaseResult(Generic[OutputT]):
    """Median scores and retained evidence/outputs for one evaluation case."""

    case_id: str
    weight: float
    scores: Mapping[str, float]
    evidence: Mapping[str, tuple[str, ...]]
    outputs: tuple[OutputT, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType({name: tuple(items) for name, items in self.evidence.items()}),
        )
        object.__setattr__(self, "outputs", tuple(self.outputs))


@dataclass(frozen=True)
class Focus:
    """The highest-value weak dimension and its weakest cases."""

    dimension: str
    score: float
    target: float
    gap: float
    case_ids: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport(Generic[OutputT]):
    candidate_id: str
    objectives: tuple[Objective, ...]
    cases: tuple[CaseResult[OutputT], ...]
    dimension_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(
            self, "dimension_scores", MappingProxyType(dict(self.dimension_scores))
        )

    @property
    def target_met(self) -> bool:
        return all(self.dimension_scores[item.name] >= item.target for item in self.objectives)

    @property
    def progress(self) -> float:
        total_weight = sum(item.weight for item in self.objectives)
        return sum(
            item.weight
            * min(1.0, max(0.0, self.dimension_scores[item.name] / item.target))
            for item in self.objectives
        ) / total_weight

    def focus(self) -> Focus:
        if self.target_met:
            raise EvaluationError("all objectives already meet target")
        objective = max(
            self.objectives,
            key=lambda item: item.weight
            * max(0.0, 1.0 - self.dimension_scores[item.name] / item.target),
        )
        ordered_cases = sorted(
            self.cases,
            key=lambda result: (
                result.scores[objective.name] / objective.target,
                -result.weight,
                result.case_id,
            ),
        )
        evidence = tuple(
            item
            for result in ordered_cases
            for item in result.evidence[objective.name]
        )
        score = self.dimension_scores[objective.name]
        return Focus(
            dimension=objective.name,
            score=score,
            target=objective.target,
            gap=max(0.0, objective.target - score),
            case_ids=tuple(result.case_id for result in ordered_cases),
            evidence=evidence,
        )


Harness = Callable[
    [CandidateT, EvaluationCase[InputT]], OutputT | Awaitable[OutputT]
]
Evaluator = Callable[
    [EvaluationCase[InputT], OutputT], Assessment | Awaitable[Assessment]
]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class EvaluationSuite(Generic[CandidateT, InputT, OutputT]):
    """Evaluate any candidate on stable cases with bounded parallelism."""

    def __init__(
        self,
        cases: Sequence[EvaluationCase[InputT]],
        objectives: Sequence[Objective],
        harness: Harness[CandidateT, InputT, OutputT],
        evaluator: Evaluator[InputT, OutputT],
        *,
        samples: int = 1,
        max_concurrency: int = 1,
        retain_outputs: bool = False,
    ) -> None:
        self.cases = tuple(cases)
        self.objectives = tuple(objectives)
        if not self.cases:
            raise ValueError("evaluation suite requires at least one case")
        if not self.objectives:
            raise ValueError("evaluation suite requires at least one objective")
        case_ids = [case.id for case in self.cases]
        objective_names = [objective.name for objective in self.objectives]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case ids must be unique")
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("objective names must be unique")
        if samples <= 0:
            raise ValueError("samples must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        self.samples = samples
        self.max_concurrency = max_concurrency
        self.retain_outputs = retain_outputs
        self._harness = harness
        self._evaluator = evaluator

    async def evaluate(
        self, candidate: CandidateT, *, candidate_id: str
    ) -> EvaluationReport[OutputT]:
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        total_jobs = len(self.cases) * self.samples
        assessments: list[list[Assessment | None]] = [
            [None] * self.samples for _ in self.cases
        ]
        outputs: list[list[OutputT | None]] | None = (
            [[None] * self.samples for _ in self.cases]
            if self.retain_outputs
            else None
        )
        next_job = 0

        async def worker() -> None:
            nonlocal next_job
            while next_job < total_jobs:
                job = next_job
                next_job += 1
                case_index, sample_index = divmod(job, self.samples)
                case = self.cases[case_index]
                try:
                    output = cast(OutputT, await _resolve(self._harness(candidate, case)))
                    assessment = await _resolve(self._evaluator(case, output))
                except Exception as error:
                    raise EvaluationError(
                        f"evaluation failed for case {case.id!r} sample {sample_index}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if not isinstance(assessment, Assessment):
                    raise EvaluationError(
                        f"evaluator for case {case.id!r} must return Assessment"
                    )
                self._validate_assessment(case, assessment)
                assessments[case_index][sample_index] = assessment
                if outputs is not None:
                    outputs[case_index][sample_index] = output

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self.max_concurrency, total_jobs))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        names = tuple(item.name for item in self.objectives)
        results: list[CaseResult[OutputT]] = []
        for case_index, case in enumerate(self.cases):
            group = cast(list[Assessment], assessments[case_index])
            retained_outputs = (
                tuple(cast(list[OutputT], outputs[case_index]))
                if outputs is not None
                else ()
            )
            results.append(
                CaseResult(
                    case_id=case.id,
                    weight=case.weight,
                    scores={
                        name: float(statistics.median(item.scores[name] for item in group))
                        for name in names
                    },
                    evidence={
                        name: tuple(item.evidence[name] for item in group)
                        for name in names
                    },
                    outputs=retained_outputs,
                )
            )
        total_case_weight = sum(case.weight for case in self.cases)
        dimension_scores = {
            name: sum(result.weight * result.scores[name] for result in results)
            / total_case_weight
            for name in names
        }
        return EvaluationReport(
            candidate_id=candidate_id,
            objectives=self.objectives,
            cases=tuple(results),
            dimension_scores=dimension_scores,
        )

    def _validate_assessment(
        self, case: EvaluationCase[InputT], assessment: Assessment
    ) -> None:
        expected = {item.name for item in self.objectives}
        if set(assessment.scores) != expected or set(assessment.evidence) != expected:
            raise EvaluationError(
                f"case {case.id!r} assessment must contain the exact objective set "
                f"{sorted(expected)!r} in both scores and evidence"
            )
        empty = [name for name, value in assessment.evidence.items() if not value.strip()]
        if empty:
            raise EvaluationError(
                f"case {case.id!r} has empty evidence for objectives {sorted(empty)!r}"
            )


@dataclass(frozen=True)
class Regression:
    scope: str
    dimension: str
    case_id: str
    baseline: float
    candidate: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dimension": self.dimension,
            "case_id": self.case_id,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Regression:
        return cls(
            scope=str(value["scope"]),
            dimension=str(value["dimension"]),
            case_id=str(value["case_id"]),
            baseline=float(value["baseline"]),
            candidate=float(value["candidate"]),
            tolerance=float(value["tolerance"]),
        )


@dataclass(frozen=True)
class Comparison:
    accepted: bool
    progress_delta: float
    focus_delta: float
    regressions: tuple[Regression, ...]


@dataclass(frozen=True)
class AcceptancePolicy:
    """Focused hill-climbing policy with aggregate and per-case guardrails."""

    min_progress_delta: float = 1e-9
    min_focus_delta: float = 1e-9

    def __post_init__(self) -> None:
        progress = _finite("min_progress_delta", self.min_progress_delta)
        focus = _finite("min_focus_delta", self.min_focus_delta)
        if progress < 0 or focus < 0:
            raise ValueError("minimum deltas must be >= 0")
        object.__setattr__(self, "min_progress_delta", progress)
        object.__setattr__(self, "min_focus_delta", focus)

    def compare(
        self,
        baseline: EvaluationReport[Any],
        candidate: EvaluationReport[Any],
        focus: Focus,
    ) -> Comparison:
        if baseline.objectives != candidate.objectives:
            raise EvaluationError("cannot compare reports with different objective contracts")
        baseline_cases = {item.case_id: item for item in baseline.cases}
        candidate_cases = {item.case_id: item for item in candidate.cases}
        if set(baseline_cases) != set(candidate_cases):
            raise EvaluationError("cannot compare reports with different cases")
        if any(
            baseline_cases[case_id].weight != candidate_cases[case_id].weight
            for case_id in baseline_cases
        ):
            raise EvaluationError("cannot compare reports with different case weights")

        regressions: list[Regression] = []
        for objective in baseline.objectives:
            old = baseline.dimension_scores[objective.name]
            new = candidate.dimension_scores[objective.name]
            if old - new > objective.regression_tolerance + 1e-12:
                regressions.append(
                    Regression(
                        "dimension",
                        objective.name,
                        "",
                        old,
                        new,
                        objective.regression_tolerance,
                    )
                )
            for case_id, old_case in baseline_cases.items():
                old = old_case.scores[objective.name]
                new = candidate_cases[case_id].scores[objective.name]
                if old - new > objective.regression_tolerance + 1e-12:
                    regressions.append(
                        Regression(
                            "case",
                            objective.name,
                            case_id,
                            old,
                            new,
                            objective.regression_tolerance,
                        )
                    )
        progress_delta = candidate.progress - baseline.progress
        focus_delta = (
            candidate.dimension_scores[focus.dimension]
            - baseline.dimension_scores[focus.dimension]
        )
        accepted = (
            progress_delta + 1e-12 >= self.min_progress_delta
            and focus_delta + 1e-12 >= self.min_focus_delta
            and not regressions
        )
        return Comparison(accepted, progress_delta, focus_delta, tuple(regressions))


@dataclass(frozen=True)
class ImprovementRecord:
    iteration: int
    incumbent_id: str
    candidate_id: str
    focus_dimension: str
    focus_case_ids: tuple[str, ...]
    focus_evidence: tuple[str, ...]
    baseline_scores: Mapping[str, float]
    candidate_scores: Mapping[str, float]
    candidate_evidence: Mapping[str, Mapping[str, tuple[str, ...]]]
    baseline_progress: float
    candidate_progress: float
    progress_delta: float
    focus_delta: float
    accepted: bool
    regressions: tuple[Regression, ...]
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        created_at = _finite("improvement record created_at", self.created_at)
        if created_at <= 0:
            raise ValueError("improvement record created_at must be > 0")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "focus_case_ids", tuple(self.focus_case_ids))
        object.__setattr__(self, "focus_evidence", tuple(self.focus_evidence))
        object.__setattr__(
            self, "baseline_scores", MappingProxyType(dict(self.baseline_scores))
        )
        object.__setattr__(
            self, "candidate_scores", MappingProxyType(dict(self.candidate_scores))
        )
        object.__setattr__(
            self,
            "candidate_evidence",
            MappingProxyType(
                {
                    case_id: MappingProxyType(
                        {
                            dimension: tuple(evidence)
                            for dimension, evidence in dimensions.items()
                        }
                    )
                    for case_id, dimensions in self.candidate_evidence.items()
                }
            ),
        )
        object.__setattr__(self, "regressions", tuple(self.regressions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "incumbent_id": self.incumbent_id,
            "candidate_id": self.candidate_id,
            "focus_dimension": self.focus_dimension,
            "focus_case_ids": list(self.focus_case_ids),
            "focus_evidence": list(self.focus_evidence),
            "baseline_scores": dict(self.baseline_scores),
            "candidate_scores": dict(self.candidate_scores),
            "candidate_evidence": {
                case_id: {
                    dimension: list(evidence)
                    for dimension, evidence in dimensions.items()
                }
                for case_id, dimensions in self.candidate_evidence.items()
            },
            "baseline_progress": self.baseline_progress,
            "candidate_progress": self.candidate_progress,
            "progress_delta": self.progress_delta,
            "focus_delta": self.focus_delta,
            "accepted": self.accepted,
            "regressions": [item.to_dict() for item in self.regressions],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ImprovementRecord:
        return cls(
            iteration=int(value["iteration"]),
            incumbent_id=str(value["incumbent_id"]),
            candidate_id=str(value["candidate_id"]),
            focus_dimension=str(value["focus_dimension"]),
            focus_case_ids=tuple(str(item) for item in value["focus_case_ids"]),
            focus_evidence=tuple(str(item) for item in value["focus_evidence"]),
            baseline_scores={
                str(name): float(score)
                for name, score in cast(Mapping[str, Any], value["baseline_scores"]).items()
            },
            candidate_scores={
                str(name): float(score)
                for name, score in cast(Mapping[str, Any], value["candidate_scores"]).items()
            },
            candidate_evidence={
                str(case_id): {
                    str(dimension): tuple(str(item) for item in evidence)
                    for dimension, evidence in dimensions.items()
                }
                for case_id, dimensions in cast(
                    Mapping[str, Mapping[str, Sequence[Any]]],
                    value["candidate_evidence"],
                ).items()
            },
            baseline_progress=float(value["baseline_progress"]),
            candidate_progress=float(value["candidate_progress"]),
            progress_delta=float(value["progress_delta"]),
            focus_delta=float(value["focus_delta"]),
            accepted=bool(value["accepted"]),
            regressions=tuple(
                Regression.from_dict(cast(Mapping[str, Any], item))
                for item in cast(Sequence[Any], value["regressions"])
            ),
            # Legacy records predate retention timestamps. Treat migration time
            # as creation time so enabling retention never deletes them blindly.
            created_at=float(value.get("created_at", time.time())),
        )


_HISTORY_GENESIS = "sgp-improvement-history-genesis"


@dataclass(frozen=True)
class RetentionPolicy:
    """Hard storage bounds for durable improvement history.

    Defaults retain at most 30 days, 500 experiments, and 8 MiB. Set a field to
    ``None`` to disable only that bound, or pass ``retention=None`` to
    ``ImprovementHistory`` to disable automatic compaction entirely.
    """

    max_age_seconds: float | None = 30 * 24 * 60 * 60
    max_records: int | None = 500
    max_bytes: int | None = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_age_seconds is not None:
            age = _finite("retention max_age_seconds", self.max_age_seconds)
            if age <= 0:
                raise ValueError("retention max_age_seconds must be > 0")
            object.__setattr__(self, "max_age_seconds", age)
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("retention max_records must be > 0")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("retention max_bytes must be > 0")


def _encode_history(
    records: Sequence[ImprovementRecord],
) -> tuple[str, str]:
    previous = _HISTORY_GENESIS
    lines: list[str] = []
    for sequence, record in enumerate(records):
        core = {"seq": sequence, "prev": previous, "record": record.to_dict()}
        previous = _digest(core)
        lines.append(json.dumps({**core, "digest": previous}, sort_keys=True) + "\n")
    return "".join(lines), previous


class ImprovementHistory:
    """Bounded, hash-chained experiment decisions with optional JSONL storage."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        retention: RetentionPolicy | None = RetentionPolicy(),
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._retention = retention
        self._records: list[ImprovementRecord] = []
        self._head_digest = _HISTORY_GENESIS
        if self._path is not None and self._path.exists():
            self._load()
        if self._needs_prune(time.time()):
            self.prune()

    @property
    def records(self) -> tuple[ImprovementRecord, ...]:
        return tuple(self._records)

    @property
    def head_digest(self) -> str:
        return self._head_digest

    def append(self, record: ImprovementRecord) -> None:
        core = {
            "seq": len(self._records),
            "prev": self._head_digest,
            "record": record.to_dict(),
        }
        digest = _digest(core)
        envelope = {**core, "digest": digest}
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(envelope, sort_keys=True) + "\n")
            os.chmod(self._path, 0o600)
        self._records.append(record)
        self._head_digest = digest
        if self._needs_prune(time.time()):
            self.prune()

    def prune(self, *, now: float | None = None) -> int:
        """Drop stale/overflow records and atomically rebuild the digest chain.

        Returns the number of removed records. Recent records always win when a
        count or byte cap is reached.
        """
        if self._retention is None:
            return 0
        current_time = time.time() if now is None else _finite("prune now", now)
        original = list(self._records)
        retained = original
        policy = self._retention
        if policy.max_age_seconds is not None:
            cutoff = current_time - policy.max_age_seconds
            retained = [record for record in retained if record.created_at >= cutoff]
        if policy.max_records is not None and len(retained) > policy.max_records:
            retained = retained[-policy.max_records :]
        if policy.max_bytes is not None and self._path is not None:
            encoded, _ = _encode_history(retained)
            if len(encoded.encode("utf-8")) > policy.max_bytes:
                low, high = 0, len(retained)
                while low < high:
                    middle = (low + high) // 2
                    candidate, _ = _encode_history(retained[middle:])
                    if len(candidate.encode("utf-8")) <= policy.max_bytes:
                        high = middle
                    else:
                        low = middle + 1
                retained = retained[low:]

        removed = len(original) - len(retained)
        if removed == 0:
            return 0
        self._records = retained
        payload, head = _encode_history(retained)
        if self._path is not None:
            temporary = self._path.with_name(
                f".{self._path.name}.{uuid4().hex}.tmp"
            )
            try:
                with open(temporary, "x", encoding="utf-8") as handle:
                    os.chmod(temporary, 0o600)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        self._head_digest = head
        return removed

    def _needs_prune(self, now: float) -> bool:
        policy = self._retention
        if policy is None:
            return False
        if policy.max_records is not None and len(self._records) > policy.max_records:
            return True
        if policy.max_age_seconds is not None:
            cutoff = now - policy.max_age_seconds
            if any(record.created_at < cutoff for record in self._records):
                return True
        return bool(
            policy.max_bytes is not None
            and self._path is not None
            and self._path.exists()
            and self._path.stat().st_size > policy.max_bytes
        )

    def _load(self) -> None:
        assert self._path is not None
        previous = _HISTORY_GENESIS
        try:
            with open(self._path, encoding="utf-8") as handle:
                for sequence, line in enumerate(handle):
                    envelope = json.loads(line)
                    core = {
                        "seq": envelope["seq"],
                        "prev": envelope["prev"],
                        "record": envelope["record"],
                    }
                    if (
                        envelope["seq"] != sequence
                        or envelope["prev"] != previous
                        or envelope["digest"] != _digest(core)
                    ):
                        raise SignalSerializationError(
                            f"improvement history chain broken at seq {sequence}"
                        )
                    self._records.append(ImprovementRecord.from_dict(envelope["record"]))
                    previous = str(envelope["digest"])
        except SignalSerializationError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SignalSerializationError(
                f"improvement history chain broken at seq {len(self._records)}"
            ) from error
        self._head_digest = previous


class ImprovementEvent(Signal):
    """Observable lifecycle signal for a focused-improvement run."""

    __signal_type__: ClassVar[str] = "sgp.improvement.event"
    action: str = ""
    iteration: int = 0
    candidate_id: str = ""
    focus_dimension: str = ""
    progress: float = 0.0
    accepted: bool | None = None


@dataclass(frozen=True)
class ImprovementContext(Generic[CandidateT, OutputT]):
    iteration: int
    incumbent: CandidateT
    report: EvaluationReport[OutputT]
    focus: Focus
    history: tuple[ImprovementRecord, ...]


@dataclass(frozen=True)
class ImprovementResult(Generic[CandidateT, OutputT]):
    incumbent: CandidateT
    report: EvaluationReport[OutputT]
    records: tuple[ImprovementRecord, ...]
    stop_reason: str


Improver = Callable[
    [ImprovementContext[CandidateT, OutputT]],
    CandidateT | None | Awaitable[CandidateT | None],
]
Observer = Callable[[ImprovementEvent], None | Awaitable[None]]
Identifier = Callable[[CandidateT], str]


class ImprovementLoop(Generic[CandidateT, InputT, OutputT]):
    """Evaluate, focus, propose, guard, retain, and repeat."""

    def __init__(
        self,
        suite: EvaluationSuite[CandidateT, InputT, OutputT],
        improve: Improver[CandidateT, OutputT],
        *,
        identify: Identifier[CandidateT],
        policy: AcceptancePolicy | None = None,
        history: ImprovementHistory | None = None,
        observer: Observer | None = None,
    ) -> None:
        self._suite = suite
        self._improve = improve
        self._identify = identify
        self._policy = policy if policy is not None else AcceptancePolicy()
        self._history = history if history is not None else ImprovementHistory()
        self._observer = observer
        self.observer_errors = 0

    async def run(
        self, initial: CandidateT, *, max_iterations: int = 10
    ) -> ImprovementResult[CandidateT, OutputT]:
        if max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        trace_id = uuid4().hex
        prior_history = self._history.records
        run_records: list[ImprovementRecord] = []
        incumbent = initial
        incumbent_id = self._identify(incumbent)
        report = await self._suite.evaluate(incumbent, candidate_id=incumbent_id)
        await self._emit(
            trace_id,
            "baseline_evaluated",
            0,
            incumbent_id,
            progress=report.progress,
        )
        if report.target_met:
            await self._emit(
                trace_id, "target_met", 0, incumbent_id, progress=report.progress
            )
            return self._result(incumbent, report, run_records, "target_met")

        for iteration in range(1, max_iterations + 1):
            focus = report.focus()
            await self._emit(
                trace_id,
                "focus_selected",
                iteration,
                incumbent_id,
                focus_dimension=focus.dimension,
                progress=report.progress,
            )
            context = ImprovementContext(
                iteration=iteration,
                incumbent=incumbent,
                report=report,
                focus=focus,
                # Disk retention may compact older attempts while this run is
                # active. Keep the run snapshot and all current attempts in
                # memory so focused improvement never forgets mid-run.
                history=prior_history + tuple(run_records),
            )
            candidate = cast(CandidateT | None, await _resolve(self._improve(context)))
            if candidate is None:
                await self._emit(
                    trace_id,
                    "improver_stopped",
                    iteration,
                    incumbent_id,
                    focus_dimension=focus.dimension,
                    progress=report.progress,
                )
                return self._result(
                    incumbent, report, run_records, "improver_stopped"
                )
            candidate_id = self._identify(candidate)
            candidate_report = await self._suite.evaluate(
                candidate, candidate_id=candidate_id
            )
            comparison = self._policy.compare(report, candidate_report, focus)
            record = ImprovementRecord(
                iteration=iteration,
                incumbent_id=incumbent_id,
                candidate_id=candidate_id,
                focus_dimension=focus.dimension,
                focus_case_ids=focus.case_ids,
                focus_evidence=focus.evidence,
                baseline_scores=report.dimension_scores,
                candidate_scores=candidate_report.dimension_scores,
                candidate_evidence={
                    case.case_id: case.evidence for case in candidate_report.cases
                },
                baseline_progress=report.progress,
                candidate_progress=candidate_report.progress,
                progress_delta=comparison.progress_delta,
                focus_delta=comparison.focus_delta,
                accepted=comparison.accepted,
                regressions=comparison.regressions,
            )
            self._history.append(record)
            run_records.append(record)
            await self._emit(
                trace_id,
                "candidate_evaluated",
                iteration,
                candidate_id,
                focus_dimension=focus.dimension,
                progress=candidate_report.progress,
                accepted=comparison.accepted,
            )
            await self._emit(
                trace_id,
                "candidate_accepted" if comparison.accepted else "candidate_rejected",
                iteration,
                candidate_id,
                focus_dimension=focus.dimension,
                progress=candidate_report.progress,
                accepted=comparison.accepted,
            )
            if comparison.accepted:
                incumbent = candidate
                incumbent_id = candidate_id
                report = candidate_report
                if report.target_met:
                    await self._emit(
                        trace_id,
                        "target_met",
                        iteration,
                        incumbent_id,
                        progress=report.progress,
                    )
                    return self._result(
                        incumbent, report, run_records, "target_met"
                    )

        await self._emit(
            trace_id,
            "budget_exhausted",
            max_iterations,
            incumbent_id,
            progress=report.progress,
        )
        return self._result(incumbent, report, run_records, "budget_exhausted")

    def _result(
        self,
        incumbent: CandidateT,
        report: EvaluationReport[OutputT],
        records: Sequence[ImprovementRecord],
        reason: str,
    ) -> ImprovementResult[CandidateT, OutputT]:
        return ImprovementResult(
            incumbent=incumbent,
            report=report,
            records=tuple(records),
            stop_reason=reason,
        )

    async def _emit(
        self,
        trace_id: str,
        action: str,
        iteration: int,
        candidate_id: str,
        *,
        focus_dimension: str = "",
        progress: float,
        accepted: bool | None = None,
    ) -> None:
        if self._observer is None:
            return
        event = ImprovementEvent(
            action=action,
            iteration=iteration,
            candidate_id=candidate_id,
            focus_dimension=focus_dimension,
            progress=progress,
            accepted=accepted,
            source="improvement-loop",
            trace_id=trace_id,
        )
        try:
            await _resolve(self._observer(event))
        except Exception:
            self.observer_errors += 1
