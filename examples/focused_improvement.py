"""Deterministic focused improvement: reject a regression, then reach target."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from signal_gating import (
    Assessment,
    EvaluationCase,
    EvaluationSuite,
    ImprovementLoop,
    Objective,
)


@dataclass(frozen=True)
class HarnessConfig:
    version: str
    reasoning: float
    reliability: float


CASES = (
    EvaluationCase("routine", "routine task"),
    EvaluationCase("adversarial", "adversarial task", weight=2.0),
)
OBJECTIVES = (
    Objective("reasoning", target=0.8, weight=2.0),
    Objective("reliability", target=0.8, regression_tolerance=0.02),
)


def harness(config: HarnessConfig, case: EvaluationCase[str]) -> dict[str, float]:
    return {
        "reasoning": config.reasoning - (0.1 if case.id == "adversarial" else 0.0),
        "reliability": config.reliability,
    }


def evaluate(case: EvaluationCase[str], output: dict[str, float]) -> Assessment:
    return Assessment(
        scores=output,
        evidence={
            dimension: f"{case.id}: observed {score:.2f}"
            for dimension, score in output.items()
        },
    )


def improve(context) -> HarnessConfig:
    current = context.incumbent
    if not context.history:
        # A tempting first intervention improves reasoning but harms reliability.
        # The acceptance policy will reject it and keep the original incumbent.
        return HarnessConfig("fast-but-fragile", current.reasoning + 0.25, 0.55)
    return replace(
        current,
        version=f"focused-{context.iteration}",
        reasoning=current.reasoning + 0.25,
        reliability=max(current.reliability, 0.85),
    )


async def main() -> None:
    suite = EvaluationSuite(CASES, OBJECTIVES, harness, evaluate)
    loop = ImprovementLoop(suite, improve, identify=lambda config: config.version)
    result = await loop.run(
        HarnessConfig("baseline", reasoning=0.45, reliability=0.75),
        max_iterations=4,
    )

    for record in result.records:
        decision = "ACCEPT" if record.accepted else "REJECT"
        print(
            f"{record.iteration}: {decision} {record.candidate_id} "
            f"focus={record.focus_dimension} progress={record.candidate_progress:.3f}"
        )
    print(f"stop={result.stop_reason} scores={dict(result.report.dimension_scores)}")


if __name__ == "__main__":
    asyncio.run(main())
