"""Read-only projections. Idle, successful proof and unsettled spending differ."""

from __future__ import annotations

from typing import Any

from .records import (
    ActionRecord,
    Budget,
    Goal,
    KernelView,
    Ref,
    RunSummary,
    StateView,
    TaskFrame,
    Usage,
)


def inspect_state(state: StateView, goal_id: str | None = None) -> KernelView:
    goals = {r.id for r in state.records if isinstance(r, Goal) and goal_id in (None, r.id)}
    tasks = tuple(r for r in state.records if isinstance(r, TaskFrame) and r.goal_id in goals)
    actions = tuple(r for r in state.records if isinstance(r, ActionRecord) and r.goal_id in goals)
    budgets = tuple(
        r
        for r in state.records
        if isinstance(r, Budget)
        and r.scope.goal_id in goals
        and (goal_id is not None or r.parent_id is None)
    )
    quantities = ("input_tokens", "output_tokens", "calls", "elapsed_seconds", "cost_microunits")
    currencies = {r.actual.currency for r in budgets}
    uncertain = any(r.accounting_uncertain or r.actual.status != "known" for r in budgets)
    usage = Usage.model_validate(
        {
            key: None
            if len(currencies) > 1 or any(getattr(r.actual, key) is None for r in budgets)
            else sum(getattr(r.actual, key) for r in budgets)
            for key in quantities
        }
        | {
            "status": "unknown" if uncertain or len(currencies) > 1 else "known",
            "currency": next(iter(currencies), "USD"),
            "source": "kernel-budget-summary",
        }
    )
    summary = RunSummary(
        completed_goal_ids=tuple(
            r.id
            for r in state.records
            if isinstance(r, Goal) and r.id in goals and r.status == "completed"
        ),
        blocked_task_ids=tuple(r.id for r in tasks if r.status == "blocked"),
        pending_review_ids=tuple(
            r.id for r in actions if r.status in {"awaiting_review", "needs_human"}
        ),
        unknown_action_ids=tuple(
            r.id
            for r in actions
            if r.status in {"dispatching", "outcome_unknown", "reconciling", "needs_human"}
        ),
        usage=usage,
    )
    # Keep the full typed proof graph available for inspection; goal_id filters
    # the summary, not authoritative references required to explain a finding.
    return KernelView(state=state, summary=summary, goal_id=goal_id)


def operator_view(view: KernelView) -> dict[str, Any]:
    """Current operator proof metadata, without context or artifact content.

    Keep reference identities explicit. Publication is not sharing and an idle
    finding is not settled spending. The typed KernelView remains authoritative.
    """
    records = view.state.records
    references = {r.id: Ref(id=r.id, revision=r.revision) for r in records}
    for record in records:
        if isinstance(record, Goal):
            for definition in (*record.criteria, *record.constraints):
                references[definition.id] = Ref(id=definition.id, revision=definition.revision)
    groups = {
        "goals": "goal",
        "tasks": "task",
        "commitments": "commitment",
        "actions": "action",
        "budgets": "budget",
        "reservations": "reservation",
        "findings": "verification",
        "approvals": "criterion_approval",
        "receipts": "receipt",
        "principals": "principal",
    }
    result: dict[str, Any] = {
        name: [r.model_dump(mode="json") for r in records if r.kind == kind]
        for name, kind in groups.items()
    }
    result.update(
        sequence=view.state.sequence,
        goal_id=view.goal_id,
        summary=view.summary.model_dump(mode="json"),
        current_refs=[ref.model_dump(mode="json") for ref in references.values()],
        artifacts=[
            r.model_dump(
                mode="json",
                include={
                    "id",
                    "revision",
                    "digest",
                    "media_type",
                    "size_bytes",
                    "shared_with",
                    "valid",
                },
            )
            for r in records
            if r.kind == "artifact"
        ],
        pending_reviews=[
            r.model_dump(mode="json") for r in records if r.id in view.summary.pending_review_ids
        ],
    )
    selected = [r for r in records if isinstance(r, Goal) and r.id == view.goal_id]
    result["goal_status"] = selected[0].status if selected else None
    return result
