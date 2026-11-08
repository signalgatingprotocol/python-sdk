"""Read-only projections. Idle, successful proof and unsettled spending differ."""

from __future__ import annotations

from .records import ActionRecord, Budget, Goal, KernelView, RunSummary, StateView, TaskFrame, Usage


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
