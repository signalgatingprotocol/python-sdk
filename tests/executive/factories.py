"""Deterministic, valid executive records for component tests."""

from datetime import datetime, timezone
from typing import Any

from signal_gating.executive import (
    ActionRecord,
    Belief,
    BeliefChange,
    ContextManifest,
    ContextPacket,
    Criterion,
    ExecutiveSignal,
    Goal,
    Grant,
    Limits,
    Principal,
    Scope,
    TaskFrame,
)

NOW = datetime(2026, 11, 7, tzinfo=timezone.utc)
OWNER_ID = "1" * 32
GOAL_ID = "2" * 32
TASK_ID = "3" * 32
COMMITMENT_ID = "4" * 32
EPISODE_ID = "5" * 32


def owner(**overrides: Any) -> Principal:
    return Principal(
        **{
            "id": OWNER_ID,
            "created_at": NOW,
            "name": "Owner",
            "trust_domain": "local",
            "authenticated_origin": "local-bootstrap",
            "roles": ("owner",),
            **overrides,
        }
    )


def criterion(**overrides: Any) -> Criterion:
    return Criterion(
        **{
            "id": "6" * 32,
            "created_at": NOW,
            "description": "Artifact matches expected text",
            "verifier_id": "text-equals-v1",
            "acceptance_rule": "exact match",
            **overrides,
        }
    )


def goal(**overrides: Any) -> Goal:
    return Goal(
        **{
            "id": GOAL_ID,
            "created_at": NOW,
            "owner_id": OWNER_ID,
            "outcome": "Produce a checked artifact",
            "worker_id": "fixture",
            "limits": Limits(calls=4, input_tokens=8192, output_tokens=2048, cost_microunits=0),
            "criteria": (criterion(),),
            **overrides,
        }
    )


def frame(**overrides: Any) -> TaskFrame:
    return TaskFrame(
        **{
            "id": TASK_ID,
            "created_at": NOW,
            "goal_id": GOAL_ID,
            "commitment_id": COMMITMENT_ID,
            "subgoal": "Create the artifact",
            **overrides,
        }
    )


def grant(**overrides: Any) -> Grant:
    return Grant(
        **{
            "id": "7" * 32,
            "created_at": NOW,
            "issuer_id": OWNER_ID,
            "subject_id": OWNER_ID,
            "capabilities": ("workspace.read",),
            "scope": Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
            **overrides,
        }
    )


def signal(**overrides: Any) -> ExecutiveSignal:
    return ExecutiveSignal(
        **{
            "id": "8" * 32,
            "created_at": NOW,
            "received_at": NOW,
            "signal_kind": "evidence",
            "principal_id": OWNER_ID,
            "producer_id": "fixture",
            "scope": Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
            **overrides,
        }
    )


def proposal(**overrides: Any) -> BeliefChange:
    return BeliefChange(
        **{
            "id": "9" * 32,
            "created_at": NOW,
            "episode_id": EPISODE_ID,
            "fencing_token": 1,
            "belief": Belief(
                id="a" * 32,
                created_at=NOW,
                claim="The artifact is drafted",
                scope=Scope(goal_id=GOAL_ID),
            ),
            **overrides,
        }
    )


def action(**overrides: Any) -> ActionRecord:
    return ActionRecord(
        **{
            "id": "b" * 32,
            "created_at": NOW,
            "goal_id": GOAL_ID,
            "task_id": TASK_ID,
            "capability_id": "workspace.read",
            "binding_version": "1",
            "arguments_digest": "a" * 64,
            "scope": Scope(owner_id=OWNER_ID, goal_id=GOAL_ID),
            "idempotency_key": "action-b",
            **overrides,
        }
    )


def packet(**overrides: Any) -> ContextPacket:
    return ContextPacket(
        **{
            "id": "c" * 32,
            "created_at": NOW,
            "goal": goal(),
            "task": frame(),
            "episode_id": EPISODE_ID,
            "fencing_token": 1,
            "remaining_limits": goal().limits,
            "manifest": ContextManifest(),
            "rendered_input": "Produce a checked artifact.",
            **overrides,
        }
    )
