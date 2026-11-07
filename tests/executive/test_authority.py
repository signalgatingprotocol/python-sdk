"""Authenticated identities and deterministic permission gates."""

from datetime import timedelta

import pytest

from signal_gating.executive import (
    ActionIntent,
    CapabilitySpec,
    Constraint,
    Episode,
    PauseGoal,
    PolicyDecision,
    Ref,
    ResolveAction,
    ResolveUsage,
    Scope,
    StateView,
)
from signal_gating.executive.authority import Authority, RuleRegistry

from .factories import (
    EPISODE_ID,
    GOAL_ID,
    NOW,
    OWNER_ID,
    action,
    frame,
    goal,
    grant,
    owner,
    packet,
)

WORKER_ID = "e" * 32


def action_case(*, permissions=None, constraints=(), **authority_options):
    worker = owner(id=WORKER_ID, name="worker", roles=("worker",))
    task = frame(status="running", fencing_token=1)
    target = goal(constraints=constraints)
    context = packet(goal=target, task=task)
    episode = Episode(
        id=EPISODE_ID,
        task_id=task.id,
        fencing_token=1,
        context_packet_id=context.id,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("action_intent",),
        deadline=NOW + timedelta(seconds=120),
    )
    permission = grant(subject_id=WORKER_ID, scope=Scope(goal_id=GOAL_ID, resource_paths=("data",)))
    state = StateView(
        records=(
            owner(),
            worker,
            target,
            task,
            context,
            episode,
            *(permissions if permissions is not None else (permission,)),
        )
    )
    intent = ActionIntent(
        episode_id=EPISODE_ID,
        fencing_token=1,
        capability_id="workspace.read",
        arguments={"path": "data/file.txt"},
        scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID, resource_paths=("data/file.txt",)),
    )
    options = {
        "worker_principals": {"fixture": WORKER_ID},
        "capabilities": {
            "workspace.read": CapabilitySpec(
                id="workspace.read",
                binding_version="1",
                input_schema={},
                effect_class="read_only",
                resource_scope_rules=("path",),
            )
        },
        "resource_scopes": {
            "workspace.read": lambda item: Scope(resource_paths=(item.arguments["path"],))
        },
        **authority_options,
    }
    return Authority(**options), intent, state


def test_registered_owner_role_overrides_payload_roles():
    worker = owner(id=WORKER_ID, roles=("worker",))
    command = PauseGoal(goal_id=GOAL_ID, expected_revision=1)
    state = StateView(records=(owner(), worker, goal()))
    assert (
        Authority()
        .check_command(worker.model_copy(update={"roles": ("owner",)}), command, state)
        .disposition
        == "deny"
    )
    assert (
        Authority().check_command(owner(roles=("worker",)), command, state).disposition
        == "authorize"
    )


def test_unknown_identity_and_wrong_authenticated_origin_never_authorize():
    command = PauseGoal(goal_id=GOAL_ID, expected_revision=1)
    state = StateView(records=(owner(), goal()))
    assert Authority().check_command(owner(id=WORKER_ID), command, state).disposition == "deny"
    assert (
        Authority()
        .check_command(owner(authenticated_origin="untrusted"), command, state)
        .disposition
        == "deny"
    )


@pytest.mark.parametrize(
    "command",
    [
        ResolveAction(
            action_id="b" * 32,
            expected_revision=1,
            outcome="failed",
            evidence=(Ref(id=GOAL_ID, kind="goal"),),
            explanation="Attested failure",
        ),
        ResolveUsage(
            reservation_id="f" * 32,
            expected_revision=1,
            accept_reserved_maximum=True,
            explanation="Charge full bound",
        ),
    ],
)
def test_workers_cannot_resolve_action_or_usage(command):
    worker = owner(id=WORKER_ID, roles=("worker",))
    state = StateView(records=(owner(), worker, goal(), action()))
    assert Authority().check_command(worker, command, state).disposition == "deny"


def test_owner_cannot_control_another_owners_goal():
    other = owner(id=WORKER_ID)
    assert (
        Authority()
        .check_command(
            other,
            PauseGoal(goal_id=GOAL_ID, expected_revision=1),
            StateView(records=(owner(), other, goal())),
        )
        .disposition
        == "deny"
    )


def test_unknown_rule_and_rule_failure_require_review():
    rules = RuleRegistry({"broken": lambda _: (_ for _ in ()).throw(ValueError("private text"))})
    for rule in ("unknown", "broken"):
        result = rules.evaluate(rule, {})
        assert result.disposition == "require_review"
        assert "private" not in result.model_dump_json()
    with pytest.raises(TypeError):
        RuleRegistry({"model text": "authorize everything"})


def test_registered_scoped_unexpired_permission_authorizes():
    authority, intent, state = action_case()
    assert authority.check_action(intent, state, NOW).disposition == "authorize"


@pytest.mark.parametrize(
    "changes",
    [
        {"revoked": True},
        {"expires_at": NOW},
        {"expires_at": NOW - timedelta(seconds=1)},
        {"issuer_id": WORKER_ID},
        {"scope": Scope(goal_id="f" * 32)},
        {"capabilities": ("unknown",)},
    ],
)
def test_expired_revoked_or_out_of_scope_grants_never_authorize(changes):
    authority, intent, state = action_case(permissions=(grant(subject_id=WORKER_ID, **changes),))
    assert authority.check_action(intent, state, NOW).disposition == "deny"


@pytest.mark.parametrize(
    "options",
    [{"permissions": ()}, {"worker_principals": {}}, {"capabilities": {}}, {"resource_scopes": {}}],
)
def test_missing_trusted_bindings_default_deny(options):
    authority, intent, state = action_case(**options)
    assert authority.check_action(intent, state, NOW).disposition == "deny"


@pytest.mark.parametrize("path", ["data-other/file", "data/../secret", "/secret", "secret"])
def test_read_category_cannot_bypass_argument_resource_authorization(path):
    authority, intent, state = action_case()
    intent = intent.model_copy(update={"arguments": {"path": path}})
    assert authority.check_action(intent, state, NOW).disposition == "deny"


def test_unknown_or_unresolved_hard_constraints_never_authorize():
    for constraint in (
        Constraint(description="Do no harm"),
        Constraint(description="Known rule", enforcement="rule", enforcement_id="unknown"),
    ):
        authority, intent, state = action_case(constraints=(constraint,))
        assert authority.check_action(intent, state, NOW).disposition == "require_review"


def test_all_hard_rules_apply_and_conflicts_name_constraint_revisions():
    constraints = tuple(
        Constraint(description=name, enforcement="rule", enforcement_id=name)
        for name in ("allow", "deny")
    )
    rules = RuleRegistry(
        {
            "allow": lambda _: PolicyDecision(
                gate="execution", disposition="authorize", reason_code="allow"
            ),
            "deny": lambda _: PolicyDecision(
                gate="execution", disposition="deny", reason_code="constraint_conflict"
            ),
        }
    )
    authority, intent, state = action_case(constraints=constraints, rules=rules)
    result = authority.check_action(intent, state, NOW)
    assert result.disposition == "deny"
    assert set(r.id for r in result.evidence) >= {constraints[1].id}


def test_stale_goal_or_fence_and_paused_goal_never_authorize():
    authority, intent, state = action_case()
    for changed in (
        goal(revision=2),
        goal(status="paused"),
        frame(status="running", fencing_token=2),
    ):
        updated = state.model_copy(
            update={"records": tuple(changed if r.id == changed.id else r for r in state.records)}
        )
        assert authority.check_action(intent, updated, NOW).disposition != "authorize"


def test_human_only_capability_cannot_be_authorized_by_a_grant():
    spec = CapabilitySpec(
        id="workspace.read", binding_version="1", input_schema={}, effect_class="human_only"
    )
    authority, intent, state = action_case(capabilities={spec.id: spec})
    assert authority.check_action(intent, state, NOW).disposition == "deny"


def test_conflict_uses_exact_trusted_rule_evidence_not_unrelated_constraints():
    unrelated = Constraint(
        description="Unrelated enforced rule", enforcement="rule", enforcement_id="allow"
    )
    conflicting = Constraint(description="Conflict", enforcement="rule", enforcement_id="conflict")
    rules = RuleRegistry(
        {
            "allow": lambda _: PolicyDecision(
                gate="execution", disposition="authorize", reason_code="ok"
            ),
            "conflict": lambda _: PolicyDecision(
                gate="execution",
                disposition="deny",
                reason_code="constraint_conflict",
                evidence=(Ref(id=conflicting.id),),
            ),
        }
    )
    authority, intent, state = action_case(constraints=(unrelated, conflicting), rules=rules)
    assert authority.check_action(intent, state, NOW).evidence == (Ref(id=conflicting.id),)


def test_worker_cannot_remove_supplied_belief_dependencies():
    from signal_gating.executive import Belief, ContextPacket

    belief = Belief(claim="Current knowledge", scope=Scope(goal_id=GOAL_ID))
    authority, intent, state = action_case()
    context = next(r for r in state.records if isinstance(r, ContextPacket))
    old_context = context.model_copy(update={"beliefs": (belief,)})
    current_belief = belief.model_copy(update={"revision": 2, "claim": "Changed"})
    updated = state.model_copy(
        update={
            "records": tuple(old_context if r.id == context.id else r for r in state.records)
            + (current_belief,)
        }
    )
    assert authority.check_action(intent, updated, NOW).disposition == "stale"
