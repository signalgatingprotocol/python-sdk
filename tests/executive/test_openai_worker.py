"""No-network model boundary tests using the structural legacy client contract."""

import json
from types import SimpleNamespace

import pytest

from signal_gating.executive.adapters.openai_worker import OpenAIWorker
from signal_gating.executive.context import UTF8ByteCounter, render_packet
from signal_gating.executive.records import ContextManifest, ContextPacket, Limits
from signal_gating.executive.workers import WorkerError

from .factories import EPISODE_ID, frame, goal


def packet(**updates):
    value = ContextPacket(
        goal=goal(),
        task=frame(status="running", fencing_token=1),
        episode_id=EPISODE_ID,
        fencing_token=1,
        remaining_limits=Limits(),
        manifest=ContextManifest(
            input_tokens=1,
            output_tokens_reserved=2048,
            framing_tokens_reserved=256,
            context_token_limit=100_000,
        ),
        rendered_input='{"trusted_authority":{}}',
    )
    return value.model_copy(update=updates)


def payload(**updates):
    return {"stop_reason": "proposed", **updates}


class FakeClient:
    def __init__(self, content=None, *, usage=None, tool_calls=None, error=None):
        self.chat = SimpleNamespace(completions=self)
        self.content = json.dumps(payload()) if content is None else content
        self.usage = usage
        self.tool_calls = tool_calls
        self.error = error
        self.calls = 0
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content, tool_calls=self.tool_calls)
                )
            ],
            usage=self.usage,
        )


def worker(client, **kwargs):
    return OpenAIWorker(
        client,
        model="fixture-model",
        counter=UTF8ByteCounter(),
        capacity=100_000,
        quote=None,
        **kwargs,
    )


async def test_exact_messages_output_control_and_no_model_usage_authority():
    client = FakeClient(
        json.dumps(
            payload(
                usage={
                    "status": "known",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "elapsed_seconds": 0,
                    "cost_microunits": 0,
                }
            )
        )
    )
    value = packet()
    result = await worker(client).run_episode(value)
    assert result.stop_reason == "proposed"
    assert result.usage.status == "unknown"
    assert result.usage.calls == 1
    assert result.usage.cost_microunits is None
    assert client.last_kwargs == {
        "model": "fixture-model",
        "messages": list(render_packet(value)),
        "max_tokens": 2048,
    }


async def test_all_closed_proposal_kinds_decode_without_execution():
    proposals = [
        {"kind": "belief_change", "belief": {"claim": "hypothesis", "scope": {}}},
        {"kind": "task_decomposition", "tasks": [frame().model_dump(mode="json")]},
        {"kind": "action_intent", "capability_id": "never.execute", "arguments": {}},
        {"kind": "evidence_request", "query": "evidence"},
        {"kind": "completion_claim", "criterion_ids": ["c" * 32]},
    ]
    for proposal in proposals:
        proposal.update(episode_id=EPISODE_ID, fencing_token=1, read_set=[])
    client = FakeClient(json.dumps(payload(proposals=proposals)))
    result = await worker(client).run_episode(packet())
    assert [p.kind for p in result.proposals] == [p["kind"] for p in proposals]
    assert client.calls == 1


@pytest.mark.parametrize(
    "content",
    [
        "SECRET bad JSON",
        "[]",
        '{"stop_reason":"unknown"}',
        json.dumps(payload(extra="SECRET")),
        json.dumps(payload(stop_reason="blocked")),
        json.dumps(payload(proposals=[{"kind": "execute", "secret": "SECRET"}])),
        json.dumps(payload(artifact_drafts=[{"key": "../bad", "media_type": "x", "text": "x"}])),
    ],
)
async def test_invalid_full_dto_fails_safely_without_retry(content):
    client = FakeClient(content)
    with pytest.raises(WorkerError) as caught:
        await worker(client).run_episode(packet())
    assert caught.value.code == "invalid_worker_result"
    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.usage.status == "unknown"
    assert client.calls == 1


async def test_provider_tool_calls_rejected_without_execution():
    client = FakeClient(tool_calls=[SimpleNamespace(function=lambda: pytest.fail("executed"))])
    with pytest.raises(WorkerError, match="tools"):
        await worker(client).run_episode(packet())
    assert client.calls == 1


@pytest.mark.parametrize("name", ["model", "messages", "tools", "timeout", "stream", "max_retries"])
def test_output_parameter_cannot_override_other_controls(name):
    with pytest.raises(ValueError):
        worker(FakeClient(), output_limit_parameter=name)


async def test_configurable_output_parameter():
    client = FakeClient()
    await worker(client, output_limit_parameter="max_completion_tokens").run_episode(packet())
    assert client.last_kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in client.last_kwargs


async def test_context_recount_output_overflow_and_invalid_counter_precede_call():
    client = FakeClient()
    value = packet()
    with pytest.raises(WorkerError) as caught:
        await OpenAIWorker(
            client, model="m", counter=UTF8ByteCounter(), capacity=2050, quote=None
        ).run_episode(value)
    assert caught.value.code == "context_budget_insufficient"
    assert client.calls == 0
    value = value.model_copy(update={"remaining_limits": Limits(output_tokens=1)})
    with pytest.raises(WorkerError):
        await worker(client).run_episode(value)
    assert client.calls == 0

    class Counter:
        def count(self, text):
            return True

    with pytest.raises(WorkerError):
        await OpenAIWorker(
            client, model="m", counter=Counter(), capacity=100_000, quote=None
        ).run_episode(packet())
    assert client.calls == 0


async def test_observed_usage_preserves_partial_and_overage_on_invalid_output():
    client = FakeClient(
        "bad JSON", usage=SimpleNamespace(prompt_tokens=999_999, completion_tokens=3000)
    )
    with pytest.raises(WorkerError) as caught:
        await worker(client).run_episode(packet())
    assert caught.value.usage.input_tokens == 999_999
    assert caught.value.usage.output_tokens == 3000
    assert caught.value.usage.cost_microunits is None
    assert caught.value.usage.status == "unknown"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": -1},
        {"prompt_tokens": 7},
        {"prompt_tokens": "7", "completion_tokens": 2},
    ],
)
async def test_optional_usage_is_conservative(usage):
    result = await worker(FakeClient(usage=usage)).run_episode(packet())
    assert result.usage.status == "unknown"
    assert result.usage.input_tokens == (7 if usage == {"prompt_tokens": 7} else None)
    assert result.usage.cost_microunits is None


async def test_provider_error_sanitized_zero_retries():
    client = FakeClient(error=RuntimeError("SECRET credential"))
    with pytest.raises(WorkerError) as caught:
        await worker(client).run_episode(packet())
    assert caught.value.code == "provider_failure"
    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert client.calls == 1


async def test_large_response_without_usage_cannot_bypass_output_bound():
    client = FakeClient(json.dumps(payload(resume_brief="x" * 3000)))
    with pytest.raises(WorkerError) as caught:
        await worker(client).run_episode(packet())
    assert caught.value.code == "worker_bound_exceeded"
    assert caught.value.usage.output_tokens is None
    assert caught.value.usage.status == "unknown"


async def test_reported_output_overage_retained_on_valid_json():
    client = FakeClient(usage={"prompt_tokens": 8, "completion_tokens": 2049})
    with pytest.raises(WorkerError) as caught:
        await worker(client).run_episode(packet())
    assert caught.value.code == "worker_bound_exceeded"
    assert caught.value.usage.output_tokens == 2049


def test_client_retries_must_be_disabled_and_quote_provenance_required():
    client = FakeClient()
    client.max_retries = 2
    with pytest.raises(ValueError, match="retries"):
        worker(client)
    with pytest.raises(ValueError, match="provenance"):
        OpenAIWorker(
            FakeClient(),
            model="m",
            counter=UTF8ByteCounter(),
            capacity=100_000,
            quote=Limits(cost_microunits=1),
        )


def test_real_constructor_is_optional_zero_retry_and_credential_reference(monkeypatch):
    import sys

    seen = {}

    def client(**kwargs):
        seen.update(kwargs)
        result = FakeClient()
        result.max_retries = kwargs["max_retries"]
        return result

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=client))
    monkeypatch.setenv("FIXTURE_PROVIDER_KEY", "synthetic-key")
    value = OpenAIWorker.from_environment(
        api_key_env="FIXTURE_PROVIDER_KEY",
        model="explicit",
        counter=UTF8ByteCounter(),
        capacity=100_000,
        quote=Limits(cost_microunits=1),
        quote_version="fixture-v1",
        quote_source="operator-fixture",
    )
    assert value.model == "explicit"
    assert seen["max_retries"] == 0
    assert value.client.calls == 0
    assert value.quote_source == "operator-fixture"
    with pytest.raises(ValueError, match="unset"):
        OpenAIWorker.from_environment(
            api_key_env="UNSET_FIXTURE_KEY",
            model="explicit",
            counter=UTF8ByteCounter(),
            capacity=100_000,
            quote=Limits(cost_microunits=1),
            quote_version="fixture-v1",
            quote_source="operator-fixture",
        )


def test_reservation_helper_uses_real_policy_and_actual_subject_atomically(tmp_path):
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.messages import CommitBatch
    from signal_gating.executive.records import Budget, Commitment, Reservation, Scope
    from signal_gating.executive.store import Store

    from .factories import COMMITMENT_ID, GOAL_ID, NOW, OWNER_ID, owner
    from .test_budgets import changed
    from .test_workers import episode

    bound = Limits(
        input_tokens=10_000, output_tokens=2048, calls=1, elapsed_seconds=10, cost_microunits=100
    )
    adapter = OpenAIWorker(
        FakeClient(),
        model="fixture",
        counter=UTF8ByteCounter(),
        capacity=100_000,
        quote=bound,
        quote_version="fixture-price-v1",
        quote_source="operator",
    )
    policy = BudgetPolicy(clock=lambda: NOW)
    value = packet()
    current = episode(value, reservation_id="f" * 32)
    budget = Budget(id="d" * 32, scope=Scope(owner_id=OWNER_ID, goal_id=GOAL_ID), limits=bound)
    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    with Store.open(tmp_path / "kernel") as store:
        store.commit(
            CommitBatch(events=(changed((owner(), value.goal, commitment, value.task, budget)),))
        )
        before = store.view()
        batch = adapter.reservation_batch(policy, before, budget.id, episode=current, packet=value)
        assert store.view() == before
        assert batch.events[-1].payload["details"]["quote_version"] == "fixture-price-v1"
        # Production merges final subject authorization and context into this same event.
        records = batch.events[0].model_dump(mode="json")["payload"]["records"]
        merged = batch.events[0].model_copy(
            update={
                "payload": {
                    "records": [
                        *[dict(record) for record in records],
                        value.model_dump(mode="json"),
                        current.model_dump(mode="json"),
                    ]
                }
            }
        )
        store.commit(batch.model_copy(update={"events": (merged, *batch.events[1:])}))
        reserved = next(r for r in store.view().records if isinstance(r, Reservation))
        assert reserved.subject_id == current.id
        assert reserved.trusted_maximum_quote is True
        store.commit(policy.mark_unknown(store.view(), reserved.id))
        held = next(r for r in store.view().records if isinstance(r, Budget))
        assert held.reserved.cost_microunits == 100
        assert held.accounting_uncertain
        # Later partial observations retain overage and block reuse of a held reservation.
        from signal_gating.executive.errors import BudgetError
        from signal_gating.executive.records import Usage

        store.commit(policy.settle(store.view(), reserved.id, Usage(output_tokens=3000, calls=1)))
        with pytest.raises(BudgetError):
            adapter.reservation_batch(
                policy, store.view(), budget.id, episode=current, packet=value
            )


def test_quote_cannot_admit_longer_deadline_or_uncovered_packet(tmp_path):
    from signal_gating.executive.budgets import BudgetPolicy
    from signal_gating.executive.records import StateView

    from .factories import NOW
    from .test_workers import episode

    adapter = OpenAIWorker(
        FakeClient(),
        model="fixture",
        counter=UTF8ByteCounter(),
        capacity=100_000,
        quote=Limits(
            input_tokens=10_000, output_tokens=2048, calls=1, elapsed_seconds=1, cost_microunits=100
        ),
        quote_version="v1",
        quote_source="operator",
    )
    value = packet()
    with pytest.raises(WorkerError) as caught:
        adapter.reservation_batch(
            BudgetPolicy(clock=lambda: NOW),
            StateView(),
            "d" * 32,
            episode=episode(value, reservation_id="f" * 32),
            packet=value,
        )
    assert caught.value.code == "invalid_worker_quote"


async def test_nondefault_currency_and_explicit_per_call_limits():
    value = packet(remaining_limits=Limits(currency="EUR"))
    adapter = worker(FakeClient(usage={"prompt_tokens": 7, "completion_tokens": 9}))
    result = await adapter.run_episode(value)
    assert result.usage.currency == "EUR"
    maxima = adapter.per_call_limits(value)
    assert maxima.calls == 1
    assert maxima.output_tokens == 2048
    assert maxima.input_tokens > 256
    assert maxima.cost_microunits is None
    assert maxima.currency == "EUR"
