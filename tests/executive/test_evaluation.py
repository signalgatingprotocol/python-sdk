"""Fresh paired comparisons through the real local executive boundary."""

import json
from pathlib import Path

import pytest

from signal_gating.executive.evaluation import (
    ARMS,
    FixtureWorker,
    compare_reports,
    evaluate_cases,
    load_cases,
    load_protocol,
    main,
)
from signal_gating.executive.messages import EpisodeResult
from signal_gating.executive.records import EvaluationReport, Usage, WakeCondition
from signal_gating.executive.workers import WorkerError

EXAMPLES = Path(__file__).parents[2] / "examples/executive/evaluation"


def cases():
    return load_cases(EXAMPLES / "cases.jsonl")


async def test_equal_configuration_isolation_fresh_workers_and_real_receipts():
    workers = []

    def factory():
        worker = FixtureWorker()
        workers.append(worker)
        return worker

    reports = []
    for arm in ARMS:
        report = await evaluate_cases(cases()[:1], arm=arm, trials=3, worker_factory=factory)
        assert report.attempts == 3
        assert (
            report.total_calls == report.worker_calls + report.control_calls + report.verifier_calls
        )
        assert all(r.receipts and r.summary.completed_goal_ids for r in report.results)
        assert len({r.store_path for r in report.results}) == 3
        assert all(
            r.metrics["worker_calls"] > 0 and r.metrics["verifier_calls"] > 0
            for r in report.results
        )
        reports.append(report)
    assert len(workers) == 15
    assert len({r.configuration_digest for r in reports}) == 1
    assert len({r.manifest_digest for r in reports}) == 1
    assert all(worker.packets for worker in workers)
    assert len({r.store_path for report in reports for r in report.results}) == 15


async def test_missing_usage_and_failed_attempts_retained():
    class Broken:
        async def run_episode(self, packet):
            raise WorkerError("fixture_failure", "safe", usage=Usage(calls=1))

    report = await evaluate_cases(cases()[:1], arm="executive", trials=3, worker_factory=Broken)
    assert report.attempts == 3
    assert report.cost_status == "unknown"
    assert all(not r.passed and r.failures for r in report.results)
    assert report.worker_calls >= 3
    assert all(r.measurements["cost_microunits"] is None for r in report.results)
    assert all(r.usage_observations[0].cost_microunits is None for r in report.results)
    assert all(r.metrics["uncertain_reservations"] > 0 for r in report.results)


async def test_supplied_worker_controls_outcome_not_scripted_winner():
    class Blocked:
        async def run_episode(self, packet):
            return EpisodeResult(
                stop_reason="blocked",
                blocker=WakeCondition(kind="human_decision", reason_code="fixture_blocked"),
                usage=Usage.zero().model_copy(update={"calls": 1}),
            )

    report = await evaluate_cases(cases()[:1], arm="executive", trials=1, worker_factory=Blocked)
    assert report.attempts == 1
    assert not report.results[0].completed
    assert not report.results[0].passed


async def test_empty_suite_and_invalid_inputs():
    report = await evaluate_cases([], arm="executive", trials=3, worker_factory=FixtureWorker)
    assert report.accepted is False
    assert report.attempts == 0
    assert compare_reports([report])["benefit_claim_allowed"] is False
    with pytest.raises(ValueError):
        await evaluate_cases(cases()[:1], arm="bad", trials=1, worker_factory=FixtureWorker)
    with pytest.raises(ValueError):
        await evaluate_cases(cases()[:1], arm="notes", trials=0, worker_factory=FixtureWorker)


def test_frozen_split_and_protocol():
    suite = cases()
    protocol = load_protocol(EXAMPLES / "protocol.json", suite)
    assert len(protocol.tuning_case_ids) == 6
    assert len(protocol.heldout_case_ids) == 24
    assert not set(protocol.tuning_case_ids) & set(protocol.heldout_case_ids)
    assert protocol.trials == 3
    categories = [case.name.split("/")[1] for case in suite if case.id in protocol.heldout_case_ids]
    assert sorted(categories.count(category) for category in set(categories)) == [4] * 6
    assert protocol.arms == ARMS


async def test_pairing_uncertainty_and_synthetic_claim_guard():
    reports = [
        await evaluate_cases(cases()[:1], arm=arm, trials=3, worker_factory=FixtureWorker)
        for arm in ARMS
    ]
    comparison = compare_reports(reports)
    assert comparison["paired_attempts"] == 3
    assert comparison["benefit_claim_allowed"] is False
    assert comparison["scope"] == "synthetic_fixture"
    assert comparison["completion_differences"]
    assert comparison["uncertainty_method"] == "case_cluster_bootstrap_95_seed_0"
    with pytest.raises(ValueError, match="paired"):
        compare_reports(
            [reports[0], reports[1].model_copy(update={"results": reports[1].results[:1]})]
        )


def test_live_budget_preflight_before_constructor(tmp_path, monkeypatch, capsys):
    import signal_gating.executive.evaluation as evaluation

    def forbidden(**kwargs):
        raise AssertionError("provider must not be constructed without valid live budgets")

    monkeypatch.setattr(evaluation.OpenAIWorker, "from_environment", forbidden)
    assert (
        main(
            [
                "--cases",
                str(EXAMPLES / "cases.jsonl"),
                "--protocol",
                str(EXAMPLES / "protocol.json"),
                "--output",
                str(tmp_path / "out.json"),
                "--live",
            ]
        )
        == 2
    )
    assert "planned_request_ceiling" in capsys.readouterr().out


def test_report_roundtrip_preserves_unknown():
    report = EvaluationReport(results=())
    assert EvaluationReport.model_validate(json.loads(report.model_dump_json())) == report


async def test_actual_categories_and_observed_context_selection():
    from signal_gating.executive.records import ArtifactRef, ContextPacket
    from signal_gating.executive.store import Store

    suite = cases()[:6]
    report = await evaluate_cases(suite, arm="executive", trials=1, worker_factory=FixtureWorker)
    assert report.accepted
    assert report.results[1].metrics["correct_resumptions"] == 1
    assert report.results[3].metrics["failed_worker_calls"] == 2
    assert report.results[3].failures == ("no_progress",) * 2
    assert report.results[4].metrics["unknown_effects_before_restart"] == 1
    assert report.results[4].metrics["effects"] == 1
    contradiction = suite[2]
    ablated = await evaluate_cases(
        [contradiction], arm="executive_no_memory_gating", trials=1, worker_factory=FixtureWorker
    )
    selected = []
    for result in (report.results[2], ablated.results[0]):
        with Store.open_readonly(Path(result.store_path)) as store:
            artifacts = {r.id: r for r in store.view().records if isinstance(r, ArtifactRef)}
            packets = sorted(
                (r for r in store.view().records if isinstance(r, ContextPacket)),
                key=lambda p: p.created_at,
            )
            packet = next(p for p in packets if p.goal.criteria[0].revision == 2)
            sources = [
                a["ref"]["id"]
                for a in json.loads(packet.rendered_input)["untrusted_evidence"]["artifacts"]
                if artifacts[a["ref"]["id"]].media_type == "text/x-evaluation-source"
            ]
            checkpoint = next(
                r
                for r in artifacts.values()
                if r.media_type == "text/x-evaluation-source"
                and r.id in {ref.id for ref in packets[0].manifest.included}
            )
            selected.append(bool(sources) and sources[0] == checkpoint.id)
            assert (
                packet.manifest.input_tokens
                + packet.manifest.output_tokens_reserved
                + packet.manifest.framing_tokens_reserved
                <= packet.manifest.context_token_limit
            )
    assert selected == [True, False]


async def test_notebook_compaction_rule_plan_dedup_and_fifo_are_real():
    from signal_gating.executive.evaluation import NOTE_TYPE, PLAN_TYPE
    from signal_gating.executive.records import ArtifactRef, ContextPacket
    from signal_gating.executive.store import Store

    first_tasks = []
    for arm, media_type in (("notes", NOTE_TYPE), ("rules", PLAN_TYPE)):
        report = await evaluate_cases([cases()[5]], arm=arm, trials=1, worker_factory=FixtureWorker)
        assert report.accepted
        with Store.open_readonly(Path(report.results[0].store_path)) as store:
            records = store.view().records
            packets = sorted(
                (r for r in records if isinstance(r, ContextPacket)), key=lambda p: p.created_at
            )
            first_tasks.append(packets[0].goal.priority)
            notebook = next(
                r
                for r in records
                if isinstance(r, ArtifactRef)
                and r.media_type == media_type
                and r.scope.goal_id == packets[0].goal.id
            )
            # Exact persisted messages, selected through normal ACL/capacity checks.
            body = json.loads(packets[0].rendered_input)
            text = next(
                a["text"]
                for a in body["untrusted_evidence"]["artifacts"]
                if a["ref"]["id"] == notebook.id
            )
            notes = json.loads(text)
            assert notes["compacted_operations"] >= 0
            if arm == "rules":
                assert notes["plan"] and notes["deduplicated_operations"] > 0
            else:
                assert "plan" not in notes
    assert first_tasks == [0, 10]


async def test_wrong_bytes_cannot_complete_from_expected_oracle():
    class Wrong(FixtureWorker):
        async def run_episode(self, packet):
            result = await super().run_episode(packet)
            if result.artifact_drafts:
                draft = result.artifact_drafts[0].model_copy(update={"text": "wrong"})
                return result.model_copy(update={"artifact_drafts": (draft,)})
            return result

    report = await evaluate_cases(cases()[1:2], arm="executive", trials=1, worker_factory=Wrong)
    assert not report.results[0].completed
    assert not report.results[0].summary.completed_goal_ids
    assert report.results[0].measurements["verifier_seconds"] > 0


async def test_factory_failures_retained_and_replay_not_fresh():
    def broken():
        raise RuntimeError("do not echo provider data")

    report = await evaluate_cases(cases()[:1], arm="notes", trials=3, worker_factory=broken)
    assert report.attempts == 3
    assert report.cost_status == "unknown"
    assert all(r.failures == ("worker_factory_failure",) for r in report.results)
    replay = report.model_copy(update={"evaluation_kind": "replay"})
    with pytest.raises(ValueError, match="fresh"):
        compare_reports([replay])


async def test_optional_bounded_adapter_executes_supplied_worker_without_sdk_or_network():
    from signal_gating.executive.context import UTF8ByteCounter
    from signal_gating.executive.evaluation import CAPACITY, OUTPUT, OpenAIWorker
    from signal_gating.executive.records import Limits

    class Client:
        max_retries = 0
        closed = False

        async def close(self):
            self.closed = True

    clients = []

    class Bounded(OpenAIWorker):
        def __init__(self):
            client = Client()
            clients.append(client)
            self.fixture = FixtureWorker()
            super().__init__(
                client,
                model="injected-no-network",
                counter=UTF8ByteCounter(),
                capacity=CAPACITY,
                quote=Limits(
                    input_tokens=CAPACITY,
                    output_tokens=OUTPUT,
                    calls=1,
                    elapsed_seconds=30,
                    cost_microunits=10,
                ),
                quote_version="test-v1",
                quote_source="free-test-double",
            )

        async def run_episode(self, packet):
            return await self.fixture.run_episode(packet)

    report = await evaluate_cases(cases()[1:2], arm="executive", trials=1, worker_factory=Bounded)
    assert report.results[0].completed
    assert report.worker_calls > 0
    assert all(client.closed for client in clients)


async def test_every_arm_denies_unregistered_effect_with_same_hard_requirement():
    from signal_gating.executive.messages import ActionIntent

    class Forbidden:
        async def run_episode(self, packet):
            assert all(c.hard for c in packet.goal.constraints)
            return EpisodeResult(
                stop_reason="proposed",
                usage=Usage.zero().model_copy(update={"calls": 1}),
                proposals=(
                    ActionIntent(
                        episode_id=packet.episode_id,
                        fencing_token=packet.fencing_token,
                        read_set=packet.mandatory_read_set,
                        capability_id="send_email",
                        arguments={},
                    ),
                ),
            )

    for arm in ARMS:
        report = await evaluate_cases(cases()[1:2], arm=arm, trials=1, worker_factory=Forbidden)
        result = report.results[0]
        assert not result.completed
        assert any(receipt.disposition in {"deny", "reject"} for receipt in result.receipts)
        assert not (Path(result.store_path) / "marker.txt").exists()
        assert result.metrics.get("effects", 0) == 0


def test_threshold_math_unknown_cost_and_heldout_guard_are_conservative():
    """Constructed scorer inputs only, never evidence from an observed run."""
    from signal_gating.executive.records import EvaluationResult, RunSummary

    heldout = [case for case in cases() if case.name.startswith("heldout/")]
    reports = []
    for arm in ARMS:
        results = tuple(
            EvaluationResult(
                case_id=case.id,
                trial=trial,
                arm=arm,
                policy_version="scorer-unit-fixture",
                passed=True,
                summary=RunSummary(completed_goal_ids=case.expected_completed_goal_ids),
                metrics={
                    "unauthorized_effects": 0,
                    "hard_constraint_violations": 0,
                    "eligible_resumptions": int("/interruption/" in case.name),
                    "correct_resumptions": int("/interruption/" in case.name),
                },
                measurements={"cost_microunits": 80 if arm == "executive" else 100},
            )
            for case in heldout
            for trial in range(3)
        )
        reports.append(
            EvaluationReport(
                results=results,
                arm=arm,
                configuration_digest="same",
                manifest_digest="scorer-unit-fixture",
                case_names={case.id: case.name for case in heldout},
            )
        )
    comparison = compare_reports(reports)
    assert comparison["criteria_met"] is True
    assert comparison["benefit_claim_allowed"] is False
    unknown = reports[-1].results[0].model_copy(update={"measurements": {"cost_microunits": None}})
    reports[-1] = reports[-1].model_copy(update={"results": (unknown, *reports[-1].results[1:])})
    comparison = compare_reports(reports)
    assert comparison["criteria_met"] is False
    assert comparison["cost_claim_allowed"] is False
    assert comparison["total_cost_microunits"][ARMS[-1]] is None


def test_cost_measurements_preserve_integer_precision_and_estimates_remain_unknown():
    from signal_gating.executive.evaluation import _aggregate_usage
    from signal_gating.executive.records import EvaluationResult, RunSummary

    cost = 2**60 + 1
    result = EvaluationResult(
        case_id=cases()[0].id,
        policy_version="precision-unit-fixture",
        passed=False,
        summary=RunSummary(),
        measurements={"cost_microunits": cost},
    )
    assert result.measurements["cost_microunits"] == cost
    estimate = Usage(status="estimated", cost_microunits=100)
    assert _aggregate_usage([estimate]).cost_microunits is None


async def test_category_requirements_cannot_be_bypassed_by_arithmetic_completion():
    class IgnoresRobustnessInstruction(FixtureWorker):
        async def run_episode(self, packet):
            ordinary = packet.task.model_copy(update={"subgoal": "Compute the arithmetic answer"})
            return await super().run_episode(packet.model_copy(update={"task": ordinary}))

    for case in cases()[3:5]:
        report = await evaluate_cases(
            [case], arm="executive", trials=1, worker_factory=IgnoresRobustnessInstruction
        )
        result = report.results[0]
        assert result.summary.completed_goal_ids
        assert not result.passed
        assert any(failure.startswith("category_") for failure in result.failures)
