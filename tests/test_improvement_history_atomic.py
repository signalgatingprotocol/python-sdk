"""Pruning must not publish memory state before persistence succeeds."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import signal_gating.improvement as improvement
from signal_gating.improvement import (
    ImprovementHistory,
    ImprovementRecord,
    RetentionPolicy,
)


def record(number: int) -> ImprovementRecord:
    return ImprovementRecord(
        iteration=number,
        incumbent_id=f"v{number - 1}",
        candidate_id=f"v{number}",
        focus_dimension="quality",
        focus_case_ids=("case",),
        focus_evidence=("deterministic fixture",),
        baseline_scores={"quality": 0.2},
        candidate_scores={"quality": 0.3},
        candidate_evidence={"case": {"quality": ("verified fixture",)}},
        baseline_progress=0.2,
        candidate_progress=0.3,
        progress_delta=0.1,
        focus_delta=0.1,
        accepted=True,
        regressions=(),
        created_at=990.0 + 10.0 * number,
    )


@pytest.fixture
def history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ImprovementHistory:
    monkeypatch.setattr(improvement.time, "time", lambda: 1000.0)
    result = ImprovementHistory(
        tmp_path / "history.jsonl",
        retention=RetentionPolicy(max_age_seconds=60, max_records=None, max_bytes=None),
    )
    result.append(record(1))
    result.append(record(2))
    return result


def inject_failure(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"injected {stage} failure")

    if stage == "encode":
        monkeypatch.setattr(improvement, "_encode_history", fail)
    elif stage in ("chmod", "fsync", "replace"):
        monkeypatch.setattr(improvement.os, stage, fail)
    elif stage == "open":
        monkeypatch.setattr(improvement, "open", fail, raising=False)
    else:
        real_open = open

        @contextmanager
        def failing_open(*args: Any, **kwargs: Any):
            with real_open(*args, **kwargs) as handle:
                class FailingHandle:
                    def __getattr__(self, name: str) -> Any:
                        return fail if name == stage else getattr(handle, name)

                yield FailingHandle()

        monkeypatch.setattr(improvement, "open", failing_open, raising=False)


STAGES = ("encode", "open", "chmod", "write", "flush", "fsync", "replace")


@pytest.mark.parametrize("stage", STAGES)
def test_failed_prune_preserves_memory_disk_and_retry(
    history: ImprovementHistory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    path = tmp_path / "history.jsonl"
    old_records = history.records
    old_head = history.head_digest
    old_bytes = path.read_bytes()

    with monkeypatch.context() as fault:
        inject_failure(fault, stage)
        with pytest.raises(OSError, match=f"injected {stage} failure"):
            history.prune(now=1061.0)

    assert path.read_bytes() == old_bytes
    assert history.head_digest == old_head
    assert history.records == old_records
    assert not list(tmp_path.glob(".history.jsonl.*.tmp"))
    reloaded = ImprovementHistory(path, retention=None)
    assert reloaded.records == old_records
    assert reloaded.head_digest == old_head

    assert history.prune(now=1061.0) == 1
    reloaded = ImprovementHistory(path, retention=None)
    assert reloaded.records == history.records == (record(2),)
    assert reloaded.head_digest == history.head_digest
    assert history.prune(now=1061.0) == 0


@pytest.mark.parametrize("stage", STAGES)
def test_append_after_failed_prune_keeps_reloadable_chain(
    history: ImprovementHistory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    with monkeypatch.context() as fault:
        inject_failure(fault, stage)
        with pytest.raises(OSError, match=f"injected {stage} failure"):
            history.prune(now=1061.0)

    history.append(record(3))
    reloaded = ImprovementHistory(tmp_path / "history.jsonl", retention=None)
    assert reloaded.records == history.records == (record(1), record(2), record(3))
    assert reloaded.head_digest == history.head_digest


def test_successful_prune_remains_appendable(
    history: ImprovementHistory, tmp_path: Path
) -> None:
    assert history.prune(now=1061.0) == 1
    history.append(record(3))
    reloaded = ImprovementHistory(tmp_path / "history.jsonl", retention=None)
    assert reloaded.records == history.records == (record(2), record(3))
    assert reloaded.head_digest == history.head_digest
    assert not list(tmp_path.glob(".history.jsonl.*.tmp"))


def test_memory_only_prune_still_updates_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(improvement.time, "time", lambda: 1000.0)
    history = ImprovementHistory(
        retention=RetentionPolicy(max_age_seconds=60, max_records=None, max_bytes=None)
    )
    history.append(record(1))
    history.append(record(2))
    old_head = history.head_digest
    assert history.prune(now=1061.0) == 1
    assert history.records == (record(2),)
    assert history.head_digest != old_head


def test_retention_disabled_does_not_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(improvement.time, "time", lambda: 1000.0)
    history = ImprovementHistory(retention=None)
    history.append(record(1))
    assert history.prune(now=9999.0) == 0
    assert history.records == (record(1),)
