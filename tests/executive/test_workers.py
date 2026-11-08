"""Controlled cancellation/deadline races without sleeps or paid requests."""

import asyncio
from datetime import timedelta

import pytest

from signal_gating.executive.messages import EpisodeResult
from signal_gating.executive.records import Episode
from signal_gating.executive.workers import EpisodeRunner, WorkerError

from .factories import NOW
from .test_openai_worker import packet


def episode(value, **updates):
    return Episode(
        id=value.episode_id,
        task_id=value.task.id,
        fencing_token=value.fencing_token,
        context_packet_id=value.id,
        worker_id="fixture",
        provider="fixture",
        model="fixture",
        allowed_proposal_types=("evidence_request",),
        deadline=NOW + timedelta(seconds=10),
        **updates,
    )


class ControlledWorker:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.get_running_loop().create_future()
        self.done = asyncio.Event()

    async def run_episode(self, value):
        self.started.set()
        try:
            try:
                return await asyncio.shield(self.release)
            except asyncio.CancelledError:
                self.cancelled.set()
                return await self.release
        finally:
            self.done.set()


class Deadline:
    def __init__(self):
        self.expired = asyncio.Event()

    async def __call__(self, at):
        await self.expired.wait()


async def test_timeout_fences_late_completion_and_bounded_close():
    value = packet()
    deadline = Deadline()
    worker = ControlledWorker()
    runner = EpisodeRunner(
        is_current=lambda e, p: True, clock=lambda: NOW, wait_until=deadline, max_active=1
    )
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    deadline.expired.set()
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.code == "episode_timeout"
    assert caught.value.usage.status == "unknown"
    assert caught.value.usage.calls == 1
    await worker.cancelled.wait()
    with pytest.raises(WorkerError) as bounded:
        await runner.run(episode(value), value, ControlledWorker())
    assert bounded.value.code == "worker_capacity"
    await runner.close()
    worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    await worker.done.wait()
    assert caught.value.usage.status == "unknown"


async def test_caller_cancellation_returns_before_uncooperative_worker():
    value = packet()
    worker = ControlledWorker()
    runner = EpisodeRunner(is_current=lambda e, p: True, clock=lambda: NOW, wait_until=Deadline())
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await worker.cancelled.wait()
    worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    await worker.done.wait()
    await runner.close()


async def test_current_fence_rechecked_after_completion_and_before_dispatch():
    value = packet()
    current = [False]
    worker = ControlledWorker()
    runner = EpisodeRunner(
        is_current=lambda e, p: current[0], clock=lambda: NOW, wait_until=Deadline()
    )
    with pytest.raises(WorkerError) as caught:
        await runner.run(episode(value), value, worker)
    assert caught.value.code == "episode_fenced"
    assert not worker.started.is_set()
    current[0] = True
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    current[0] = False
    worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.code == "episode_fenced"
    await runner.close()


async def test_expired_or_mismatched_episode_does_not_call_worker():
    value = packet()
    worker = ControlledWorker()
    runner = EpisodeRunner(is_current=lambda e, p: True, clock=lambda: NOW, wait_until=Deadline())
    for invalid in [
        episode(value).model_copy(update={"deadline": NOW}),
        episode(value).model_copy(update={"fencing_token": 2}),
        episode(value).model_copy(update={"status": "accepted"}),
    ]:
        with pytest.raises(WorkerError):
            await runner.run(invalid, value, worker)
    assert not worker.started.is_set()
    await runner.close()


async def test_close_interrupts_active_run_and_late_result_cannot_return():
    value = packet()
    worker = ControlledWorker()
    runner = EpisodeRunner(is_current=lambda e, p: True, clock=lambda: NOW, wait_until=Deadline())
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    await runner.close()
    await worker.cancelled.wait()
    # A test deadline prevents hanging the suite while reproducing the old bug.
    if not running.done():
        worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.code == "runner_closed"
    if not worker.release.done():
        worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    await worker.done.wait()


async def test_timeout_race_retains_already_observed_overage():
    from signal_gating.executive.records import Usage

    value = packet()
    deadline = Deadline()
    worker = ControlledWorker()
    runner = EpisodeRunner(is_current=lambda e, p: True, clock=lambda: NOW, wait_until=deadline)
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    worker.release.set_result(
        EpisodeResult(stop_reason="proposed", usage=Usage(output_tokens=3000))
    )
    deadline.expired.set()
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.usage.output_tokens == 3000
    await runner.close()


async def test_late_usage_is_accounting_only_and_retains_overage_until_drained():
    from signal_gating.executive.records import Usage

    value = packet()
    deadline = Deadline()
    worker = ControlledWorker()
    runner = EpisodeRunner(
        is_current=lambda e, p: True, clock=lambda: NOW, wait_until=deadline, max_active=1
    )
    current = episode(value)
    running = asyncio.create_task(runner.run(current, value, worker))
    await worker.started.wait()
    deadline.expired.set()
    with pytest.raises(WorkerError):
        await running
    await worker.cancelled.wait()
    worker.release.set_result(
        EpisodeResult(stop_reason="proposed", usage=Usage(output_tokens=3000))
    )
    await runner.late_usage_ready.wait()
    with pytest.raises(WorkerError) as caught:
        await runner.run(current, value, ControlledWorker())
    assert caught.value.code == "worker_capacity"
    observations = runner.take_late_usage()
    assert observations == ((current, Usage(output_tokens=3000)),)
    assert runner.take_late_usage() == ()
    assert not runner.late_usage_ready.is_set()
    await runner.close()


async def test_deadline_rechecked_even_if_completion_wins_timer():
    value = packet()
    now = [NOW]
    worker = ControlledWorker()
    runner = EpisodeRunner(
        is_current=lambda e, p: True, clock=lambda: now[0], wait_until=Deadline()
    )
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    await worker.started.wait()
    now[0] = NOW + timedelta(seconds=11)
    worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.code == "episode_timeout"
    await runner.close()


async def test_current_state_rechecked_at_actual_child_dispatch():
    value = packet()
    worker = ControlledWorker()
    checks = [0]

    def current(e, p):
        checks[0] += 1
        return checks[0] == 1

    runner = EpisodeRunner(is_current=current, clock=lambda: NOW, wait_until=Deadline())
    running = asyncio.create_task(runner.run(episode(value), value, worker))
    # Old implementation reaches the worker; complete it so this RED cannot hang.
    watcher = asyncio.create_task(worker.started.wait())
    completed, _ = await asyncio.wait((running, watcher), return_when=asyncio.FIRST_COMPLETED)
    if watcher in completed:
        worker.release.set_result(EpisodeResult(stop_reason="proposed"))
    with pytest.raises(WorkerError):
        await running
    watcher.cancel()
    assert not worker.started.is_set()
    await runner.close()


async def test_default_timer_callback_is_safe_after_cancellation(monkeypatch):
    from types import SimpleNamespace

    from signal_gating.executive import workers

    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    armed = asyncio.Event()
    timer = SimpleNamespace(cancel=lambda: None)
    callbacks = []

    def later(delay, callback, *args):
        callbacks.append(lambda: callback(*args))
        armed.set()
        return timer

    fake_loop = SimpleNamespace(create_future=lambda: ready, call_later=later)
    monkeypatch.setattr(workers, "asyncio", SimpleNamespace(get_running_loop=lambda: fake_loop))
    waiting = asyncio.create_task(workers._wait_until(NOW))
    await armed.wait()
    waiting.cancel()
    try:
        callbacks[0]()
    finally:
        with pytest.raises(asyncio.CancelledError):
            await waiting


async def test_provider_adapter_late_completion_retains_usage_without_result_effects():
    from .test_openai_worker import FakeClient
    from .test_openai_worker import worker as adapter

    value = packet()
    controlled = ControlledWorker()

    class Client(FakeClient):
        async def create(self, **kwargs):
            await controlled.run_episode(value)
            return await super().create(**kwargs)

    client = Client(usage={"prompt_tokens": 10, "completion_tokens": 3000})
    deadline = Deadline()
    runner = EpisodeRunner(is_current=lambda e, p: True, clock=lambda: NOW, wait_until=deadline)
    current = episode(value)
    running = asyncio.create_task(runner.run(current, value, adapter(client)))
    await controlled.started.wait()
    deadline.expired.set()
    with pytest.raises(WorkerError) as caught:
        await running
    assert caught.value.code == "episode_timeout"
    assert caught.value.usage.status == "unknown"
    await controlled.cancelled.wait()
    await runner.close()
    controlled.release.set_result(EpisodeResult(stop_reason="proposed"))
    await runner.late_usage_ready.wait()
    observations = runner.take_late_usage()
    assert len(observations) == 1
    assert observations[0][0] == current
    assert observations[0][1].output_tokens == 3000
    assert observations[0][1].status == "unknown"
    assert observations[0][1].source == "provider-observation"
    assert client.calls == 1
