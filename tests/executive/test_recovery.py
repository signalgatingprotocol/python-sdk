"""Durable restart recovery, without inventing external outcomes."""

from signal_gating.executive import CreateGoal, Episode, Goal, PauseGoal, Reservation, TaskFrame
from signal_gating.executive.actions import CapabilityRegistry
from signal_gating.executive.recovery import recover
from signal_gating.executive.reducers import rebuild

from .factories import owner
from .test_kernel import FakeClock, FunctionWorker, blocked, new_goal, open_kernel, records


async def test_lost_lease_recovery_fences_and_holds_actual_subject(tmp_path):
    clock = FakeClock()
    kernel = open_kernel(tmp_path, FunctionWorker(blocked), clock=clock)
    await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
    admitted = kernel._admit()
    assert len(admitted) == 1
    episode, packet = admitted[0]
    before = kernel.store.view()
    batch = recover(kernel.store, registry=CapabilityRegistry(), now=clock.now())
    assert kernel.store.view() == before
    kernel.store.close()  # Simulated process loss after committed admission.
    async with open_kernel(tmp_path, FunctionWorker(blocked), clock=clock) as restarted:
        assert next(e for e in records(restarted, Episode) if e.id == episode.id).status == "fenced"
        reservation = records(restarted, Reservation)[0]
        assert reservation.subject_id == episode.id
        assert reservation.status == "uncertain"
        assert records(restarted, TaskFrame)[0].status == "suspended"
        assert restarted.store.view() == rebuild(restarted.store.events())
        assert batch.events


async def test_paused_goal_restart_never_resumes(tmp_path):
    worker = FunctionWorker(blocked)
    async with open_kernel(tmp_path, worker) as kernel:
        await kernel.submit(CreateGoal(goal=new_goal()), principal=owner())
        current = records(kernel, Goal)[0]
        await kernel.submit(
            PauseGoal(goal_id=current.id, expected_revision=current.revision), principal=owner()
        )
    async with open_kernel(tmp_path, worker) as restarted:
        await restarted.run_until_idle()
        assert records(restarted, Goal)[0].status == "paused"
        assert not worker.packets
        assert restarted.store.view() == rebuild(restarted.store.events())
