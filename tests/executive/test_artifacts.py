"""Scoped artifact bytes, journalled validity, and explicit retention."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from signal_gating.executive import (
    ArtifactDraft,
    ArtifactRef,
    CommitBatch,
    ConflictError,
    Goal,
    InvalidTransition,
    JournalEvent,
    Principal,
    Ref,
    StoreError,
)
from signal_gating.executive.artifacts import AccessDenied, ArtifactStore
from signal_gating.executive.reducers import rebuild
from signal_gating.executive.store import Store

from .factories import GOAL_ID, OWNER_ID, goal, owner


def write(store: Store, *records: Principal | Goal | ArtifactRef) -> None:
    current = {r.id: r for r in store.view().records}
    store.commit(
        CommitBatch(
            events=(
                JournalEvent(
                    sequence=1,
                    event_type="state.changed",
                    producer_id="fixture",
                    principal_id=OWNER_ID,
                    subject=Ref(id=records[0].id),
                    payload={"records": [r.model_dump(mode="json") for r in records]},
                ),
            ),
            expected_revisions=tuple(
                Ref(id=r.id, revision=current[r.id].revision) for r in records if r.id in current
            ),
        )
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "kernel") as value:
        write(value, owner(), goal())
        yield value


def put(artifacts: ArtifactStore, data: bytes = b"evidence") -> ArtifactRef:
    return artifacts.put(data, principal=owner(), goal_id=GOAL_ID, media_type="text/plain")


def blob(store: Store, ref: ArtifactRef) -> Path:
    return store.root / "artifacts" / ref.digest


def test_put_bytes_and_reference_are_private_atomic_and_replayable(store: Store) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"
    assert stat.S_IMODE(blob(store, ref).stat().st_mode) == 0o600
    assert stat.S_IMODE(blob(store, ref).parent.stat().st_mode) == 0o700
    assert rebuild(store.events()) == store.view()
    assert ArtifactStore(store).read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"


@pytest.mark.parametrize("size,allowed", [(4, True), (5, False)])
def test_byte_limit_is_enforced_before_writing(store: Store, size: int, allowed: bool) -> None:
    artifacts = ArtifactStore(store, max_artifact_bytes=4)
    if allowed:
        assert put(artifacts, b"x" * size).size_bytes == 4
    else:
        before = store.view()
        with pytest.raises(StoreError, match="limit"):
            put(artifacts, b"x" * size)
        assert store.view() == before
        assert list((store.root / "artifacts").glob("*")) == []


def test_drafts_count_utf8_and_aggregate_before_staging_any_bytes(store: Store) -> None:
    artifacts = ArtifactStore(store, max_artifact_bytes=4, max_draft_bytes=5)
    drafts = (
        ArtifactDraft(key="a", media_type="text/plain", text="雪"),
        ArtifactDraft(key="b", media_type="text/plain", text="雪"),
    )
    with pytest.raises(StoreError, match="limit"):
        artifacts.stage_drafts(drafts, principal=owner(), goal_id=GOAL_ID)
    assert list((store.root / "artifacts").glob("*")) == []
    with pytest.raises(StoreError, match="limit"):
        artifacts.stage_drafts(
            (ArtifactDraft(key="a", media_type="text/plain", text="雪雪"),),
            principal=owner(),
            goal_id=GOAL_ID,
        )


def test_active_staging_survives_collection_until_explicit_release(store: Store) -> None:
    artifacts = ArtifactStore(store)
    before = store.view()
    refs = artifacts.stage_drafts(
        (ArtifactDraft(key="draft", media_type="text/plain", text="雪"),),
        principal=owner(),
        goal_id=GOAL_ID,
    )
    ref = refs["draft"]
    assert store.view() == before
    with pytest.raises(AccessDenied):
        artifacts.read(ref, principal=owner(), goal_id=GOAL_ID)
    assert artifacts.collect_orphans() == 0
    assert blob(store, ref).read_bytes() == "雪".encode()
    artifacts.release_staged(refs.values())
    assert artifacts.collect_orphans() == 1
    assert not blob(store, ref).exists()


def test_accepted_draft_remains_live_after_staging_release(store: Store) -> None:
    artifacts = ArtifactStore(store)
    refs = artifacts.stage_drafts(
        (ArtifactDraft(key="draft", media_type="text/plain", text="ok"),),
        principal=owner(),
        goal_id=GOAL_ID,
    )
    ref = refs["draft"]
    write(store, ref)
    artifacts.release_staged(refs.values())
    assert artifacts.collect_orphans() == 0
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"ok"


def test_cross_owner_and_cross_goal_reads_do_not_touch_bytes(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    stranger = owner(id="9" * 32)

    def forbidden(*args: object, **kwargs: object) -> int:
        pytest.fail("denied read opened artifact bytes")

    monkeypatch.setattr(os, "open", forbidden)
    for principal, goal_id in ((stranger, GOAL_ID), (owner(), "a" * 32)):
        with pytest.raises(AccessDenied):
            artifacts.read(ref, principal=principal, goal_id=goal_id)
    with pytest.raises(AccessDenied):
        artifacts.delete(ref, principal=stranger)


def test_shared_reference_uses_current_acl_and_ignores_forged_path(
    store: Store, tmp_path: Path
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    stranger = owner(id="9" * 32)
    write(store, stranger)
    shared = ref.model_copy(update={"revision": 2, "shared_with": (stranger.id,)})
    write(store, shared)
    # Supplied metadata never grants authority or selects content; ID loads current ACL.
    forged = ref.model_copy(
        update={"storage_location": str(tmp_path / "secret"), "digest": "0" * 64}
    )
    assert artifacts.read(forged, principal=stranger, goal_id=GOAL_ID) == b"evidence"
    revoked = shared.model_copy(update={"revision": 3, "shared_with": ()})
    write(store, revoked)
    with pytest.raises(AccessDenied):
        artifacts.read(shared, principal=stranger, goal_id=GOAL_ID)
    with pytest.raises(AccessDenied):
        artifacts.read(
            ref.model_copy(update={"id": uuid4().hex, "shared_with": (stranger.id,)}),
            principal=stranger,
            goal_id=GOAL_ID,
        )


@pytest.mark.parametrize("damage", ["missing", "corrupt", "oversized", "symlink"])
def test_missing_or_damaged_bytes_never_read_as_evidence(
    store: Store, damage: str, tmp_path: Path
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    path = blob(store, ref)
    path.unlink()
    if damage == "corrupt":
        path.write_bytes(b"bad")
    elif damage == "oversized":
        path.write_bytes(b"evidence extra")
    elif damage == "symlink":
        external = tmp_path / "external"
        external.write_bytes(b"evidence")
        path.symlink_to(external)
    with pytest.raises(StoreError):
        artifacts.read(ref, principal=owner(), goal_id=GOAL_ID)


def test_identical_bytes_have_separate_scopes_and_reference_deletion(store: Store) -> None:
    artifacts = ArtifactStore(store)
    first = put(artifacts)
    other_goal = goal(id="a" * 32, criteria=())
    write(store, other_goal)
    second = artifacts.put(
        b"evidence", principal=owner(), goal_id=other_goal.id, media_type="text/plain"
    )
    assert first.id != second.id
    assert first.digest == second.digest
    assert len(list((store.root / "artifacts").iterdir())) == 1
    event = artifacts.delete(first, principal=owner())
    assert event.event_type == "artifact_invalidated"
    assert event.payload == {}
    assert blob(store, second).exists()
    assert artifacts.read(second, principal=owner(), goal_id=other_goal.id) == b"evidence"
    with pytest.raises(AccessDenied):
        artifacts.read(first, principal=owner(), goal_id=GOAL_ID)
    artifacts.delete(second, principal=owner())
    assert not blob(store, second).exists()
    assert rebuild(store.events()) == store.view()


def test_failed_invalidation_keeps_reference_and_bytes(store: Store) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    store._connection.execute(
        "CREATE TRIGGER refuse_delete BEFORE INSERT ON journal "
        "WHEN json_extract(NEW.payload, '$.event_type') = 'artifact_invalidated' "
        "BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    before = store.view()
    with pytest.raises(ConflictError):
        artifacts.delete(ref, principal=owner())
    assert store.view() == before
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"


def test_failed_rename_leaves_no_reference_or_partial_blob(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(store)
    before = store.view()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(StoreError):
        put(artifacts)
    assert store.view() == before
    assert list((store.root / "artifacts").glob("*")) == []


def test_collection_preserves_old_required_active_evidence(store: Store) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    updated = goal(
        revision=2,
        criteria=(
            goal()
            .criteria[0]
            .model_copy(
                update={
                    "required_evidence": (Ref(id=ref.id, kind="artifact"),),
                }
            ),
        ),
    )
    write(store, updated)
    os.utime(blob(store, ref), (1, 1))
    orphan = store.root / "artifacts" / ("a" * 64)
    orphan.write_bytes(b"orphan")
    assert artifacts.collect_orphans() == 1
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"


def test_readonly_store_cannot_mutate_artifacts(store: Store) -> None:
    ref = put(ArtifactStore(store))
    with Store.open_readonly(store.root) as snapshot:
        artifacts = ArtifactStore(snapshot)
        assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"
        for operation in (
            lambda: put(artifacts),
            lambda: artifacts.delete(ref, principal=owner()),
            artifacts.collect_orphans,
        ):
            with pytest.raises(StoreError):
                operation()


def test_new_directory_and_bytes_are_fsynced_before_reference_commit(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(store)
    synced: list[tuple[int, int]] = []
    original_fsync = os.fsync
    original_replace = os.replace

    def sync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        synced.append((info.st_dev, info.st_ino))
        original_fsync(descriptor)

    def rename(source: str, destination: str, **kwargs: int) -> None:
        # A fully flushed file exists, but no authoritative reference exists yet.
        directory = kwargs["src_dir_fd"]
        info = os.stat(source, dir_fd=directory)
        assert (info.st_dev, info.st_ino) in synced
        assert not any(isinstance(r, ArtifactRef) for r in store.view().records)
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", rename)
    ref = put(artifacts)
    for path in (store.root, store.root / "artifacts", blob(store, ref)):
        info = path.stat()
        assert (info.st_dev, info.st_ino) in synced
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"


def test_existing_deduplicated_bytes_have_private_permissions_restored(store: Store) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    blob(store, ref).chmod(0o644)
    second = put(artifacts)
    assert stat.S_IMODE(blob(store, second).stat().st_mode) == 0o600


def test_failed_staging_releases_earlier_pins_for_collection(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(store)
    original = os.replace
    count = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("second stage failed")
        original(*args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(StoreError):
        artifacts.stage_drafts(
            (
                ArtifactDraft(key="a", media_type="text/plain", text="first"),
                ArtifactDraft(key="b", media_type="text/plain", text="second"),
            ),
            principal=owner(),
            goal_id=GOAL_ID,
        )
    assert artifacts.collect_orphans() == 1
    assert list((store.root / "artifacts").iterdir()) == []


def test_other_wrapper_cannot_collect_active_staged_bytes(store: Store) -> None:
    first = ArtifactStore(store)
    refs = first.stage_drafts(
        (ArtifactDraft(key="a", media_type="text/plain", text="draft"),),
        principal=owner(),
        goal_id=GOAL_ID,
    )
    second = ArtifactStore(store)
    assert second.collect_orphans() == 0
    first.release_staged(refs.values())
    assert second.collect_orphans() == 1


@pytest.mark.parametrize("change", ["revive", "rescope", "content"])
def test_state_changes_cannot_redefine_reference_identity_or_revive_it(
    store: Store, change: str
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    if change == "revive":
        artifacts.delete(ref, principal=owner())
        updated = ref.model_copy(update={"revision": 3})
    elif change == "rescope":
        updated = ref.model_copy(
            update={"revision": 2, "scope": ref.scope.model_copy(update={"goal_id": None})}
        )
    else:
        updated = ref.model_copy(update={"revision": 2, "digest": "0" * 64})
    before = store.view()
    with pytest.raises(InvalidTransition):
        write(store, updated)
    assert store.view() == before


def test_pinned_snapshot_cannot_retrieve_bytes_under_a_revoked_acl(store: Store) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    reader = owner(id="9" * 32)
    write(store, reader)
    shared = ref.model_copy(update={"revision": 2, "shared_with": (reader.id,)})
    write(store, shared)
    with Store.open_readonly(store.root) as snapshot:
        write(store, shared.model_copy(update={"revision": 3, "shared_with": ()}))
        with pytest.raises(AccessDenied):
            ArtifactStore(snapshot).read(shared, principal=reader, goal_id=GOAL_ID)


@pytest.mark.parametrize("fault", ["fsync", "commit"])
def test_failed_put_never_publishes_a_reference_and_leftovers_are_collectible(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    artifacts = ArtifactStore(store)
    before = store.view()
    if fault == "commit":
        store._connection.execute(
            "CREATE TRIGGER refuse_reference BEFORE INSERT ON journal "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        error: type[Exception] = ConflictError
    else:

        def fail(descriptor: int) -> None:
            raise OSError("injected fsync failure")

        monkeypatch.setattr(os, "fsync", fail)
        error = StoreError
    with pytest.raises(error):
        put(artifacts)
    assert store.view() == before
    monkeypatch.undo()
    assert artifacts.collect_orphans() == (1 if fault == "commit" else 0)
    assert list((store.root / "artifacts").iterdir()) == []


def test_failed_physical_deletion_still_revokes_access_and_replays(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    original = os.unlink

    def fail(path: str, **kwargs: int) -> None:
        if path == ref.digest:
            raise OSError("injected unlink failure")
        original(path, **kwargs)

    monkeypatch.setattr(os, "unlink", fail)
    with pytest.raises(StoreError):
        artifacts.delete(ref, principal=owner())
    with pytest.raises(AccessDenied):
        artifacts.read(ref, principal=owner(), goal_id=GOAL_ID)
    assert blob(store, ref).exists()
    assert rebuild(store.events()) == store.view()
    monkeypatch.undo()
    assert artifacts.collect_orphans() == 1
    with Store.open_readonly(store.root) as reopened:
        assert not next(r for r in reopened.view().records if r.id == ref.id).valid


def test_symlinked_artifact_directory_cannot_escape_root(store: Store, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "artifacts").symlink_to(outside, target_is_directory=True)
    before = store.view()
    with pytest.raises(StoreError):
        put(ArtifactStore(store))
    assert store.view() == before
    assert list(outside.iterdir()) == []


def test_collection_unlinks_orphan_symlink_without_touching_its_target(
    store: Store, tmp_path: Path
) -> None:
    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    outside = tmp_path / "keep"
    outside.write_bytes(b"keep")
    (store.root / "artifacts" / ("a" * 64)).symlink_to(outside)
    assert artifacts.collect_orphans() == 1
    assert outside.read_bytes() == b"keep"
    assert artifacts.read(ref, principal=owner(), goal_id=GOAL_ID) == b"evidence"


def test_invalidated_artifact_cannot_enable_a_new_ready_task(store: Store) -> None:
    from signal_gating.executive import Commitment

    from .factories import COMMITMENT_ID, frame

    artifacts = ArtifactStore(store)
    ref = put(artifacts)
    artifacts.delete(ref, principal=owner())
    commitment = Commitment(id=COMMITMENT_ID, goal_id=GOAL_ID, responsible_principal_id=OWNER_ID)
    task = frame(dependencies=(Ref(id=ref.id, revision=2, kind="artifact"),))
    event = JournalEvent(
        sequence=1,
        event_type="state.changed",
        producer_id="fixture",
        principal_id=OWNER_ID,
        subject=Ref(id=task.id),
        payload={
            "records": [
                commitment.model_dump(mode="json"),
                task.model_dump(mode="json"),
            ]
        },
    )
    with pytest.raises(InvalidTransition, match="dependencies"):
        store.commit(CommitBatch(events=(event,)))


@pytest.mark.parametrize("payload,revision", [({"extra": True}, 1), ({}, 2)])
def test_artifact_invalidation_rejects_extra_payload_and_stale_subject(
    store: Store,
    payload: dict[str, bool],
    revision: int,
) -> None:
    ref = put(ArtifactStore(store))
    before = store.view()
    event = JournalEvent(
        sequence=1,
        event_type="artifact_invalidated",
        producer_id="fixture",
        principal_id=OWNER_ID,
        subject=Ref(id=ref.id, revision=revision),
        payload=payload,
    )
    with pytest.raises(InvalidTransition):
        store.commit(CommitBatch(events=(event,), expected_revisions=(Ref(id=ref.id),)))
    assert store.view() == before
