"""Private content-addressed evidence with journalled, goal-scoped references.

Use on the Store's owning thread. Byte I/O occurs outside SQLite transactions.
One writer coordinates staging, commit and collection. Staged references are
pinned across ArtifactStore instances sharing that Store until release_staged;
call it in a finally block after accepting or rejecting a proposal bundle.
Process loss drops pins, leaving uncommitted bytes explicitly collectible.

This is a local cooperative boundary, not protection against a hostile process
with the same OS identity. Callers supply adapter-authenticated Principals.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from uuid import uuid4
from weakref import WeakKeyDictionary

from .errors import AuthorityError, StoreError
from .messages import CommitBatch
from .records import (
    ArtifactDraft,
    ArtifactRef,
    Goal,
    JournalEvent,
    Principal,
    Ref,
    Scope,
    StateView,
)
from .store import Store

_PINS: WeakKeyDictionary[Store, dict[str, ArtifactRef]] = WeakKeyDictionary()
_BLOB_NAME = re.compile(r"[0-9a-f]{64}\Z")
_TEMP_NAME = re.compile(r"\.stage-[0-9a-f]{32}\Z")


class AccessDenied(AuthorityError):  # noqa: N818 - public artifact access contract
    """An artifact reference is absent, invalid or outside the caller's scope."""


def _deny() -> None:
    # Identical messages avoid disclosing existence, ACLs or storage metadata.
    raise AccessDenied("artifact_access_denied", "Artifact access denied")


class ArtifactStore:
    """Manage bytes; authoritative validity and ACLs live only in Store records.

    Limits are trusted operator configuration, never values from a worker.
    max_draft_bytes bounds the aggregate UTF-8 bytes in one staged bundle.
    Constructing this object performs no filesystem mutation, so pinned read-only
    Store snapshots can use read(). Each snapshot-backed read opens/closes a
    fresh Store snapshot for current authorization (including its preflight disk
    I/O cost). Authorization linearizes at that snapshot; later revocation cannot
    retract already-authorized bytes. All mutation checks writer ownership.
    """

    def __init__(
        self,
        store: Store,
        *,
        max_artifact_bytes: int = 8_388_608,
        max_draft_bytes: int = 8_388_608,
    ) -> None:
        for limit in (max_artifact_bytes, max_draft_bytes):
            if type(limit) is not int or limit < 1:
                raise ValueError("Artifact byte limits must be positive integers")
        self.store = store
        self.root = store.root / "artifacts"
        self.max_artifact_bytes = max_artifact_bytes
        self.max_draft_bytes = max_draft_bytes
        self._staged = _PINS.setdefault(store, {})

    @contextmanager
    def _directory(self, *, create: bool = False) -> Iterator[int]:
        if create:
            self.store.check_writable()
        else:
            self.store.view()  # Refuse closed/inherited stores before any byte access.
        descriptor = None
        try:
            if create:
                self.root.mkdir(mode=0o700, exist_ok=True)
                # Persist the directory entry as well as later blob renames.
                parent = os.open(self.store.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if create:
                os.fchmod(descriptor, 0o700)
            yield descriptor
        except OSError as exc:
            raise StoreError("artifact_io_failed", "Artifact storage I/O failed") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _owner_goal(self, principal: Principal, goal_id: str) -> Goal:
        records = {r.id: r for r in self.store.view().records}
        goal = records.get(goal_id)
        if (
            not isinstance(goal, Goal)
            or goal.owner_id != principal.id
            or not isinstance(records.get(principal.id), Principal)
        ):
            _deny()
        assert isinstance(goal, Goal)
        return goal

    def _current(
        self,
        ref: ArtifactRef,
        *,
        principal: Principal,
        goal_id: str | None = None,
        owner_only: bool = False,
        state: StateView | None = None,
    ) -> ArtifactRef:
        records = {r.id: r for r in (state or self.store.view()).records}
        current = records.get(ref.id)
        if not isinstance(current, ArtifactRef):
            _deny()
        assert isinstance(current, ArtifactRef)
        goal = records.get(current.scope.goal_id or "")
        if (
            not current.valid
            or not isinstance(records.get(principal.id), Principal)
            or not isinstance(goal, Goal)
            or goal.owner_id != current.scope.owner_id
            or (goal_id is not None and current.scope.goal_id != goal_id)
            or (
                principal.id != current.scope.owner_id
                and (owner_only or principal.id not in current.shared_with)
            )
        ):
            _deny()
        return current

    @staticmethod
    def _read_blob(directory: int, digest: str, size: int, *, private: bool = False) -> bytes:
        descriptor = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise StoreError("artifact_corrupt", "Artifact bytes failed integrity verification")
            data = stream.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise StoreError("artifact_corrupt", "Artifact bytes failed integrity verification")
            if private:
                os.fchmod(stream.fileno(), 0o600)
                os.fsync(stream.fileno())
        return data

    def _stage(self, data: bytes, principal: Principal, goal: Goal, media_type: str) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        ref = ArtifactRef(
            digest=digest,
            storage_location=f"artifacts/{digest}",
            media_type=media_type,
            origin=principal.id,
            scope=Scope(owner_id=goal.owner_id, goal_id=goal.id),
            size_bytes=len(data),
        )
        with self._directory(create=True) as directory:
            try:
                os.stat(digest, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                temporary = f".stage-{uuid4().hex}"
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=directory,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        os.fchmod(stream.fileno(), 0o600)
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, digest, src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except FileNotFoundError:
                        pass
            else:
                self._read_blob(directory, digest, len(data), private=True)
        self._staged[ref.id] = ref
        return ref

    def _limit(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("Artifact data must be bytes")
        if len(data) > self.max_artifact_bytes:
            raise StoreError("artifact_too_large", "Artifact exceeds configured byte limit")

    def put(
        self, data: bytes, *, principal: Principal, goal_id: str, media_type: str
    ) -> ArtifactRef:
        """Stage bytes, then journal a new scoped reference; failures leave collectible bytes."""
        self.store.check_writable()
        goal = self._owner_goal(principal, goal_id)
        self._limit(data)
        ref = self._stage(data, principal, goal, media_type)
        try:
            event = JournalEvent(
                sequence=1,
                event_type="state.changed",
                producer_id="artifact-store",
                principal_id=principal.id,
                subject=Ref(id=ref.id, kind="artifact"),
                payload={"records": [ref.model_dump(mode="json")]},
            )
            self.store.commit(
                CommitBatch(
                    events=(event,),
                    expected_revisions=(Ref(id=goal.id, revision=goal.revision, kind="goal"),),
                )
            )
            return ref
        finally:
            self.release_staged((ref,))

    def stage_drafts(
        self,
        drafts: Iterable[ArtifactDraft],
        *,
        principal: Principal,
        goal_id: str,
    ) -> dict[str, ArtifactRef]:
        """Validate a whole bundle before I/O, then pin uncommitted refs by draft key."""
        self.store.check_writable()
        goal = self._owner_goal(principal, goal_id)
        prepared: list[tuple[ArtifactDraft, bytes]] = []
        keys: set[str] = set()
        total = 0
        for draft in drafts:
            data = draft.text.encode("utf-8")
            self._limit(data)
            total += len(data)
            if total > self.max_draft_bytes:
                raise StoreError("drafts_too_large", "Artifact drafts exceed aggregate byte limit")
            if draft.key in keys:
                raise ValueError("Artifact draft keys must be unique")
            keys.add(draft.key)
            prepared.append((draft, data))
        refs: dict[str, ArtifactRef] = {}
        try:
            for draft, data in prepared:
                refs[draft.key] = self._stage(data, principal, goal, draft.media_type)
        except BaseException:
            self.release_staged(refs.values())
            raise
        return refs

    def release_staged(self, refs: Iterable[ArtifactRef]) -> None:
        """End pins after bundle acceptance/rejection; collection remains explicit."""
        for ref in refs:
            self._staged.pop(ref.id, None)

    def read(self, ref: ArtifactRef, *, principal: Principal, goal_id: str) -> bytes:
        """Resolve current validity/ACL by ID before opening any artifact bytes."""
        try:
            self.store.check_writable()
        except StoreError as exc:
            if exc.code != "readonly_store":
                raise
            # A long-lived inspection snapshot cannot preserve a revoked ACL.
            with Store.open_readonly(self.store.root) as snapshot:
                current = self._current(
                    ref, principal=principal, goal_id=goal_id, state=snapshot.view()
                )
        else:
            current = self._current(ref, principal=principal, goal_id=goal_id)
        with self._directory() as directory:
            return self._read_blob(directory, current.digest, current.size_bytes)

    def delete(self, ref: ArtifactRef, *, principal: Principal) -> JournalEvent:
        """Commit revocation first; failed physical cleanup leaves only an orphan."""
        self.store.check_writable()
        current = self._current(ref, principal=principal, owner_only=True)
        event = JournalEvent(
            sequence=1,
            event_type="artifact_invalidated",
            producer_id="artifact-store",
            principal_id=principal.id,
            subject=Ref(id=current.id, revision=current.revision, kind="artifact"),
            payload={},
        )
        committed = self.store.commit(
            CommitBatch(
                events=(event,),
                expected_revisions=(
                    Ref(id=current.id, revision=current.revision, kind="artifact"),
                ),
            )
        )
        self.collect_orphans()
        return committed[0]

    def collect_orphans(self) -> int:
        """Remove unreferenced blobs/abandoned temporaries, never age-delete evidence."""
        self.store.check_writable()
        live = {
            r.digest for r in self.store.view().records if isinstance(r, ArtifactRef) and r.valid
        }
        live.update(ref.digest for ref in self._staged.values())
        count = 0
        with self._directory(create=True) as directory:
            for name in os.listdir(directory):
                if name in live or not (_BLOB_NAME.fullmatch(name) or _TEMP_NAME.fullmatch(name)):
                    continue
                os.unlink(name, dir_fd=directory)
                count += 1
            if count:
                os.fsync(directory)
        return count
