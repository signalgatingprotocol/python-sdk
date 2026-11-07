"""Local POSIX ownership. Persistent lock files are never deleted or stolen."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

from .errors import StoreError


def require_writer_platform() -> None:
    if sys.platform not in {"linux", "darwin"}:
        raise StoreError("unsupported_platform", "Executive writers require Linux or macOS")


class OwnershipLock:
    """An advisory OS lock, released even when the owning process is killed."""

    def __init__(self, descriptor: int, locking: ModuleType) -> None:
        self._descriptor: int | None = descriptor
        self._locking = locking
        self._pid = os.getpid()

    @classmethod
    def acquire(cls, root: Path) -> OwnershipLock:
        require_writer_platform()
        # Keep package imports portable; unsupported writer platforms fail above.
        import fcntl

        descriptor = os.open(
            root / ".owner.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise StoreError("store_owned", "Another writer owns this executive store") from exc
        except BaseException:
            os.close(descriptor)
            raise
        return cls(descriptor, fcntl)

    def close(self) -> None:
        if self._descriptor is not None:
            # A forked child must never unlock the parent's shared description.
            if self._pid == os.getpid():
                self._locking.flock(self._descriptor, self._locking.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None
