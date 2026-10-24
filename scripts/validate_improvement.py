"""Apply a bounded maintenance patch before running it without write credentials."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath


def allowed_path(name: str) -> bool:
    """Keep workflow, policy, dependency, and agent controls out of proposals."""
    item = PurePosixPath(name)
    if item.is_absolute() or any(part.startswith(".") for part in item.parts):
        return False
    if item.name in {"AGENTS.md", "CLAUDE.md", "GEMINI.md"}:
        return False
    if name in {"README.md", "CHANGELOG.md"}:
        return True
    if item.suffix == ".py":
        return name.startswith(("src/signal_gating/", "tests/", "examples/"))
    return item.suffix == ".md" and name.startswith("docs/")


def apply_patch(patch: Path) -> None:
    """Fail before executing proposed code if the patch exceeds its authority."""
    if patch.stat().st_size > 100_000:
        raise ValueError("maintenance patches must be at most 100 KB")
    if not patch.stat().st_size:
        print("No change proposed")
        return
    subprocess.run(["git", "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "apply", "--index", str(patch)], check=True)
    names = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--no-renames", "-z"]
    ).decode().split("\0")[:-1]
    if not names or len(names) > 10 or any(not allowed_path(name) for name in names):
        raise ValueError("maintenance patches may only change up to 10 approved SDK files")
    stats = subprocess.check_output(["git", "diff", "--cached", "--numstat", "-z"])
    if any(entry.startswith(b"-\t-\t") for entry in stats.split(b"\0")):
        raise ValueError("maintenance patches may not contain binary files")
    # Check indexed modes, not working-tree paths: a patch may introduce symlinks
    # or executable files that look like Python or Markdown by filename alone.
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    for entry in filter(None, entries):
        metadata, name = entry.split(b"\t", 1)
        if name.decode() in names and metadata.split()[0] != b"100644":
            raise ValueError("maintenance patches may only create or update regular files")
    print(f"Validated maintenance patch: {len(names)} files")


if __name__ == "__main__":
    try:
        apply_patch(Path(sys.argv[1]).resolve())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"error: {error}")
