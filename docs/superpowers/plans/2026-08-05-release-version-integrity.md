# Release Version Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a release from publishing a wheel whose distribution version, runtime `signal_gating.__version__`, and GitHub release tag disagree.

**Architecture:** Add one repository-local release checker with a pure validation function and a thin command-line boundary. Unit tests lock down accepted tags and actionable mismatch errors; package CI and the publish workflow run the same checker against the installed wheel so release truth has one enforcement path.

**Tech Stack:** Python 3.10+, `argparse`, `importlib.metadata`, pytest, GitHub Actions, Hatchling.

## Global Constraints

- Preserve Python support at `>=3.10`.
- Add no runtime or development dependency.
- Keep `pyproject.toml` as the distribution-version source and `signal_gating.__version__` as the public runtime version.
- Accept release tags only as `<version>` or `v<version>`.
- Validate the installed wheel before publishing, not only checkout source files.
- Error messages must name every conflicting value and state the required fix.
- Do not publish a release, create a tag, push, or merge an external pull request in this task.

---

## File Structure

- `scripts/__init__.py`: make repository automation importable by focused unit tests without adding it to the wheel.
- `scripts/check_release_version.py`: own pure version comparison and the installed-distribution CLI boundary.
- `tests/test_release_version.py`: prove accepted tag shapes and deterministic mismatch failures.
- `.github/workflows/ci.yml`: replace the hard-coded `0.1.0` wheel assertion with the shared checker.
- `.github/workflows/release.yml`: validate the installed wheel and release tag with the shared checker before PyPI publication.

---

### Task 1: Enforce one release-version contract

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/check_release_version.py`
- Create: `tests/test_release_version.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`

**Interfaces:**
- Produces: `check_versions(distribution_version: str, runtime_version: str, release_tag: str | None = None) -> None`.
- Produces: `python scripts/check_release_version.py [--tag TAG]`, exiting nonzero with an actionable message on mismatch.
- Consumes: installed `importlib.metadata.version("signal-gating")`, `signal_gating.__version__`, and optional GitHub release tag.

- [x] **Step 1: Write focused contract tests**

Create `tests/test_release_version.py`:

```python
import pytest

from scripts import check_release_version
from scripts.check_release_version import check_versions


@pytest.mark.parametrize("tag", [None, "0.1.0", "v0.1.0"])
def test_check_versions_accepts_matching_versions(tag: str | None) -> None:
    check_versions("0.1.0", "0.1.0", tag)


def test_check_versions_rejects_runtime_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"distribution version 0\.2\.0 does not match runtime version 0\.1\.0",
    ):
        check_versions("0.2.0", "0.1.0")


def test_check_versions_rejects_release_tag_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"release tag v0\.2\.0 does not match distribution version 0\.1\.0",
    ):
        check_versions("0.1.0", "0.1.0", "v0.2.0")


def test_main_reports_release_tag_mismatch_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check_release_version, "version", lambda _: "0.1.0")
    monkeypatch.setattr(check_release_version.signal_gating, "__version__", "0.1.0")
    monkeypatch.setattr("sys.argv", ["check_release_version.py", "--tag", "v0.2.0"])

    with pytest.raises(SystemExit) as error:
        check_release_version.main()

    assert error.value.code == 1
    assert capsys.readouterr().err == (
        "error: release tag v0.2.0 does not match distribution version 0.1.0; "
        "publish tag 0.1.0 or v0.1.0\n"
    )
```

- [x] **Step 2: Run the tests and capture RED**

Run:

```bash
.venv/bin/pytest tests/test_release_version.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.check_release_version'`.

- [x] **Step 3: Implement the checker and CLI**

Create an empty `scripts/__init__.py`, then create `scripts/check_release_version.py`:

```python
"""Verify that package metadata, runtime API, and a release tag agree."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import signal_gating


def check_versions(
    distribution_version: str,
    runtime_version: str,
    release_tag: str | None = None,
) -> None:
    """Raise an actionable error when any release version source disagrees."""
    if distribution_version != runtime_version:
        raise ValueError(
            f"distribution version {distribution_version} does not match runtime version "
            f"{runtime_version}; update pyproject.toml and signal_gating.__version__ together"
        )
    if release_tag not in (None, distribution_version, f"v{distribution_version}"):
        raise ValueError(
            f"release tag {release_tag} does not match distribution version "
            f"{distribution_version}; publish tag {distribution_version} or "
            f"v{distribution_version}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify installed Signal Gating release versions."
    )
    parser.add_argument("--tag", help="GitHub release tag, with optional v prefix")
    args = parser.parse_args()
    distribution_version = version("signal-gating")
    runtime_version = signal_gating.__version__
    try:
        check_versions(distribution_version, runtime_version, args.tag)
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Verified Signal Gating release version {distribution_version}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run focused GREEN and the real installed-package check**

Run:

```bash
.venv/bin/pytest tests/test_release_version.py -q
.venv/bin/python scripts/check_release_version.py --tag v0.1.0
```

Expected: `6 passed` and `Verified Signal Gating release version 0.1.0`.

- [x] **Step 5: Wire the checker into package CI and publishing**

In `.github/workflows/ci.yml`, replace the hard-coded wheel smoke assertion with:

```yaml
      - name: Smoke-test built wheel
        run: |
          python -m pip install --force-reinstall dist/*.whl
          python scripts/check_release_version.py
```

In `.github/workflows/release.yml`, remove the source-only `Validate tag and package version` step. Replace the wheel smoke command with:

```yaml
      - name: Verify installed release version
        run: |
          python -m pip install --force-reinstall dist/*.whl
          python scripts/check_release_version.py --tag "${GITHUB_REF_NAME}"
```

Keep this step immediately before `pypa/gh-action-pypi-publish` so a mismatch blocks publication.

- [x] **Step 6: Run the release-readiness verification gate**

Run:

```bash
.venv/bin/pytest tests/test_release_version.py tests/test_public_api.py -q
.venv/bin/ruff check scripts/check_release_version.py tests/test_release_version.py
.venv/bin/mypy src/
.venv/bin/pytest -q
git diff --check
```

Expected: all commands pass; the full suite reports 735 passing tests.

- [x] **Step 7: Review the bounded diff**

Confirm the diff contains only the plan, checker, checker tests, and the two workflow integrations. Leave the implementation uncommitted for user review.
