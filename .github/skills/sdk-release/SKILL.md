---
name: sdk-release
description: Prepare Python SDK release candidates and verify release readiness. Use for version changes, distribution checks, or release preparation in this repository.
---

# SDK release

Read `RELEASING.md` for the release procedure and trusted-publisher configuration.
Inspect Git status and existing release PRs first. Preserve unrelated changes.

Keep the version in `pyproject.toml`, `src/signal_gating/__init__.py`, the target
release tag, and the changelog consistent. Base release notes on actual commits.
Do not describe an unpublished candidate as available on PyPI.

Install the declared development and packaging dependencies in a virtual
environment. Run `make check smoke`, then check the intended tag explicitly:

```bash
python scripts/check_release_version.py --tag "v<version>"
```

The smoke target builds and checks distributions, installs the wheel, verifies
its version, and runs the installed console demo. Use the existing release
workflow for authorized publication; retain its `pypi` environment and OIDC.

Report the candidate commit, version, verification results, and any missing
publisher configuration. After an authorized publication, follow the clean
environment installation check in `RELEASING.md` before claiming it succeeded.
