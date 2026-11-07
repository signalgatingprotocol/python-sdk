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

## Installation evidence

Apply these checks when changing README install commands or verifying a release:

- Match advertised behavior to the revision the commands install. While PyPI
  publication is pending, use a full, tested Git commit hash and keep base and
  optional-extra commands consistent. Label behavior outside that snapshot as
  unreleased until the install reference includes it.
- Run the documented base command in a fresh virtual environment outside the
  checkout. Verify the installed version and, for a Git install, the commit in
  the distribution's `direct_url.json`. A checkout import can mask an old or
  incomplete installed package.
- Run the README Quick Start and installed demo. Resolve optional-extra
  commands as well; report a dry run as dependency-resolution evidence, and
  distinguish it from runtime or live-provider testing.

Report the candidate commit, version, verification results, and any missing
publisher configuration. After an authorized publication, follow the clean
environment installation check in `RELEASING.md` before claiming it succeeded.
