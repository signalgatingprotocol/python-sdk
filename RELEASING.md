# Releasing

Publishing a GitHub release triggers `.github/workflows/release.yml`, which
builds the source distribution and wheel, checks both artifacts, verifies the
installed version against the release tag, and publishes to PyPI with trusted
publishing.

## One-time PyPI setup

Configure a trusted publisher for the `signal-gating` project with these exact
values:

| Field | Value |
| --- | --- |
| PyPI project | `signal-gating` |
| Owner | `signalgatingprotocol` |
| Repository | `python-sdk` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Protect the GitHub `pypi` environment with required reviewers. The workflow
needs no long-lived PyPI token.

## Release checklist

1. Start from a clean, current `main` checkout.
2. Finalize `CHANGELOG.md`: move unreleased entries into
   `## <version> - YYYY-MM-DD`. For the first release, replace
   `## 0.1.0 - release candidate` with the release date.
3. Update the version in both `pyproject.toml` and
   `src/signal_gating/__init__.py`.
4. Run the same checks as CI:

   ```bash
   RELEASE_VERSION=0.1.0
   python -m pip install -e ".[dev]"
   ruff check .
   mypy src/
   pytest -q
   python -m pip install build twine
   python -m build
   python -m twine check dist/*
   python -m pip install --force-reinstall dist/*.whl
   python scripts/check_release_version.py --tag "v${RELEASE_VERSION}"
   ```

5. Merge the release-preparation pull request and wait for every `main` check
   to pass.
6. In GitHub, create a release from the current `main` commit. Use tag
   `v<version>` (for example, `v0.1.0`) and paste the matching changelog section
   into the release notes.
7. Publish the GitHub release, approve the `pypi` environment deployment, and
   wait for the `release` workflow to finish.

## Verify the published release

Use a clean virtual environment so a local checkout cannot mask packaging
errors:

```bash
RELEASE_CHECK_DIR="$(mktemp -d)"
python -m venv "${RELEASE_CHECK_DIR}"
"${RELEASE_CHECK_DIR}/bin/python" -m pip install --upgrade pip
"${RELEASE_CHECK_DIR}/bin/python" -m pip install signal-gating==0.1.0
"${RELEASE_CHECK_DIR}/bin/python" -c \
  "import signal_gating; print(signal_gating.__version__)"
```

The final command must print the released version. PyPI files are immutable;
if a published artifact is wrong, fix it and publish a new patch version rather
than trying to replace the existing file.
