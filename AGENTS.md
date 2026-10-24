# Python SDK maintenance

SGP is a typed signal-routing SDK. Development is focused on maintenance and
verified correctness. Preserve the compatibility API in `signal_gating.core`.
Read `README.md` for public behavior and `SECURITY.md` for the mediation boundary.
For CI repair or recurring cleanup, use `.github/skills/sdk-maintenance/SKILL.md`.

## Working rules

- Reuse existing primitives and standard tooling. Delete duplication before
  introducing abstractions. Prefer a smaller complete change.
- Before fixing a bug, reproduce it. Turn recurring failures into regression
  tests so later runs can learn from executable evidence.
- Check recent PRs and open work before starting. Do not repeat a completed or
  already proposed change.
- Preserve unrelated changes. Do not expand the framework without the outcome
  evidence required in the README.
- Synthetic fixtures establish fixture behavior. They do not establish model
  quality, production savings, or security protection.
- Never print secrets or add agent attribution to commits or PRs. Outbound email
  is human-only. Agents may draft it but may never transmit or schedule it.

## Validation

Install `python -m pip install -e ".[dev]" build twine`, then run:

```bash
ruff check .
mypy src/
pytest -q
python -m build
python -m twine check dist/*
```

CI tests Python 3.10 through 3.14 and smoke-tests the built wheel. Automation
configuration and operating instructions are in `docs/automation.md`.
