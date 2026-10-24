# Changelog

This file records user-visible changes to the Signal Gating Python SDK. The
project follows [Semantic Versioning](https://semver.org/); APIs outside the
documented stable core may still change during the `0.x` series.

## Unreleased

## 0.1.0 - release candidate

The first public alpha establishes the SDK's signal-routing model and a stable
integration surface for early adopters.

### Added

- Typed, immutable signals with metadata, priority, correlation, and evolution.
- Composable gates for filtering, transformation, deduplication, timing,
  resilience, batching, and conditional routing.
- Managed agents and meshes with lifecycle hooks, request/reply, content-based
  routing, supervision, graceful draining, and observable delivery receipts.
- Trajectory recording and replay, durable recovery, agent pools, teams,
  taskboards, scripted workflows, and focused improvement loops.
- Optional OpenAI-compatible LLM agents and OpenTelemetry export integrations.
- A deterministic incident-triage example that demonstrates 75% handler-load
  reduction while preserving every unique critical incident in its fixture,
  runnable from an installed package with `signal-gating-demo`.
- Fail-closed tool authorization (`signal_gating.security`): per-tool risk
  classification (READ / DRAFT / SEND / DESTRUCTIVE, UNKNOWN gates as
  DESTRUCTIVE), a default-deny policy engine with approver callbacks,
  append-only audit trails, and dry-run mode. `PolicyEngine.as_gate()`
  wires enforcement directly into SGP gate composition.
- Jev risk classifier (`JevRiskClassifier`): an optional decision-model
  classifier for tool risk. It reads `TYPESAFE_API_KEY`, returns calibrated
  probabilities as typed results, and fails closed: below 0.7 confidence, or
  on any error, the tool is UNKNOWN and gates as DESTRUCTIVE. It runs at
  registration time only, never per call; `fallback_to_heuristic=True`
  opts into the regex classifier when Jev is unreachable.
- MCP annotation distrust (`audit_tool`, `scan_source`): verifies server
  self-declared hints against observed behavior; fails tools where the two
  do not match.
- Policy as code (`signal_gating.policy`): declarative YAML policies with
  first-match-wins rules, versioning, and dry-run validation.
- Agent Skills compatibility (`signal_gating.skills`): export and load
  skills per the open standard, with risk classification on install.
  Ships the `sgp-control` skill.
- Executable attack scenarios (`docs/attack-scenarios.md`): five documented
  attack classes, all blocked by default policy (5/5 verified).
- Benchmarks: 2.3us p50 gating overhead per evaluation; 5/5 attack block rate.
- `ToolSpec.risk` field and `risk=` parameter on `Agent.tool()`: explicit
  risk declarations at registration time; undeclared tools default to UNKNOWN.
- `SECURITY.md`: supported versions, private vulnerability reporting, and
  the security scope of the SDK.

### Fixed

- Dead-letter replay now retains the failed signal and all later signals when
  a destination rejects delivery, so a partial replay cannot discard work.
- The security layer now passes the project's own quality gates in strict
  mode (`ruff check .` and `mypy src/` are green on `main`), and the `dev`
  extra includes `types-PyYAML` so the policy loader type-checks. No runtime
  behavior changed.

### Stability

- Imports from `signal_gating.core` are the compatibility-focused surface for
  the `0.1.x` line.
- Package-root imports remain compatible throughout `0.1.x`.
- Advanced orchestration modules remain alpha and may change before `1.0`.

### Requirements

- Python 3.10 through 3.14.
- Pydantic 2 or newer.
