# Changelog

This file records user-visible changes to the Signal Gating Python SDK. The
project follows [Semantic Versioning](https://semver.org/); APIs outside the
documented stable core may still change during the `0.x` series.

## Unreleased

No user-visible changes yet.

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

### Stability

- Imports from `signal_gating.core` are the compatibility-focused surface for
  the `0.1.x` line.
- Package-root imports remain compatible throughout `0.1.x`.
- Advanced orchestration modules remain alpha and may change before `1.0`.

### Requirements

- Python 3.10 through 3.14.
- Pydantic 2 or newer.
