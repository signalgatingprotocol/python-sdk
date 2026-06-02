# Mesh Event Recording

- **Date:** 2026-06-02
- **Status:** Implemented in SDK, ready for docs/spec alignment
- **Repo:** `signalgatingprotocol/python-sdk`
- **Scope:** Expand trajectories from edge-hop receipts to structured mesh execution events, without changing the signal wire format.

## Problem

`TrajectoryRecorder` originally attached through `mesh.intercept(...)`, so it only saw connected agent-to-agent hops. Core orchestration APIs such as `inject`, `request`, `workflow`, `scatter`, `race`, `publish`, and `call_tool` send directly through agent inboxes and outboxes. Those paths can complete successfully while leaving no receipt trail.

Production agent systems need durable evidence of direct orchestration, especially tool calls and request/reply workflows. This is also the substrate needed before integrating with external agent runtimes such as Claude Agent SDK sessions, hooks, permissions, MCP, subagents, and OpenTelemetry.

## Goal

Add a central mesh event boundary:

```python
recorder = TrajectoryRecorder()
mesh.record(recorder)
```

The recorder captures signal-carrying mesh events as tamper-evident `Receipt` objects and exports them as JSONL. Existing `mesh.intercept(recorder)` remains supported for legacy edge-hop-only capture.

## Non-goals

- No full workflow/session replay engine. `TrajectoryReplayRunner.replay_into(mesh)` re-delivers selected recorded entry signals into a configured mesh; it does not recreate futures, workflow loops, agent state, filesystem state, LLM context, or Claude Agent SDK sessions.
- No MCP adapter yet. SGP tools are native mesh tools and OpenAI-compatible function schemas through `MeshToolProvider`.
- No change to signal wire format: signals still serialize as `{"sgp": 1, "type": ..., "data": ...}`.
- No storage dependency. JSONL remains the durable boundary.

## `MeshEvent`

`MeshEvent` is the in-memory event object emitted by `Mesh.record(...)` sinks:

```python
@dataclass
class MeshEvent:
    action: str
    signal: Signal
    source: str
    target: str = ""
    event_kind: str = "mesh"
    timestamp: float = time.time()
    metadata: dict[str, Any] = {}
```

`event_kind="signal"` is used for connected route outcomes. `event_kind="mesh"` is used for direct orchestration events.

## `Receipt`

`Receipt` remains the durable JSONL record. It now includes:

- `event_kind`: broad event namespace, e.g. `mesh` or `signal`.
- `action`: exact operation, e.g. `inject`, `request_sent`, `response_received`, `workflow_step_start`, `tool_call_complete`, `routed`.
- `metadata`: operation-specific JSON-safe context such as topic, tool name, argument names, correlation id, branch, step, or mapper count.
- `wire`: the full signal wire envelope for reconstructing the typed signal.

The digest covers the event fields, payload projection, and wire envelope. Legacy hop receipts without event fields still verify when they carry the default `event_kind="signal"`, `action="hop"`, and empty metadata.

## Recorded Actions

Direct mesh operations:

- `inject`: `mesh -> agent`
- `request_sent`: `mesh -> target`
- `response_received`: `target -> mesh`
- `scatter_sent`: `mesh -> target`
- `scatter_response`: `target -> mesh`
- `race_sent`: `mesh -> target`
- `race_response`: `target -> mesh`
- `race_winner`: winning response
- `published`: `mesh -> subscriber`, with `topic`
- `workflow_step_start`: `mesh -> step`
- `workflow_step_complete`: `step -> mesh`
- `map_reduce_reduce_start`
- `map_reduce_reduce_complete`
- `branch_selected`
- `tool_call_start`
- `tool_call_complete`

Connected route outcomes:

- `routed`: `source -> target`
- `edge_rejected`
- `intercepted`

## API

```python
mesh.record(recorder)
mesh.record_events(recorder)  # explicit alias
mesh.event_sink_errors        # best-effort sink failure count
```

If a sink exposes `record_event(event)`, that method is registered. Otherwise the sink object itself is called with the `MeshEvent`.

Sinks are best-effort observers. A recorder or exporter failure increments `event_sink_errors` and does not block delivery.

## Delivery Replay

`TrajectoryReplayRunner` is the first execution replay layer:

```python
runner = TrajectoryReplayRunner.from_jsonl("runs.jsonl")
result = await runner.replay_into(mesh)
```

It re-delivers recorded entry events into a fresh mesh:

- `inject`
- `request_sent`
- `scatter_sent`
- `race_sent`
- `published`

It skips audit/control events such as `response_received`, `workflow_step_start`, `workflow_step_complete`, `tool_call_complete`, `race_winner`, and connected route outcomes. Replayed deliveries are recorded as `replay_delivered` events when the target mesh has event sinks attached; metadata points back to the original action, source, and signal id.

This is not full session resume. It does not recreate pending futures, workflow loops, or Claude Agent SDK conversation sessions. That future layer should build on the same event log with explicit run/session ids and idempotency keys.

Claude Agent SDK resume must remain a separate integration boundary: store and pass the Claude `session_id`, working-directory/project key, optional `SessionStore`, and subagent `agentId`/subpath explicitly. SGP receipts may correlate to those IDs, but they do not replace SDK transcript persistence.

## Success Criteria

1. A workflow with no `mesh.connect(...)` records requests, responses, and workflow step events.
2. `mesh.call_tool(...)` records tool call and result events without requiring a connected edge.
3. `scatter`, `race`, `publish`, and `inject` each record direct delivery events.
4. Connected routes are recorded through `mesh.record(...)`, not only through interceptors.
5. JSONL receipts remain verifiable and reconstruct captured typed signals through `TrajectoryRecorder.replay()`.
6. `TrajectoryReplayRunner.replay_into(mesh)` re-delivers recorded entry events into a fresh mesh and reports attempted, delivered, skipped, and missing-target counts.
