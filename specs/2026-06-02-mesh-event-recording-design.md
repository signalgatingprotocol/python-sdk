# Mesh Event Recording

- **Date:** 2026-06-02
- **Status:** Implemented and documented
- **Repo:** `signalgatingprotocol/python-sdk`
- **Scope:** Record the complete mesh delivery trust boundary as structured execution events, without changing the signal wire format.

## Problem

The mesh has one delivery trust boundary and two observation mechanisms. Interceptors mediate every delivery performed by `Mesh`: connected, content-routed, and load-balanced routes; direct orchestration sends; correlated request, scatter, and race responses; and trajectory replay. A signal allowed through that boundary is the exact signal delivered, traced, and recorded.

Raw `agent.inbox` writes are deliberately outside this boundary. They bypass mesh delivery interceptors, mesh delivery traces, and receipts. If the receiving agent processes such a signal, it can still emit its own gate and dispatch processing spans through its attached tracer. Callers must use mesh APIs when delivery authorization or audit coverage is required.

`mesh.record(...)` adds the stable, action-specific execution record needed for production audit. It covers both deliveries and orchestration control events, including tool calls and request/reply workflows. The legacy `mesh.intercept(recorder)` attachment remains useful as a generic delivery observer, but it produces `action="hop"` receipts at the interceptor observation point rather than action-specific post-enqueue receipts.

Production agent systems need durable evidence of direct orchestration, especially tool calls and request/reply workflows. This is also the substrate needed before integrating with external agent runtimes such as Claude Agent SDK sessions, hooks, permissions, MCP, subagents, and OpenTelemetry.

## Goal

Add a central mesh event boundary:

```python
recorder = TrajectoryRecorder()
mesh.record(recorder)
```

The recorder captures signal-carrying mesh events as tamper-evident `Receipt` objects and exports them as JSONL. `mesh.record(recorder)` is the preferred stable hook because it preserves operation names and metadata. Existing `mesh.intercept(recorder)` remains supported across the complete mesh-mediated delivery boundary, but records generic hop receipts and does not observe non-delivery control events.

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

`event_kind="signal"` is used for connected route outcomes. `event_kind="mesh"` is used for direct orchestration events. Blocked delivery events retain the namespace of the attempted action.

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
- `scatter_timeout`: aggregate failure with missing targets and response counts
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

Delivery outcomes:

- `routed`: `source -> target`
- `intercepted`: an interceptor returned `None`; metadata identifies the attempted delivery action and blocker
- `edge_rejected`: a route gate returned `None`; metadata identifies the attempted delivery action and gate
- `replay_delivered`: `mesh -> target`, with metadata identifying the original receipt

## API

```python
mesh.record(recorder)
mesh.record_events(recorder)  # explicit alias
mesh.event_sink_errors        # best-effort sink failure count
```

If a sink exposes `record_event(event)`, that method is registered. Otherwise the sink object itself is called with the `MeshEvent`.

Sinks are best-effort observers. A recorder or exporter failure increments `event_sink_errors` and does not block delivery. For successful inbox deliveries, the action-specific receipt is emitted only after the destination enqueue succeeds; correlated responses are recorded only after the waiting future accepts them. A failed enqueue does not emit a success receipt. An interceptor or gate block instead emits an `intercepted` or `edge_rejected` event, respectively.

Interceptors are enforcement hooks, not best-effort event sinks. They run before route gates and destination enqueue. Returning `None` blocks the attempt; returning a signal replaces the signal passed to later interceptors, gates, delivery, tracing, and action-specific recording. Awaited point-to-point operations raise `MeshError` immediately when blocked. Connected and other fire-and-forget routes drop observably, `publish()` continues and counts accepted subscribers, and `race()` continues while an allowed candidate remains.

For compatibility, this remains valid:

```python
mesh.intercept(recorder)
```

That attachment participates in every mesh-mediated delivery path, including correlated responses and replay, but produces generic `event_kind="signal"`, `action="hop"` receipts at its position in the interceptor chain. It is not a substitute for the action-specific, post-delivery `mesh.record(...)` record.

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

It skips audit/control events such as `response_received`, `scatter_timeout`, `workflow_step_start`, `workflow_step_complete`, `tool_call_complete`, `race_winner`, and connected route outcomes. Replay uses the target mesh's current interceptor boundary. Allowed replay is traced and recorded as `replay_delivered` after destination enqueue when event sinks are attached; metadata points back to the original action, source, and signal id. A blocked replay raises `MeshError` and emits an `intercepted` receipt when a record sink is attached.

This is not full session resume. It does not recreate pending futures, workflow loops, or Claude Agent SDK conversation sessions. That future layer should build on the same event log with explicit run/session ids and idempotency keys.

Claude Agent SDK resume must remain a separate integration boundary: store and pass the Claude `session_id`, working-directory/project key, optional `SessionStore`, and subagent `agentId`/subpath explicitly. SGP receipts may correlate to those IDs, but they do not replace SDK transcript persistence.

## Success Criteria

1. Every mesh-mediated delivery, including correlated responses and replay, runs through interceptors; raw inbox writes remain an explicit bypass.
2. A workflow with no `mesh.connect(...)` records requests, responses, and workflow step events.
3. `mesh.call_tool(...)` records tool call and result events without requiring a connected edge.
4. `scatter`, `race`, `publish`, `inject`, and connected routes each emit action-specific events through `mesh.record(...)`.
5. Successful delivery events contain the final transformed signal and occur after destination acceptance; blocked attempts emit `intercepted` or `edge_rejected` instead.
6. JSONL receipts remain verifiable and reconstruct captured typed signals through `TrajectoryRecorder.replay()`.
7. `TrajectoryReplayRunner.replay_into(mesh)` re-delivers recorded entry events through the current mesh boundary and reports attempted, delivered, skipped, and missing-target counts.
