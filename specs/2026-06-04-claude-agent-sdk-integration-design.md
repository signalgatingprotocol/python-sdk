# Claude Agent SDK Integration Boundary

- **Date:** 2026-06-04
- **Status:** Pilot implemented in SDK
- **Repo:** `signalgatingprotocol/python-sdk`
- **Scope:** Correlate Claude Agent SDK runs with SGP signals, mesh events, and trajectory receipts without making SGP responsible for Claude transcript persistence.

## Problem

Claude Agent SDK provides production agent-loop features: built-in filesystem and
shell tools, MCP servers, permissions, hooks, subagents, sessions, and optional
external session storage. SGP already provides typed signal flow, gates, mesh
routing, tracing, and tamper-evident trajectory receipts.

Treating Claude as another OpenAI-compatible chat client would erase important
runtime concepts such as `session_id`, MCP tool names, permission decisions, and
subagent/tool lineage. Treating SGP replay as Claude replay would also be wrong:
SGP replays SGP signals, not Claude's transcript, filesystem, tools, or session
store.

## Goals

1. Provide a lazy optional `claude` extra for users who want Claude Agent SDK integration.
2. Expose JSON-safe SGP signals for Claude runs, results, tool events, and permission decisions.
3. Preserve Claude `session_id` on SGP result signals and trajectory receipts.
4. Record direct Claude SDK lifecycle events through `mesh.record(...)` when a caller uses `ClaudeAgentSDKRunner` or `ClaudeAgentSDKClientSession`.
5. Compile SGP gates into Claude tool policy callbacks without storing raw tool inputs.
6. Keep Claude lifecycle receipts audit-only and non-replayable by `TrajectoryReplayRunner`.

## Non-goals

- No Claude transcript persistence in SGP payloads.
- No claim that SGP replay resumes Claude sessions.
- No MCP adapter from SGP tools yet.
- No generic runtime-provider framework.
- No dependency on `claude-agent-sdk` for importing `signal_gating` or replaying old receipts.

## API

Primary mesh-agent path:

```python
from signal_gating import ClaudeAgent, ClaudeAgentRunSignal, ClaudeToolPolicy

claude = ClaudeAgent(
    "claude",
    tool_policy=ClaudeToolPolicy.read_only(),
    permission_mode="acceptEdits",
)
await mesh.inject(claude, ClaudeAgentRunSignal(prompt="Review this module"))
```

Direct audit-only path:

```python
from signal_gating import ClaudeAgentSDKRunner

runner = ClaudeAgentSDKRunner()
result = await runner.run(
    "Inspect auth",
    mesh=mesh,
    allowed_tools=["Read", "mcp__filesystem__read_file"],
    mcp_servers={"filesystem": {"command": "npx", "args": ["..."]}},
)
```

Continuous client-session path:

```python
from signal_gating import ClaudeAgentSDKClientSession

async with ClaudeAgentSDKClientSession(
    allowed_tools=["Read", "mcp__filesystem__read_file"],
    permission_mode="dontAsk",
    mcp_servers={"filesystem": {"command": "npx", "args": ["..."]}},
) as session:
    result = await session.run_turn("Inspect auth", mesh=mesh, session_id="main")
    await session.set_permission_mode("acceptEdits", mesh=mesh)
```

Helpers:

- `mcp_tool_name(server, tool)` returns `mcp__<server>__<tool>`.
- `claude_options(...)` lazily builds `claude_agent_sdk.ClaudeAgentOptions`.
- `ClaudeToolPolicy` compiles static tool lists and an optional SGP `Gate` into
  Claude `allowed_tools`, `disallowed_tools`, and `can_use_tool`.
- `ClaudeToolRequestSignal` is the sanitized gate input for a Claude tool
  permission request. It includes tool name, MCP server name, and input key
  names plus safe SDK context labels/reasons, but not raw tool input values.
- `ClaudeMeshMCPAdapter` exposes mesh tools through MCP-shaped `initialize`,
  `tools/list`, and `tools/call` JSON-RPC handlers.
- `ClaudeMeshMCPStdioServer` serves the adapter over newline-delimited stdio
  JSON-RPC and writes only MCP response messages to stdout.
- `ClaudeMeshMCPHTTPApp` serves the adapter as a dependency-free ASGI app for
  MCP Streamable HTTP POST requests that return `application/json`. It creates a
  secure `Mcp-Session-Id` on `initialize`, requires that session on subsequent
  POST/GET/DELETE requests, rejects unsupported protocol versions and forbidden
  origins, returns `202` for accepted notifications and JSON-RPC responses,
  returns `405` for unsupported GET SSE streams, and supports `DELETE` session
  termination with `204 No Content`.
- `ClaudeMCPHTTPAuthorizationContext`, `ClaudeMCPHTTPAuthorizationSignal`,
  `ClaudeMCPHTTPAuthorizationResult`, and `ClaudeMCPHTTPAuthorizeFn` provide an
  optional bearer-token authorization seam for HTTP MCP deployments. When
  `authorize_http` is configured, every HTTP request must include
  `Authorization: Bearer ...`; failed validation returns `401` or `403`, and
  each `Mcp-Session-Id` is bound to the returned principal plus canonical
  audience, resource, and normalized scope set. If no principal is returned, the
  session falls back to a SHA-256 token fingerprint without storing the raw
  token. URI query-string access tokens are rejected. The hook is the
  resource-server validation boundary for issuer, expiry, audience/resource,
  and scope checks. With `mesh.record(...)`, every auth decision is receipted as
  `event_kind="claude_mcp_http"` without raw token, principal, session, request
  body, or tool-argument values.
- `ClaudeMCPBearerTokenValidator` / `ClaudeMCPJWTBearerAuthorizer` provide a
  ready-to-mount authorization hook for already-verified OAuth/JWT-style access
  token claims. They validate issuer, expiry, not-before, audience, MCP
  resource, principal, and required scopes, map invalid tokens to
  `401 invalid_token`, map valid-but-under-scoped tokens to
  `403 insufficient_scope`, and never include raw bearer tokens in denial
  messages or challenges. `ClaudeMCPBearerTokenValidator.pyjwt(...)` lazy-loads
  PyJWT/JWKS support from the optional `signal-gating[auth]` extra and enforces
  JWT access-token `typ` headers of `at+jwt` or `application/at+jwt` by default.
- `ClaudeMCPProtectedResourceMetadata` and
  `protected_resource_metadata_url(resource)` serve OAuth Protected Resource
  Metadata for HTTP MCP deployments. The metadata document includes `resource`,
  non-empty `authorization_servers`, `bearer_methods_supported=["header"]`, and
  optional scopes/name/docs/extensions, and is served at the RFC 9728
  well-known path derived from the resource URL. `ClaudeMeshMCPHTTPApp`
  advertises the derived URL in `WWW-Authenticate` when metadata is configured,
  including explicit bearer-error challenges returned by authorization hooks.
- `signal-gating-mcp module:factory` loads a `Mesh` or
  `ClaudeMeshMCPAdapter` and serves it over stdio. `Mesh` factories are owned by
  the CLI lifecycle; adapter factories are served as supplied so custom adapter
  factories can own their own lifecycle. Factory, lifecycle, and tool stdout go
  to stderr to preserve MCP JSON-RPC purity on stdout.
- `signal-gating-receipts auth <runs.jsonl> [--otel]` summarizes Claude MCP
  HTTP auth receipts as verified aggregate JSON. It verifies digests by default,
  supports `--action` filters, and auto-selects
  `event_kind="claude_mcp_http"` plus
  `sgp.integrations.claude.mcp_http_authorization.v1`. With `--otel`, it
  exports the same aggregate metrics batch through OpenTelemetry; it must not
  export raw receipts, payloads, wire envelopes, metadata, auth material, or
  tool arguments. `summary` provides the generic form with repeatable
  `--event-kind`, `--signal-type`, and `--action`.

## Recorded Claude Actions

`ClaudeAgentSDKRunner` records events with `event_kind="claude_agent_sdk"`:

- `claude_query_start`
- `claude_mcp_init`
- `claude_tool_use`
- `claude_tool_result`
- `claude_result`

These actions are intentionally absent from `TrajectoryReplayRunner.replayable_actions`.
They are evidence for audit, evals, and session correlation, not execution replay.

`ClaudeAgentSDKClientSession` records the same parser-backed actions plus client
control actions:

- `claude_client_connect`
- `claude_client_query_start`
- `claude_client_disconnect`
- `claude_permission_mode_changed`
- `claude_permission_decision`
- `claude_model_set`
- `claude_mcp_status`
- `claude_mcp_reconnect`
- `claude_mcp_toggle`
- `claude_interrupt`
- `claude_task_stop`
- `claude_rewind_files`
- `claude_stream_event`
- `claude_permission_decision`

`ClaudeMeshMCPHTTPApp` records HTTP MCP auth receipts with
`event_kind="claude_mcp_http"`:

- `claude_mcp_http_auth_allowed`
- `claude_mcp_http_auth_challenged`
- `claude_mcp_http_auth_denied`
- `claude_mcp_http_auth_session_mismatch`
- `claude_mcp_http_auth_query_token_rejected`

The typed `ClaudeMCPHTTPAuthorizationSignal` carries outcome, status, method,
path, reason, bearer-token presence, principal/session SHA-256 hashes,
audience/resource presence, scope count, identity binding kind, and the
protected-resource-metadata advertised flag. It intentionally leaves raw bearer
tokens, `Authorization` header values, raw principals, full `Mcp-Session-Id`
values, request bodies, and tool arguments out of payloads, metadata, and JSONL
exports.

### Auth Metrics From Receipts

Consumers may derive operational auth metrics with
`recorder.filter_receipts(event_kinds="claude_mcp_http")`, or export only that
safe slice with `recorder.export_jsonl(..., event_kinds="claude_mcp_http",
signal_types="sgp.integrations.claude.mcp_http_authorization.v1", verify=True)`.
Supported aggregation dimensions are `action`, `payload.outcome`,
`payload.status_code`, `payload.reason`, `payload.method`, `payload.path`,
`payload.jsonrpc_method`, `payload.identity_binding_kind`, `payload.scope_count`,
and presence booleans.

The receipt CLI is intentionally metrics-only. Its output includes aggregate
counts for actions, outcomes, status codes, reasons, methods, paths, JSON-RPC
methods, identity-binding kinds, scope counts, and presence booleans. It must
not emit raw receipt rows or raw payload values outside these approved
low-cardinality dimensions.

`auth --otel` is constrained to the same metrics-only shape: loaded/matched
totals, trace count, duration, active filters, verification status, approved
count dimensions, and presence totals. OTel labels must not include raw receipt
rows, raw payload values, bearer tokens, principals, full session IDs, request
bodies, or tool arguments. Raw path labels are omitted by default.
`--otel-include-paths` is the sole opt-in for exporting `counts.paths` as OTel
count dimensions; it requires `--otel`, does not change JSON output, and is
intended only for telemetry backends approved for high-cardinality and
potentially identifying route/path values. `--otel-max-paths` caps exported path
labels, defaults to 100, and folds overflow counts into `sgp.value="__other__"`.

Metrics pipelines must not require raw bearer tokens, `Authorization` header
values, raw principals, full `Mcp-Session-Id` values, request bodies, or tool
arguments. Principal/session hashes may be used for internal cardinality and
mismatch detection, but dashboards should prefer counts/rates over exposing hash
values as labels.

## Metadata Rules

Allowed metadata:

- Tool names, MCP server names, MCP status, MCP tool names.
- Tool call IDs and tool input key names.
- Tool policy decisions and sanitized `ClaudeToolRequestSignal` fields.
- Permission mode, denied tool names, denial counts.
- Claude `session_id`.
- HTTP MCP auth outcome/status/reason, bearer-token presence, principal/session
  hashes, audience/resource presence, scope count, identity binding kind, and
  protected-resource-metadata advertised flag.

Disallowed metadata:

- Raw tool inputs.
- MCP environment values.
- Raw Claude SDK message objects.
- Session transcript entries.
- Raw bearer tokens, `Authorization` header values, raw principals, full
  `Mcp-Session-Id` values, and raw HTTP request bodies.

## Verification

The pilot is verified by fake-SDK tests that do not import or call the real Claude
SDK:

- root import hygiene: `import signal_gating` does not import `claude_agent_sdk`.
- stable signal wire names and round trips.
- option shaping for tools, permission mode, MCP servers, resume, and continue.
- mesh receipts for `ClaudeAgent` result/tool signals.
- sanitized direct-runner lifecycle receipts.
- continuous client-session reuse, controls, permission decisions, and sanitized MCP status receipts.
- SGP-to-Claude tool policy compilation with gate-backed `can_use_tool`.
- MCP-shaped schema and call adapter over `mesh.discover_tools()` and `mesh.call_tool()`.
- stdio transport wrapper for the mesh MCP adapter.
- Streamable HTTP ASGI wrapper for the mesh MCP adapter, including Origin,
  Accept, Content-Type, protocol-version, session, notification, DELETE, and
  non-SSE GET behavior.
- Optional bearer authorization hook for Streamable HTTP, including sanitized
  auth context, `401` / `403` handling, `WWW-Authenticate` challenges, and
  session-to-principal/fingerprint binding without raw token storage or
  query-string token acceptance.
- OAuth Protected Resource Metadata helper and ASGI well-known GET response for
  HTTP MCP authorization-server discovery.
- Sanitized, verifiable HTTP MCP auth receipts for allowed, challenged, denied,
  session-mismatch, and query-token-rejected decisions, including JSONL export,
  typed replay, and tamper rejection.
- `signal-gating-receipts` console entrypoint is packaged.
- CLI `auth` summarizes filtered Claude MCP HTTP auth receipts without leaking
  raw ordinary trajectory payloads or auth material.
- CLI loading verifies receipt digests by default and fails on selected-receipt
  tampering; `--no-verify` is explicit and reflected in output.
- CLI `auth --otel` exports the same verified aggregate metrics batch through
  `OpenTelemetryReceiptMetricsExporter`, prints aggregate JSON on success, and
  never passes raw `Receipt` objects, payloads, wire envelopes, metadata, or auth
  material to OTel. Raw path labels are omitted by default; with
  `--otel-include-paths`, only the bounded `counts.paths` labels selected by
  `--otel-max-paths` are exported and overflow is folded into
  `sgp.value="__other__"`.
- CI runs `scripts/smoke_receipt_metrics_cli.py` against generated auth
  receipts with `signal-gating-receipts auth --otel --otel-include-paths
  --otel-max-paths 1 --pretty`, proving the installed console-script subprocess,
  optional OTel import path, digest verification, aggregate JSON output, and
  bounded path label export execute together. The smoke captures actual OTel
  metric calls as JSONL and asserts `sgp.receipts.count` path labels include the
  retained path plus `sgp.value="__other__"` overflow while excluding the folded
  path value and ordinary trajectory metadata.
- replay boundary: Claude lifecycle receipts are skipped as `action_not_replayable`.
