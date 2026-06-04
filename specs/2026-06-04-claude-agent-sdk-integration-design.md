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
4. Record direct Claude SDK lifecycle events through `mesh.record(...)` when a caller uses `ClaudeAgentSDKRunner`.
5. Keep Claude lifecycle receipts audit-only and non-replayable by `TrajectoryReplayRunner`.

## Non-goals

- No Claude transcript persistence in SGP payloads.
- No claim that SGP replay resumes Claude sessions.
- No MCP adapter from SGP tools yet.
- No generic runtime-provider framework.
- No dependency on `claude-agent-sdk` for importing `signal_gating` or replaying old receipts.

## API

Primary mesh-agent path:

```python
from signal_gating import ClaudeAgent, ClaudeAgentRunSignal

claude = ClaudeAgent(
    "claude",
    allowed_tools=["Read", "Glob", "Grep"],
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

Helpers:

- `mcp_tool_name(server, tool)` returns `mcp__<server>__<tool>`.
- `claude_options(...)` lazily builds `claude_agent_sdk.ClaudeAgentOptions`.

## Recorded Claude Actions

`ClaudeAgentSDKRunner` records events with `event_kind="claude_agent_sdk"`:

- `claude_query_start`
- `claude_mcp_init`
- `claude_tool_use`
- `claude_tool_result`
- `claude_result`

These actions are intentionally absent from `TrajectoryReplayRunner.replayable_actions`.
They are evidence for audit, evals, and session correlation, not execution replay.

## Metadata Rules

Allowed metadata:

- Tool names, MCP server names, MCP status, MCP tool names.
- Tool call IDs and tool input key names.
- Permission mode, denied tool names, denial counts.
- Claude `session_id`.

Disallowed metadata:

- Raw tool inputs.
- MCP environment values.
- Raw Claude SDK message objects.
- Session transcript entries.

## Verification

The pilot is verified by fake-SDK tests that do not import or call the real Claude
SDK:

- root import hygiene: `import signal_gating` does not import `claude_agent_sdk`.
- stable signal wire names and round trips.
- option shaping for tools, permission mode, MCP servers, resume, and continue.
- mesh receipts for `ClaudeAgent` result/tool signals.
- sanitized direct-runner lifecycle receipts.
- replay boundary: Claude lifecycle receipts are skipped as `action_not_replayable`.
