"""Offline Claude Agent SDK mesh example with typed receipts.

This example intentionally injects fake Claude SDK bindings. It demonstrates
the SGP integration contract without requiring credentials, live MCP servers, or
the optional ``claude-agent-sdk`` package.

Run:

    python examples/claude_agent_sdk_mesh.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from signal_gating import (
    Agent,
    ClaudeAgent,
    ClaudeAgentResultSignal,
    ClaudeAgentRunSignal,
    ClaudeToolEventSignal,
    ClaudeToolPolicy,
    Mesh,
    TrajectoryRecorder,
)


class FakeSystemMessage:
    data = {"session_id": "sess-example"}


class FakeToolUse:
    type = "tool_use"
    name = "mcp__docs__search"
    id = "tool-1"
    input = {"query": "signal gating claude integration"}


class FakeAssistantMessage:
    content = [FakeToolUse()]
    parent_tool_use_id = "assistant-1"


class FakeResultMessage:
    session_id = "sess-example"
    result = "Use typed gates around Claude Agent SDK runs."
    subtype = "success"
    total_cost_usd = 0.01


class FakeOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


policy_decisions: list[str] = []


async def fake_query(*, prompt: str, options: Any | None = None) -> Any:
    del prompt
    if isinstance(options, FakeOptions):
        can_use_tool = options.kwargs.get("can_use_tool")
        if can_use_tool is not None:
            decision = await can_use_tool(
                "mcp__docs__search",
                {"query": "signal gating claude integration"},
                object(),
            )
            policy_decisions.append(str(getattr(decision, "behavior", "")))
    for message in [FakeSystemMessage(), FakeAssistantMessage(), FakeResultMessage()]:
        yield message


async def main() -> None:
    claude = ClaudeAgent(
        "claude",
        query_fn=fake_query,
        options_factory=FakeOptions,
        tool_policy=ClaudeToolPolicy.mcp_tools("docs", ["search"]),
        permission_mode="dontAsk",
        mcp_servers={"docs": {"type": "http", "url": "https://code.claude.com/docs"}},
    )
    sink = Agent("sink")
    recorder = TrajectoryRecorder()
    results: list[ClaudeAgentResultSignal] = []
    tool_events: list[ClaudeToolEventSignal] = []
    done = asyncio.Event()

    @sink.on(ClaudeToolEventSignal)
    async def collect_tool_event(signal: ClaudeToolEventSignal) -> None:
        tool_events.append(signal)

    @sink.on(ClaudeAgentResultSignal)
    async def collect_result(signal: ClaudeAgentResultSignal) -> None:
        results.append(signal)
        done.set()

    mesh = Mesh([claude, sink])
    mesh.record(recorder)
    mesh.connect(claude, sink)

    async with mesh:
        await mesh.inject(claude, ClaudeAgentRunSignal(prompt="Review the integration"))
        await asyncio.wait_for(done.wait(), timeout=3.0)

    result = results[0]
    print(f"session_id={result.session_id}")
    print(f"policy_decision={policy_decisions[0]}")
    print(f"tool_events={len(tool_events)}")
    print(f"receipts={len(recorder.receipts)}")
    print(f"all_receipts_verify={all(receipt.verify() for receipt in recorder.receipts)}")
    print(f"RESULT {result.subtype}: {result.text}")
    for event in tool_events:
        print(f"TOOL {event.event} {event.tool_name} input_keys={event.tool_input_keys}")


if __name__ == "__main__":
    asyncio.run(main())
