import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from signal_gating import (
    Agent,
    ClaudeAgent,
    ClaudeAgentResultSignal,
    ClaudeAgentRunSignal,
    ClaudeAgentSDKRunner,
    ClaudePermissionDecisionSignal,
    ClaudeToolEventSignal,
    Mesh,
    Signal,
    TrajectoryRecorder,
    TrajectoryReplayRunner,
    mcp_tool_name,
)


class _SystemMessage:
    def __init__(self, session_id: str) -> None:
        self.data = {"session_id": session_id}


class _ToolUseBlock:
    type = "tool_use"
    name = "mcp__docs__search"
    id = "tool-1"


class _AssistantMessage:
    def __init__(self) -> None:
        self.content = [_ToolUseBlock()]
        self.parent_tool_use_id = "parent-tool"


class _ResultMessage:
    def __init__(self, session_id: str, result: str = "done") -> None:
        self.session_id = session_id
        self.result = result
        self.subtype = "success"
        self.total_cost_usd = 0.0123


class _OptionsFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return kwargs


def _query_with(messages: list[object], calls: list[dict[str, Any]]) -> Any:
    async def query(*, prompt: str, options: Any | None = None) -> Any:
        calls.append({"prompt": prompt, "options": options})
        for message in messages:
            yield message

    return query


class _RunnerOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _capture(agent: ClaudeAgent) -> list[Signal]:
    emitted: list[Signal] = []

    async def capture(signal: Signal) -> None:
        emitted.append(signal)

    agent.emit = capture  # type: ignore[method-assign]
    return emitted


async def test_claude_agent_shapes_options_and_emits_session_result_and_tool_event() -> None:
    calls: list[dict[str, Any]] = []
    options = _OptionsFactory()
    agent = ClaudeAgent(
        "claude",
        query_fn=_query_with([
            _SystemMessage("sess-1"),
            _AssistantMessage(),
            _ResultMessage("sess-1", "ship it"),
        ], calls),
        options_factory=options,
        model="claude-sonnet-4-5",
        allowed_tools=["Read"],
        disallowed_tools=["Bash(rm *)"],
        permission_mode="acceptEdits",
        mcp_servers={"docs": {"type": "http", "url": "https://code.claude.com/docs/mcp"}},
        max_turns=5,
    )
    emitted = _capture(agent)
    run = ClaudeAgentRunSignal(
        prompt="review the SDK",
        session_id="sess-0",
        allowed_tools=["mcp__docs__search"],
    )

    await agent._dispatch(run)

    assert calls == [{"prompt": "review the SDK", "options": options.calls[0]}]
    assert options.calls[0]["model"] == "claude-sonnet-4-5"
    assert options.calls[0]["allowed_tools"] == ["mcp__docs__search"]
    assert options.calls[0]["disallowed_tools"] == ["Bash(rm *)"]
    assert options.calls[0]["permission_mode"] == "acceptEdits"
    assert options.calls[0]["mcp_servers"] == {
        "docs": {"type": "http", "url": "https://code.claude.com/docs/mcp"}
    }
    assert options.calls[0]["max_turns"] == 5
    assert options.calls[0]["resume"] == "sess-0"
    assert "continue_conversation" not in options.calls[0]

    assert isinstance(emitted[0], ClaudeToolEventSignal)
    assert emitted[0].session_id == "sess-1"
    assert emitted[0].tool_name == "mcp__docs__search"
    assert emitted[0].mcp_server == "docs"
    assert emitted[0].tool_call_id == "tool-1"
    assert emitted[0].parent_tool_use_id == "parent-tool"

    assert isinstance(emitted[1], ClaudeAgentResultSignal)
    assert emitted[1].text == "ship it"
    assert emitted[1].session_id == "sess-1"
    assert emitted[1].subtype == "success"
    assert emitted[1].total_cost_usd == 0.0123
    assert emitted[1].message_count == 3
    assert emitted[1].allowed_tools == ["mcp__docs__search"]
    assert emitted[1].disallowed_tools == ["Bash(rm *)"]
    assert emitted[1].permission_mode == "acceptEdits"
    assert emitted[1].mcp_servers == ["docs"]
    assert emitted[1].resumed_from_session_id == "sess-0"


async def test_claude_agent_continue_conversation_when_no_resume_session() -> None:
    calls: list[dict[str, Any]] = []
    options = _OptionsFactory()
    agent = ClaudeAgent(
        "claude",
        query_fn=_query_with([_ResultMessage("sess-2")], calls),
        options_factory=options,
        continue_conversation=True,
    )
    emitted = _capture(agent)

    await agent._dispatch(ClaudeAgentRunSignal(prompt="continue"))

    assert options.calls[0]["continue_conversation"] is True
    assert "resume" not in options.calls[0]
    assert isinstance(emitted[0], ClaudeAgentResultSignal)
    assert emitted[0].continued is True


async def test_claude_agent_result_and_tool_events_are_receipted_by_mesh_record() -> None:
    calls: list[dict[str, Any]] = []
    options = _OptionsFactory()
    claude = ClaudeAgent(
        "claude",
        query_fn=_query_with([
            _AssistantMessage(),
            _ResultMessage("sess-3", "answer"),
        ], calls),
        options_factory=options,
    )
    sink = Agent("sink")
    recorder = TrajectoryRecorder()
    seen: list[ClaudeAgentResultSignal] = []
    done = asyncio.Event()

    @sink.on(ClaudeAgentResultSignal)
    async def collect(signal: ClaudeAgentResultSignal) -> None:
        seen.append(signal)
        done.set()

    mesh = Mesh([claude, sink])
    mesh.record(recorder)
    mesh.connect(claude, sink)
    run = ClaudeAgentRunSignal(prompt="make receipts")

    async with mesh:
        await mesh.inject(claude, run)
        await asyncio.wait_for(done.wait(), timeout=3.0)

    assert seen[0].session_id == "sess-3"
    result_receipts = [
        receipt for receipt in recorder.receipts
        if receipt.signal_type == "sgp.integrations.claude.result.v1"
    ]
    tool_receipts = [
        receipt for receipt in recorder.receipts
        if receipt.signal_type == "sgp.integrations.claude.tool_event.v1"
    ]
    assert len(result_receipts) == 1
    assert result_receipts[0].payload["session_id"] == "sess-3"
    assert result_receipts[0].payload["text"] == "answer"
    assert len(tool_receipts) == 1
    assert tool_receipts[0].payload["mcp_server"] == "docs"
    assert all(receipt.verify() for receipt in result_receipts + tool_receipts)


def test_claude_signals_are_stable_wire_types() -> None:
    run = ClaudeAgentRunSignal(prompt="hello", allowed_tools=["Read"])
    restored = Signal.from_wire(run.to_wire())

    assert isinstance(restored, ClaudeAgentRunSignal)
    assert restored == run
    assert run.wire_type() == "sgp.integrations.claude.run.v1"
    assert ClaudeAgentResultSignal().wire_type() == "sgp.integrations.claude.result.v1"
    assert ClaudeToolEventSignal(event="tool_use", tool_name="Read").wire_type() == (
        "sgp.integrations.claude.tool_event.v1"
    )
    decision = ClaudePermissionDecisionSignal(tool_name="Bash", decision="denied")
    assert decision.wire_type() == "sgp.integrations.claude.permission_decision.v1"


def test_mcp_tool_name_formats_claude_tool_names() -> None:
    assert mcp_tool_name("filesystem", "read_file") == "mcp__filesystem__read_file"


async def test_claude_runner_records_sanitized_audit_only_receipts(tmp_path: Path) -> None:
    class _MCPSystemMessage:
        data = {
            "session_id": "sess-run",
            "mcp_servers": [
                {
                    "name": "filesystem",
                    "status": "connected",
                    "tools": ["mcp__filesystem__read_file"],
                },
                {"name": "db", "status": "failed", "tools": []},
            ],
        }

    class _DeniedResult(_ResultMessage):
        permission_denials = [{"tool_name": "Bash", "reason": "blocked"}]

    calls: list[dict[str, Any]] = []
    recorder = TrajectoryRecorder()
    mesh = Mesh()
    mesh.record(recorder)
    runner = ClaudeAgentSDKRunner(
        query=_query_with([
            _MCPSystemMessage(),
            _AssistantMessage(),
            _DeniedResult("sess-run", "runner result"),
        ], calls),
        options_factory=_RunnerOptions,
    )
    mcp_servers = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env": {"TOKEN": "secret"},
        }
    }

    async with mesh:
        result = await runner.run(
            "Inspect auth",
            mesh=mesh,
            cwd=tmp_path,
            allowed_tools=["Read", "mcp__filesystem__read_file"],
            disallowed_tools=["Bash(rm *)"],
            permission_mode="dontAsk",
            mcp_servers=mcp_servers,
            strict_mcp_config=True,
        )

    assert result.session_id == "sess-run"
    assert result.result == "runner result"
    options = calls[0]["options"]
    assert options.allowed_tools == ["Read", "mcp__filesystem__read_file"]
    assert options.disallowed_tools == ["Bash(rm *)"]
    assert options.permission_mode == "dontAsk"
    assert options.mcp_servers == mcp_servers
    assert options.strict_mcp_config is True
    assert str(options.cwd) == str(tmp_path)

    assert [receipt.action for receipt in recorder.receipts] == [
        "claude_query_start",
        "claude_mcp_init",
        "claude_tool_use",
        "claude_result",
    ]
    assert {receipt.event_kind for receipt in recorder.receipts} == {"claude_agent_sdk"}
    assert all(receipt.verify() for receipt in recorder.receipts)

    start, mcp, tool, final = recorder.receipts
    assert start.metadata["allowed_tools"] == ["Read", "mcp__filesystem__read_file"]
    assert start.metadata["disallowed_tools"] == ["Bash(rm *)"]
    assert start.metadata["permission_mode"] == "dontAsk"
    assert start.metadata["mcp_server_names"] == ["filesystem"]
    assert "TOKEN" not in str(start.metadata)
    assert "secret" not in str(start.metadata)

    assert mcp.metadata["claude_session_id"] == "sess-run"
    assert mcp.metadata["mcp_server_statuses"] == {
        "filesystem": "connected",
        "db": "failed",
    }
    assert mcp.metadata["mcp_tool_names"] == ["mcp__filesystem__read_file"]
    assert mcp.metadata["failed_mcp_servers"] == ["db"]

    assert tool.metadata["tool_name"] == "mcp__docs__search"
    assert tool.metadata["tool_use_id"] == "tool-1"
    assert tool.metadata["tool_input_keys"] == []
    assert tool.metadata["tool_allowed"] is False
    assert "tool_input" not in tool.metadata

    assert final.metadata["claude_session_id"] == "sess-run"
    assert final.metadata["permission_denial_count"] == 1
    assert final.metadata["permission_denied_tools"] == ["Bash"]


async def test_claude_runner_receipts_are_not_replayable(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    recorder = TrajectoryRecorder()
    mesh = Mesh()
    mesh.record(recorder)
    runner = ClaudeAgentSDKRunner(
        query=_query_with([_ResultMessage("sess-replay", "audit")], calls),
        options_factory=_RunnerOptions,
    )

    async with mesh:
        await runner.run("Inspect", mesh=mesh, cwd=tmp_path, allowed_tools=["Read"])

    replay = TrajectoryReplayRunner.from_recorder(recorder)
    result = await replay.replay_into(Mesh())

    assert replay.replayable_receipts() == []
    assert result.attempted == 0
    assert result.delivered == 0
    assert result.skipped == len(recorder.receipts)
    assert {delivery.reason for delivery in result.deliveries} == {"action_not_replayable"}
    assert len(calls) == 1


def test_claude_agent_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal_gating.claude as claude_module

    def missing_import(name: str) -> object:
        if name == "claude_agent_sdk":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr(claude_module, "import_module", missing_import)

    agent = ClaudeAgent("claude")
    with pytest.raises(ImportError, match=r"signal-gating\[claude\]"):
        agent._sdk_bindings()


def test_import_does_not_pull_claude_agent_sdk() -> None:
    code = "import sys, signal_gating; assert 'claude_agent_sdk' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_exports_from_package_root_and_integration_module() -> None:
    import signal_gating
    from signal_gating.integrations import claude

    assert hasattr(signal_gating, "ClaudeAgent")
    assert hasattr(claude, "ClaudeAgent")
    assert "ClaudeAgent" in signal_gating.__all__
