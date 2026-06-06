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
    ClaudeAgentSDKClientSession,
    ClaudeAgentSDKRunner,
    ClaudeAgentSDKSession,
    ClaudePermissionDecisionSignal,
    ClaudeToolEventSignal,
    ClaudeToolPolicy,
    ClaudeToolRequestSignal,
    Gate,
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


class _FakeClaudeClient:
    def __init__(self, *, options: Any | None = None, responses: list[list[object]]) -> None:
        self.options = options
        self.responses = responses
        self.connected = False
        self.connect_prompts: list[str | None] = []
        self.disconnect_count = 0
        self.queries: list[tuple[str, str]] = []
        self.permission_modes: list[str] = []
        self.models: list[str] = []
        self.interrupt_count = 0
        self.reconnected_servers: list[str] = []
        self.toggled_servers: list[tuple[str, bool]] = []
        self.stopped_tasks: list[str] = []
        self.rewound_messages: list[str] = []

    async def connect(self, prompt: str | None = None) -> None:
        self.connected = True
        self.connect_prompts.append(prompt)

    async def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    async def query(self, prompt: str, *, session_id: str = "default") -> None:
        self.queries.append((prompt, session_id))

    async def receive_response(self) -> Any:
        for message in self.responses.pop(0):
            yield message

    async def set_permission_mode(self, mode: str) -> None:
        self.permission_modes.append(mode)

    async def set_model(self, model: str) -> None:
        self.models.append(model)

    async def interrupt(self) -> None:
        self.interrupt_count += 1

    async def get_mcp_status(self) -> dict[str, Any]:
        return {"filesystem": {"status": "connected", "env": {"TOKEN": "secret"}}}

    async def reconnect_mcp_server(self, server_name: str) -> None:
        self.reconnected_servers.append(server_name)

    async def toggle_mcp_server(self, server_name: str, enabled: bool) -> None:
        self.toggled_servers.append((server_name, enabled))

    async def stop_task(self, task_id: str) -> None:
        self.stopped_tasks.append(task_id)

    async def rewind_files(self, user_message_id: str) -> None:
        self.rewound_messages.append(user_message_id)


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
    request = ClaudeToolRequestSignal(tool_name="Read", tool_input_keys=["path"])
    restored_request = Signal.from_wire(request.to_wire())
    assert isinstance(restored_request, ClaudeToolRequestSignal)
    assert restored_request == request
    assert request.wire_type() == "sgp.integrations.claude.tool_request.v1"


def test_mcp_tool_name_formats_claude_tool_names() -> None:
    assert mcp_tool_name("filesystem", "read_file") == "mcp__filesystem__read_file"


async def test_claude_tool_policy_compiles_gate_to_can_use_tool_without_raw_inputs() -> None:
    seen: list[ClaudeToolRequestSignal] = []

    def has_path(signal: Signal) -> Signal | None:
        assert isinstance(signal, ClaudeToolRequestSignal)
        seen.append(signal)
        return signal if "path" in signal.tool_input_keys else None

    policy = ClaudeToolPolicy(
        gate=Gate(has_path, name="has_path"),
        deny_message="blocked by policy",
    )
    kwargs = policy.claude_kwargs()

    allowed = await kwargs["can_use_tool"](
        "mcp__filesystem__read_file",
        {"path": "/tmp/secret.txt"},
        object(),
    )
    denied = await kwargs["can_use_tool"](
        "mcp__filesystem__write_file",
        {"content": "raw secret"},
        object(),
    )

    assert getattr(allowed, "behavior") == "allow"
    assert getattr(denied, "behavior") == "deny"
    assert getattr(denied, "message") == "blocked by policy"
    assert seen[0].tool_name == "mcp__filesystem__read_file"
    assert seen[0].mcp_server == "filesystem"
    assert seen[0].tool_input_keys == ["path"]
    assert seen[1].tool_input_keys == ["content"]
    assert "/tmp/secret.txt" not in repr(seen)
    assert "raw secret" not in repr(seen)


async def test_claude_tool_policy_disallowed_rule_wins_over_default_allow() -> None:
    policy = ClaudeToolPolicy(
        allowed_tools=["Read"],
        disallowed_tools=["Bash*"],
        default="allowed",
    )
    kwargs = policy.claude_kwargs()

    assert kwargs["allowed_tools"] == ["Read"]
    assert kwargs["disallowed_tools"] == ["Bash*"]
    denied = await kwargs["can_use_tool"]("Bash", {"command": "echo ok"}, object())
    allowed = await kwargs["can_use_tool"]("Write", {"path": "/tmp/file"}, object())

    assert getattr(denied, "behavior") == "deny"
    assert getattr(allowed, "behavior") == "allow"


async def test_claude_tool_policy_gate_can_reject_allowed_tool() -> None:
    policy = ClaudeToolPolicy(
        allowed_tools=["Read"],
        gate=Gate.filter(
            lambda signal: (
                isinstance(signal, ClaudeToolRequestSignal)
                and "path" in signal.tool_input_keys
            ),
            name="has_path",
        ),
    )

    denied = await policy.can_use_tool("Read", {"content": "raw"}, object())
    allowed = await policy.can_use_tool("Read", {"path": "/tmp/ok"}, object())

    assert getattr(denied, "behavior") == "deny"
    assert getattr(allowed, "behavior") == "allow"


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


async def test_claude_sdk_session_reuses_client_and_records_query_receipts(
    tmp_path: Path,
) -> None:
    clients: list[_FakeClaudeClient] = []
    session_store = object()

    def client_factory(*, options: Any | None = None) -> _FakeClaudeClient:
        client = _FakeClaudeClient(
            options=options,
            responses=[
                [
                    _SystemMessage("sess-cont"),
                    _AssistantMessage(),
                    _ResultMessage("sess-cont", "first answer"),
                ],
                [_ResultMessage("sess-cont", "second answer")],
            ],
        )
        clients.append(client)
        return client

    recorder = TrajectoryRecorder()
    mesh = Mesh()
    mesh.record(recorder)
    session = ClaudeAgentSDKClientSession(
        client_factory=client_factory,
        options_factory=_RunnerOptions,
        mesh=mesh,
        allowed_tools=["Read"],
        disallowed_tools=["Bash(rm *)"],
        permission_mode="dontAsk",
        mcp_servers={"filesystem": {"command": "npx", "env": {"TOKEN": "secret"}}},
        cwd=tmp_path,
        session_store=session_store,
        strict_mcp_config=True,
        max_turns=7,
        model="claude-sonnet-4-5",
        system_prompt="Act as a careful reviewer.",
    )

    async with mesh:
        async with session:
            first = await session.query("Inspect risk", mesh=mesh, session_id="main")
            second = await session.query(
                "Continue",
                mesh=mesh,
                session_id="main",
                permission_mode="acceptEdits",
                model="claude-opus-4-5",
            )

    assert first.session_id == "sess-cont"
    assert first.result == "first answer"
    assert first.message_count == 3
    assert second.session_id == "sess-cont"
    assert second.result == "second answer"
    assert len(clients) == 1
    client = clients[0]
    assert client.connect_prompts == [None]
    assert client.disconnect_count == 1
    assert client.queries == [("Inspect risk", "main"), ("Continue", "main")]
    assert client.permission_modes == ["acceptEdits"]
    assert client.models == ["claude-opus-4-5"]

    options = client.options
    assert options is not None
    assert options.allowed_tools == ["Read"]
    assert options.disallowed_tools == ["Bash(rm *)"]
    assert options.permission_mode == "dontAsk"
    assert options.mcp_servers == {
        "filesystem": {"command": "npx", "env": {"TOKEN": "secret"}}
    }
    assert options.cwd == tmp_path
    assert options.session_store is session_store
    assert options.strict_mcp_config is True
    assert options.max_turns == 7
    assert options.model == "claude-sonnet-4-5"
    assert options.system_prompt == "Act as a careful reviewer."

    actions = [receipt.action for receipt in recorder.receipts]
    assert actions == [
        "claude_client_connect",
        "claude_client_query_start",
        "claude_tool_use",
        "claude_result",
        "claude_permission_mode_changed",
        "claude_model_set",
        "claude_client_query_start",
        "claude_result",
        "claude_client_disconnect",
    ]
    assert all(receipt.event_kind == "claude_agent_sdk" for receipt in recorder.receipts)
    assert all(receipt.verify() for receipt in recorder.receipts)
    assert "secret" not in str([receipt.metadata for receipt in recorder.receipts])

    first_start = recorder.receipts[1]
    assert first_start.metadata["client_session_key"] == "main"
    assert first_start.metadata["allowed_tools"] == ["Read"]
    assert first_start.metadata["mcp_server_names"] == ["filesystem"]
    assert recorder.receipts[2].metadata["tool_name"] == "mcp__docs__search"
    assert recorder.receipts[3].metadata["claude_session_id"] == "sess-cont"
    assert recorder.receipts[4].metadata["permission_mode"] == "acceptEdits"
    assert recorder.receipts[5].metadata["model"] == "claude-opus-4-5"
    assert recorder.receipts[6].metadata["permission_mode"] == "acceptEdits"


async def test_claude_client_session_records_controls_and_permission_decisions() -> None:
    class _Allow:
        decision = "allow"
        reason = "read-only"

    clients: list[_FakeClaudeClient] = []

    def client_factory(*, options: Any | None = None) -> _FakeClaudeClient:
        client = _FakeClaudeClient(options=options, responses=[[_ResultMessage("sess-control")]])
        clients.append(client)
        return client

    async def can_use_tool(tool_name: str, input_data: Any, context: Any) -> _Allow:
        del tool_name, input_data, context
        return _Allow()

    recorder = TrajectoryRecorder()
    mesh = Mesh()
    mesh.record(recorder)
    session = ClaudeAgentSDKClientSession(
        client_factory=client_factory,
        options_factory=_RunnerOptions,
        mesh=mesh,
        can_use_tool=can_use_tool,
    )

    async with mesh:
        await session.connect()
        options = clients[0].options
        assert options is not None
        permission_result = await options.can_use_tool(
            "Read",
            {"path": "/tmp/secret.txt"},
            object(),
        )
        status = await session.get_mcp_status()
        await session.reconnect_mcp_server("filesystem")
        await session.toggle_mcp_server("filesystem", False)
        await session.interrupt()
        await session.stop_task("task-1")
        await session.rewind_files("msg-1")
        await session.disconnect()

    assert isinstance(permission_result, _Allow)
    assert status["filesystem"]["status"] == "connected"
    client = clients[0]
    assert client.reconnected_servers == ["filesystem"]
    assert client.toggled_servers == [("filesystem", False)]
    assert client.interrupt_count == 1
    assert client.stopped_tasks == ["task-1"]
    assert client.rewound_messages == ["msg-1"]

    actions = [receipt.action for receipt in recorder.receipts]
    assert actions == [
        "claude_client_connect",
        "claude_permission_decision",
        "claude_mcp_status",
        "claude_mcp_reconnect",
        "claude_mcp_toggle",
        "claude_interrupt",
        "claude_task_stop",
        "claude_rewind_files",
        "claude_client_disconnect",
    ]
    permission = recorder.receipts[1]
    assert permission.metadata["decision"] == "allowed"
    assert permission.metadata["tool_input_keys"] == ["path"]
    assert "/tmp/secret.txt" not in str(permission.metadata)
    mcp_status = recorder.receipts[2]
    assert mcp_status.metadata["status_summary"] == {
        "filesystem": {"status": "connected"}
    }
    assert "secret" not in str([receipt.metadata for receipt in recorder.receipts])


async def test_claude_client_session_uses_tool_policy_for_permission_receipts() -> None:
    clients: list[_FakeClaudeClient] = []

    def client_factory(*, options: Any | None = None) -> _FakeClaudeClient:
        client = _FakeClaudeClient(options=options, responses=[[_ResultMessage("sess-policy")]])
        clients.append(client)
        return client

    policy = ClaudeToolPolicy(
        gate=Gate.filter(
            lambda signal: (
                isinstance(signal, ClaudeToolRequestSignal)
                and "path" in signal.tool_input_keys
            ),
            name="has_path",
        ),
        disallowed_tools=["Bash*"],
        deny_message="tool policy rejected the request",
    )
    recorder = TrajectoryRecorder()
    mesh = Mesh()
    mesh.record(recorder)
    session = ClaudeAgentSDKClientSession(
        client_factory=client_factory,
        options_factory=_RunnerOptions,
        mesh=mesh,
        tool_policy=policy,
    )

    async with mesh:
        await session.connect()
        options = clients[0].options
        assert options is not None
        allowed = await options.can_use_tool("Write", {"path": "/tmp/raw.txt"}, object())
        denied = await options.can_use_tool("Write", {"content": "raw value"}, object())
        disallowed = await options.can_use_tool("Bash", {"command": "echo ok"}, object())
        await session.disconnect()

    assert getattr(allowed, "behavior") == "allow"
    assert getattr(denied, "behavior") == "deny"
    assert getattr(disallowed, "behavior") == "deny"
    assert options.disallowed_tools == ["Bash*"]
    actions = [receipt.action for receipt in recorder.receipts]
    assert actions == [
        "claude_client_connect",
        "claude_permission_decision",
        "claude_permission_decision",
        "claude_permission_decision",
        "claude_client_disconnect",
    ]
    decisions = [receipt.metadata["decision"] for receipt in recorder.receipts[1:4]]
    assert decisions == ["allowed", "denied", "denied"]
    assert recorder.receipts[1].metadata["tool_input_keys"] == ["path"]
    assert recorder.receipts[2].metadata["tool_input_keys"] == ["content"]
    assert "/tmp/raw.txt" not in str([receipt.metadata for receipt in recorder.receipts])
    assert "raw value" not in str([receipt.metadata for receipt in recorder.receipts])


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


def test_claude_sdk_session_missing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal_gating.claude as claude_module

    def missing_import(name: str) -> object:
        if name == "claude_agent_sdk":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr(claude_module, "import_module", missing_import)

    session = ClaudeAgentSDKClientSession()
    with pytest.raises(ImportError, match=r"signal-gating\[claude\]"):
        session._sdk_bindings()


def test_import_does_not_pull_claude_agent_sdk() -> None:
    code = "import sys, signal_gating; assert 'claude_agent_sdk' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_exports_from_package_root_and_integration_module() -> None:
    import signal_gating
    from signal_gating.integrations import claude

    assert hasattr(signal_gating, "ClaudeAgent")
    assert hasattr(signal_gating, "ClaudeAgentSDKClientSession")
    assert hasattr(signal_gating, "ClaudeAgentSDKSession")
    assert hasattr(signal_gating, "ClaudeToolPolicy")
    assert hasattr(signal_gating, "ClaudeToolRequestSignal")
    assert hasattr(claude, "ClaudeAgent")
    assert hasattr(claude, "ClaudeAgentSDKClientSession")
    assert hasattr(claude, "ClaudeAgentSDKSession")
    assert hasattr(claude, "ClaudeToolPolicy")
    assert hasattr(claude, "ClaudeToolRequestSignal")
    assert "ClaudeAgent" in signal_gating.__all__
    assert "ClaudeAgentSDKClientSession" in signal_gating.__all__
    assert "ClaudeAgentSDKSession" in signal_gating.__all__
    assert "ClaudeToolPolicy" in signal_gating.__all__
    assert "ClaudeToolRequestSignal" in signal_gating.__all__
    assert ClaudeAgentSDKSession is ClaudeAgentSDKClientSession
