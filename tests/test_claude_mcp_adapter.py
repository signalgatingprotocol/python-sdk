"""Tests for exposing SGP mesh tools through MCP-shaped Claude adapters."""

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from signal_gating import Agent, ClaudeMeshMCPAdapter, ClaudeMeshMCPStdioServer, Mesh, ToolSpec


class _ToolSpecMesh:
    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    def discover_tools(self) -> dict[str, list[ToolSpec]]:
        return {"docs": [self._spec]}


class _FlushSpy(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


async def test_claude_mesh_mcp_adapter_lists_tools_with_mcp_input_schema() -> None:
    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a market topic")
    async def analyze(topic: str, depth: int = 1) -> dict[str, object]:
        return {"topic": topic, "depth": depth}

    adapter = ClaudeMeshMCPAdapter(Mesh([analyst]), server_name="sgp")

    assert adapter.tool_names() == ["analyze"]
    assert adapter.claude_allowed_tools() == ["mcp__sgp__analyze"]
    assert adapter.tool_policy().allowed_tools == ["mcp__sgp__analyze"]

    result = adapter.tools_list_result()

    assert result == {
        "tools": [
            {
                "name": "analyze",
                "description": "Analyze a market topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "depth": {"type": "integer", "default": 1},
                    },
                    "required": ["topic"],
                    "additionalProperties": False,
                },
                "annotations": {"title": "analyst.analyze"},
            }
        ]
    }


def test_claude_mesh_mcp_adapter_converts_explicit_tool_spec_parameters() -> None:
    spec = ToolSpec(
        name="search",
        description="Search docs",
        parameters={
            "query": {"type": "str", "required": True},
            "limit": {"type": "int", "required": False, "default": 5},
            "threshold": {"type": "float", "required": False, "default": 0.5},
            "include_archived": {"type": "bool", "required": False, "default": False},
            "filters": {"type": "dict", "required": False},
            "items": {"type": "list", "required": False},
            "opaque": {"type": "CustomThing", "required": False},
        },
    )
    adapter = ClaudeMeshMCPAdapter(_ToolSpecMesh(spec), server_name="sgp")

    schema = adapter.tools_list_result()["tools"][0]

    assert schema == {
        "name": "search",
        "description": "Search docs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "threshold": {"type": "number", "default": 0.5},
                "include_archived": {"type": "boolean", "default": False},
                "filters": {"type": "object"},
                "items": {"type": "array"},
                "opaque": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"title": "docs.search"},
    }


async def test_claude_mesh_mcp_adapter_calls_async_mesh_tool() -> None:
    analyst = Agent("analyst")

    @analyst.tool(description="Score a symbol")
    async def score(symbol: str, confidence: float = 0.5) -> dict[str, object]:
        return {"symbol": symbol, "confidence": confidence}

    mesh = Mesh([analyst])
    adapter = ClaudeMeshMCPAdapter(mesh)

    async with mesh:
        result = await adapter.tools_call_result(
            "score",
            {"symbol": "AAPL", "confidence": 0.9},
        )

    assert result["isError"] is False
    assert result["structuredContent"] == {"symbol": "AAPL", "confidence": 0.9}
    assert "AAPL" in result["content"][0]["text"]


async def test_claude_mesh_mcp_adapter_returns_tool_errors_inside_result() -> None:
    worker = Agent("worker")

    @worker.tool(description="Fail deterministically")
    async def fail() -> str:
        raise RuntimeError("boom")

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh)

    async with mesh:
        result = await adapter.tools_call_result("fail")

    assert result["isError"] is True
    assert "RuntimeError" in result["content"][0]["text"]
    assert "boom" in result["content"][0]["text"]


async def test_claude_mesh_mcp_adapter_validates_arguments_as_jsonrpc_errors() -> None:
    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a topic")
    async def analyze(topic: str, depth: int = 1) -> dict[str, object]:
        return {"topic": topic, "depth": depth}

    mesh = Mesh([analyst])
    adapter = ClaudeMeshMCPAdapter(mesh)

    with pytest.raises(ValueError, match="missing required argument 'topic'"):
        await adapter.tools_call_result("analyze", {})
    with pytest.raises(ValueError, match="expected str, got int"):
        await adapter.tools_call_result("analyze", {"topic": 123})
    with pytest.raises(ValueError, match="unexpected argument 'extra'"):
        await adapter.tools_call_result("analyze", {"topic": "x", "extra": True})

    async with mesh:
        await adapter.handle_request({"jsonrpc": "2.0", "id": "init", "method": "initialize"})
        await adapter.handle_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        response = await adapter.handle_request({
            "jsonrpc": "2.0",
            "id": "bad",
            "method": "tools/call",
            "params": {"name": "analyze", "arguments": {"topic": 123}},
        })

    assert response is not None
    assert response["error"]["code"] == -32602
    assert "expected str, got int" in response["error"]["message"]


async def test_claude_mesh_mcp_adapter_tool_results_are_json_safe() -> None:
    worker = Agent("worker")

    @worker.tool(description="Return structured content")
    async def report() -> dict[str, Any]:
        return {
            "path": Path("/tmp/report.txt"),
            "nested": {"ok": True},
            "secret": "raw",
        }

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh)

    async with mesh:
        result = await adapter.tools_call_result("report")

    assert result["structuredContent"] == {
        "path": "/tmp/report.txt",
        "nested": {"ok": True},
    }
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    json.dumps(result["structuredContent"])
    assert "secret" not in result["content"][0]["text"]
    assert "raw" not in result["content"][0]["text"]


async def test_claude_mesh_mcp_adapter_primitive_result_omits_structured_content() -> None:
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        return "pong"

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh)

    async with mesh:
        result = await adapter.tools_call_result("ping")

    assert result == {
        "content": [{"type": "text", "text": "pong"}],
        "isError": False,
    }


async def test_claude_mesh_mcp_adapter_fails_fast_when_mesh_is_not_running() -> None:
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        return "pong"

    adapter = ClaudeMeshMCPAdapter(Mesh([worker]))

    with pytest.raises(RuntimeError, match="mesh is not running"):
        await adapter.tools_call_result("ping")


async def test_claude_mesh_mcp_adapter_handles_jsonrpc_requests() -> None:
    worker = Agent("worker")

    @worker.tool(description="Add two values")
    async def add(a: int, b: int) -> int:
        return a + b

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh, server_name="sgp")

    async with mesh:
        initialize = await adapter.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })
        initialized = await adapter.handle_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        tools = await adapter.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })
        call = await adapter.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 2, "b": 5}},
        })

    assert initialize is not None
    assert initialize["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert initialized is None
    assert tools is not None
    assert tools["result"]["tools"][0]["name"] == "add"
    assert call is not None
    assert call["result"] == {
        "content": [{"type": "text", "text": "7"}],
        "isError": False,
    }


async def test_claude_mesh_mcp_adapter_jsonrpc_errors_and_notifications() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())

    assert await adapter.handle_request({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }) is None

    await adapter.handle_request({"jsonrpc": "2.0", "id": "init", "method": "initialize"})
    await adapter.handle_request({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })
    unknown = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "nope",
    })
    invalid = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"arguments": []},
    })

    assert unknown is not None
    assert unknown["error"]["code"] == -32601
    assert invalid is not None
    assert invalid["error"]["code"] == -32602


async def test_claude_mesh_mcp_adapter_requires_initialize_before_tools() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())

    before_initialize = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    })
    await adapter.handle_request({"jsonrpc": "2.0", "id": 2, "method": "initialize"})
    before_initialized_notification = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/list",
    })

    assert before_initialize is not None
    assert before_initialize["error"]["code"] == -32002
    assert before_initialized_notification is not None
    assert before_initialized_notification["error"]["code"] == -32002


async def test_claude_mesh_mcp_adapter_rejects_invalid_jsonrpc_shape() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())

    bad_version = await adapter.handle_request({
        "jsonrpc": "1.0",
        "id": 1,
        "method": "initialize",
    })
    null_id = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": None,
        "method": "initialize",
    })
    non_string_method = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": 3,
    })

    assert bad_version is not None
    assert bad_version["error"]["code"] == -32600
    assert null_id is not None
    assert null_id["error"]["code"] == -32600
    assert non_string_method is not None
    assert non_string_method["error"]["code"] == -32600


def test_claude_mesh_mcp_adapter_rejects_duplicate_tool_names() -> None:
    first = Agent("first")
    second = Agent("second")

    @first.tool(name="ping", description="first")
    def first_ping() -> str:
        return "first"

    @second.tool(name="ping", description="second")
    def second_ping() -> str:
        return "second"

    adapter = ClaudeMeshMCPAdapter(Mesh([first, second]))

    with pytest.raises(ValueError, match="duplicate tool name"):
        adapter.tools_list_result()


async def test_claude_mesh_mcp_adapter_duplicate_tool_names_return_jsonrpc_error() -> None:
    first = Agent("first")
    second = Agent("second")

    @first.tool(name="dup", description="first")
    def first_dup() -> str:
        return "first"

    @second.tool(name="dup", description="second")
    def second_dup() -> str:
        return "second"

    adapter = ClaudeMeshMCPAdapter(Mesh([first, second]))

    await adapter.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    await adapter.handle_request({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })
    response = await adapter.handle_request({
        "jsonrpc": "2.0",
        "id": "list",
        "method": "tools/list",
    })

    assert response is not None
    assert response["error"]["code"] == -32602
    assert "duplicate tool name 'dup'" in response["error"]["message"]


def test_claude_mesh_mcp_adapter_exports_from_package_root() -> None:
    import signal_gating
    from signal_gating.integrations import claude

    assert hasattr(signal_gating, "ClaudeMeshMCPAdapter")
    assert hasattr(signal_gating, "ClaudeMeshMCPStdioServer")
    assert hasattr(claude, "ClaudeMeshMCPAdapter")
    assert hasattr(claude, "ClaudeMeshMCPStdioServer")
    assert "ClaudeMeshMCPAdapter" in signal_gating.__all__
    assert "ClaudeMeshMCPStdioServer" in signal_gating.__all__


async def test_claude_mesh_mcp_stdio_server_writes_jsonrpc_lines_only() -> None:
    worker = Agent("worker")

    @worker.tool(description="Echo text")
    async def echo(text: str) -> dict[str, str]:
        return {"echo": text}

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh, server_name="sgp")
    server = ClaudeMeshMCPStdioServer(adapter)
    output = StringIO()
    requests = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        }),
    ]

    async with mesh:
        written = await server.serve(requests, output)

    lines = output.getvalue().splitlines()

    assert written == 3
    assert len(lines) == 3
    decoded = [json.loads(line) for line in lines]
    assert [message["id"] for message in decoded] == [1, 2, 3]
    assert decoded[0]["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert decoded[1]["result"]["tools"][0]["name"] == "echo"
    assert decoded[2]["result"]["structuredContent"] == {"echo": "hi"}


async def test_claude_mesh_mcp_stdio_server_parse_and_shape_errors() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())
    server = ClaudeMeshMCPStdioServer(adapter)
    output = StringIO()

    written = await server.serve(["", "{not-json}", "[]"], output)

    lines = output.getvalue().splitlines()

    assert written == 2
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["error"]["code"] == -32700
    assert decoded[0]["id"] is None
    assert decoded[1]["error"]["code"] == -32600
    assert decoded[1]["id"] is None


async def test_claude_mesh_mcp_stdio_server_notifications_and_blanks_write_nothing() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())
    server = ClaudeMeshMCPStdioServer(adapter)
    output = StringIO()

    written = await server.serve([
        "",
        "   ",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled"}),
    ], output)

    assert written == 0
    assert output.getvalue() == ""


async def test_claude_mesh_mcp_stdio_server_continues_after_bad_line() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh(), server_name="sgp")
    server = ClaudeMeshMCPStdioServer(adapter)
    output = StringIO()

    written = await server.serve([
        "{bad json}",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    ], output)

    decoded = [json.loads(line) for line in output.getvalue().splitlines()]

    assert written == 2
    assert decoded[0]["error"]["code"] == -32700
    assert decoded[1]["id"] == 1
    assert decoded[1]["result"]["serverInfo"]["name"] == "sgp"


async def test_claude_mesh_mcp_stdio_server_outputs_compact_jsonl_and_flushes() -> None:
    adapter = ClaudeMeshMCPAdapter(Mesh())
    server = ClaudeMeshMCPStdioServer(adapter)
    output = _FlushSpy()

    written = await server.serve([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nope"}),
    ], output)

    raw = output.getvalue()
    lines = raw.splitlines(keepends=True)

    assert written == 2
    assert output.flush_count == 2
    assert len(lines) == 2
    assert all(line.endswith("\n") for line in lines)
    assert all("\n" not in line[:-1] for line in lines)
    assert all(json.loads(line) for line in lines)
    assert "\n\n" not in raw
    assert "  " not in raw


async def test_claude_mesh_mcp_stdio_server_redirects_tool_stdout(capsys: Any) -> None:
    worker = Agent("worker")

    @worker.tool(description="Noisy tool")
    async def noisy() -> dict[str, str]:
        print("raw tool stdout")
        return {"ok": "yes"}

    mesh = Mesh([worker])
    adapter = ClaudeMeshMCPAdapter(mesh)
    server = ClaudeMeshMCPStdioServer(adapter)
    requests = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "noisy"},
        }),
    ]

    async with mesh:
        written = await server.serve(requests, sys.stdout)

    captured = capsys.readouterr()
    stdout_lines = captured.out.splitlines()

    assert written == 2
    assert "raw tool stdout" not in captured.out
    assert "raw tool stdout" in captured.err
    assert len(stdout_lines) == 2
    assert [json.loads(line)["id"] for line in stdout_lines] == [1, 2]
    assert json.loads(stdout_lines[1])["result"]["structuredContent"] == {"ok": "yes"}
