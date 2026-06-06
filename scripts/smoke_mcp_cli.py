"""Smoke-test the installed MCP stdio console script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        module_dir = Path(tmp)
        _write_fixture_module(module_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(module_dir)
        result = subprocess.run(
            [
                "signal-gating-mcp",
                "sgp_mcp_smoke:make_mesh",
                "--server-name",
                "sgp-smoke",
                "--server-version",
                "9.9.9",
            ],
            capture_output=True,
            check=False,
            env=env,
            input=_jsonl(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hello"}},
                },
            ),
            text=True,
            timeout=10,
        )

    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    messages = _decode_jsonl(result.stdout)
    _assert_jsonrpc_messages(messages)
    _assert_diagnostics(result.stderr)
    return 0


def _write_fixture_module(path: Path) -> None:
    (path / "sgp_mcp_smoke.py").write_text(
        textwrap.dedent(
            """
            print("sgp-mcp-smoke: module import")

            from signal_gating import Agent, Mesh


            class LoudMesh(Mesh):
                async def start(self):
                    print("sgp-mcp-smoke: mesh start")
                    await super().start()

                async def stop(self, drain=False, drain_timeout=10.0):
                    print("sgp-mcp-smoke: mesh stop")
                    await super().stop(drain=drain, drain_timeout=drain_timeout)


            def make_mesh():
                print("sgp-mcp-smoke: factory")
                worker = Agent("worker")

                @worker.tool(description="Echo text")
                async def echo(text: str) -> dict[str, str]:
                    print("sgp-mcp-smoke: tool")
                    return {"echo": text}

                return LoudMesh([worker])
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _jsonl(*messages: dict[str, Any]) -> str:
    return "".join(f"{json.dumps(message)}\n" for message in messages)


def _decode_jsonl(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        row = json.loads(line)
        _require(isinstance(row, dict), "MCP stdout line is not a JSON object")
        rows.append(row)
    return rows


def _assert_jsonrpc_messages(messages: list[dict[str, Any]]) -> None:
    _require_equal([message["id"] for message in messages], [1, 2, 3], "response ids")
    initialize, tools, call = messages
    _require_equal(
        initialize["result"]["serverInfo"],
        {"name": "sgp-smoke", "version": "9.9.9"},
        "initialize server info",
    )
    _require_equal(
        initialize["result"]["capabilities"],
        {"tools": {"listChanged": False}},
        "initialize capabilities",
    )
    _require_equal(tools["result"]["tools"][0]["name"], "echo", "tool name")
    _require_equal(
        tools["result"]["tools"][0]["description"],
        "Echo text",
        "tool description",
    )
    _require_equal(
        call["result"]["structuredContent"],
        {"echo": "hello"},
        "tool structured content",
    )
    _require_equal(call["result"]["isError"], False, "tool error flag")


def _assert_diagnostics(stderr: str) -> None:
    expected = [
        "sgp-mcp-smoke: module import",
        "sgp-mcp-smoke: factory",
        "sgp-mcp-smoke: mesh start",
        "sgp-mcp-smoke: tool",
        "sgp-mcp-smoke: mesh stop",
    ]
    _require_equal(stderr.splitlines(), expected, "stderr diagnostics")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
