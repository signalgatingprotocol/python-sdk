"""Tests for the packaged Signal Gating Protocol CLI entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from importlib import import_module
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import signal_gating.receipts_cli as receipts_cli_module
from signal_gating import ClaudeMCPHTTPAuthorizationSignal, Receipt, Signal
from signal_gating.cli import main, run_mcp_stdio
from signal_gating.receipts_cli import main as receipts_main

AUTH_SIGNAL_TYPE = "sgp.integrations.claude.mcp_http_authorization.v1"


def _write_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> str:
    module_name = f"sgp_cli_fixture_{uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return module_name


def _console_script(name: str) -> Path:
    script_dirs: list[Path] = []
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        script_dirs.append(Path(virtual_env) / ("Scripts" if os.name == "nt" else "bin"))
    script_dirs.append(Path(sys.executable).parent)
    script_dirs.append(Path(sysconfig.get_path("scripts")))
    filenames = [f"{name}.exe", f"{name}.cmd", name] if os.name == "nt" else [name]
    searched: list[str] = []
    for scripts_dir in dict.fromkeys(script_dirs):
        for filename in filenames:
            script = scripts_dir / filename
            searched.append(str(script))
            if script.exists():
                return script
    pytest.fail(f"console script {name!r} was not installed; searched {searched}")


def _input_stream(*messages: dict[str, Any]) -> StringIO:
    return StringIO("".join(f"{json.dumps(message)}\n" for message in messages))


def _decode_jsonl(output: StringIO) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for line in output.getvalue().splitlines():
        message = json.loads(line)
        assert isinstance(message, dict)
        decoded.append(message)
    return decoded


def _write_receipt_dicts(path: Path, receipts: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(receipt, default=str) + "\n" for receipt in receipts),
        encoding="utf-8",
    )


def _write_receipts(path: Path, receipts: list[Receipt]) -> None:
    _write_receipt_dicts(path, [receipt.to_dict() for receipt in receipts])


def _auth_receipt(
    *,
    action: str,
    outcome: str,
    status_code: int,
    reason: str,
    principal_present: bool,
    path: str = "/mcp",
) -> Receipt:
    return Receipt.from_signal(
        ClaudeMCPHTTPAuthorizationSignal(
            outcome=outcome,
            status_code=status_code,
            method="POST",
            path=path,
            reason=reason,
            authorization_scheme="Bearer",
            bearer_token_present=True,
            principal_hash="sha256:principal" if principal_present else "",
            principal_present=principal_present,
            audience_present=principal_present,
            resource_present=principal_present,
            scope_count=1 if principal_present else 0,
            identity_binding_kind="claims" if principal_present else "none",
        ),
        source="claude_mcp_http",
        target="",
        event_kind="claude_mcp_http",
        action=action,
    )


def test_pyproject_declares_console_scripts() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'signal-gating-mcp = "signal_gating.cli:main"' in pyproject
    assert 'signal-gating-receipts = "signal_gating.receipts_cli:main"' in pyproject


def test_receipts_console_script_auth_smoke_reads_generated_jsonl(tmp_path: Path) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            Receipt.from_signal(
                Signal(metadata={"secret": "ordinary-trajectory-secret"}),
                source="mesh",
                target="worker",
                event_kind="mesh",
                action="inject",
            ),
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
            _auth_receipt(
                action="claude_mcp_http_auth_denied",
                outcome="denied",
                status_code=403,
                reason="insufficient_scope",
                principal_present=True,
            ),
        ],
    )

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(_console_script("signal-gating-receipts")),
            "auth",
            str(receipts_path),
            "--action",
            "claude_mcp_http_auth_denied",
            "--pretty",
        ],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "\n  " in result.stdout
    assert "ordinary-trajectory-secret" not in result.stdout
    summary = json.loads(result.stdout)
    assert summary["loaded_receipts"] == 3
    assert summary["matched_receipts"] == 1
    assert summary["filters"] == {
        "event_kinds": ["claude_mcp_http"],
        "actions": ["claude_mcp_http_auth_denied"],
        "signal_types": [AUTH_SIGNAL_TYPE],
    }
    assert summary["counts"]["actions"] == {"claude_mcp_http_auth_denied": 1}
    assert summary["counts"]["outcomes"] == {"denied": 1}
    assert summary["counts"]["status_codes"] == {"403": 1}


def test_receipts_cli_summarizes_filtered_auth_receipts_without_raw_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    ordinary_receipt = Receipt.from_signal(
        Signal(metadata={"secret": "ordinary-trajectory-secret"}),
        source="mesh",
        target="worker",
        event_kind="mesh",
        action="inject",
    )
    tampered_ordinary_receipt = ordinary_receipt.to_dict()
    tampered_ordinary_receipt["payload"] = {"secret": "ordinary-trajectory-secret"}
    _write_receipt_dicts(
        receipts_path,
        [
            tampered_ordinary_receipt,
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ).to_dict(),
            _auth_receipt(
                action="claude_mcp_http_auth_query_token_rejected",
                outcome="denied",
                status_code=400,
                reason="query_token_rejected",
                principal_present=False,
            ).to_dict(),
        ],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main([
        "auth",
        str(receipts_path),
    ])

    summary = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert summary["schema"] == "signal-gating.receipt_metrics.v1"
    assert summary["loaded"] == 3
    assert summary["loaded_receipts"] == 3
    assert summary["matched"] == 2
    assert summary["matched_receipts"] == 2
    assert summary["verified"] is True
    assert summary["filters"] == {
        "event_kinds": ["claude_mcp_http"],
        "actions": [],
        "signal_types": [AUTH_SIGNAL_TYPE],
    }
    assert summary["counts"]["actions"] == {
        "claude_mcp_http_auth_allowed": 1,
        "claude_mcp_http_auth_query_token_rejected": 1,
    }
    assert summary["counts"]["outcomes"] == {"allowed": 1, "denied": 1}
    assert summary["counts"]["status_codes"] == {"200": 1, "400": 1}
    assert summary["counts"]["reasons"] == {"allowed": 1, "query_token_rejected": 1}
    assert summary["counts"]["identity_binding_kinds"] == {"claims": 1, "none": 1}
    assert summary["presence"]["bearer_token_present"] == 2
    assert summary["presence"]["principal_present"] == 1
    assert "ordinary-trajectory-secret" not in stdout.getvalue()
    assert "token-secret" not in stdout.getvalue()


def test_receipts_cli_summary_without_filters_matches_all_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            Receipt.from_signal(
                Signal(),
                source="mesh",
                target="worker",
                event_kind="mesh",
                action="inject",
            ),
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
        ],
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main(["summary", str(receipts_path)])

    summary = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert summary["matched"] == 2
    assert summary["filters"] == {
        "event_kinds": [],
        "actions": [],
        "signal_types": [],
    }
    assert summary["counts"]["event_kinds"] == {"mesh": 1, "claude_mcp_http": 1}


def test_receipts_cli_otel_exports_aggregates_and_still_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_denied",
                outcome="denied",
                status_code=403,
                reason="insufficient_scope",
                principal_present=True,
            ),
        ],
    )
    exported: list[dict[str, Any]] = []
    include_path_flags: list[bool] = []
    max_path_args: list[int | None] = []

    class FakeExporter:
        def __init__(
            self,
            *,
            include_path_values: bool = False,
            max_path_values: int | None = None,
        ) -> None:
            include_path_flags.append(include_path_values)
            max_path_args.append(max_path_values)

        def __call__(self, metrics: dict[str, Any]) -> None:
            exported.append(metrics)

    monkeypatch.setattr(
        receipts_cli_module,
        "OpenTelemetryReceiptMetricsExporter",
        FakeExporter,
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main(["auth", str(receipts_path), "--otel"])

    summary = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert include_path_flags == [False]
    assert max_path_args == [100]
    assert exported == [summary]
    assert summary["matched_receipts"] == 1
    assert summary["counts"]["outcomes"] == {"denied": 1}


def test_receipts_cli_otel_can_include_paths_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
        ],
    )
    include_path_flags: list[bool] = []
    max_path_args: list[int | None] = []

    class FakeExporter:
        def __init__(
            self,
            *,
            include_path_values: bool = False,
            max_path_values: int | None = None,
        ) -> None:
            include_path_flags.append(include_path_values)
            max_path_args.append(max_path_values)

        def __call__(self, metrics: dict[str, Any]) -> None:
            assert metrics["counts"]["paths"] == {"/mcp": 1}

    monkeypatch.setattr(
        receipts_cli_module,
        "OpenTelemetryReceiptMetricsExporter",
        FakeExporter,
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main([
        "auth",
        str(receipts_path),
        "--otel",
        "--otel-include-paths",
        "--otel-max-paths",
        "1",
    ])

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert include_path_flags == [True]
    assert max_path_args == [1]


def test_receipts_cli_otel_path_cap_uses_real_exporter_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                path="/mcp/a",
                reason="allowed",
                principal_present=True,
            ),
            _auth_receipt(
                action="claude_mcp_http_auth_denied",
                outcome="denied",
                status_code=403,
                path="/mcp/b",
                reason="insufficient_scope",
                principal_present=True,
            ),
        ],
    )

    class FakeInstrument:
        def __init__(self) -> None:
            self.add_calls: list[tuple[int | float, dict[str, Any]]] = []
            self.record_calls: list[tuple[int | float, dict[str, Any]]] = []

        def add(self, amount: int | float, *, attributes: dict[str, Any]) -> None:
            self.add_calls.append((amount, attributes))

        def record(self, amount: int | float, *, attributes: dict[str, Any]) -> None:
            self.record_calls.append((amount, attributes))

    class FakeMeter:
        def __init__(self) -> None:
            self.counters: dict[str, FakeInstrument] = {}
            self.histograms: dict[str, FakeInstrument] = {}

        def create_counter(
            self,
            name: str,
            *,
            unit: str,
            description: str,
        ) -> FakeInstrument:
            assert unit
            assert description
            instrument = FakeInstrument()
            self.counters[name] = instrument
            return instrument

        def create_histogram(
            self,
            name: str,
            *,
            unit: str,
            description: str,
        ) -> FakeInstrument:
            assert unit
            assert description
            instrument = FakeInstrument()
            self.histograms[name] = instrument
            return instrument

    fake_meter = FakeMeter()

    class FakeOtelMetrics:
        @staticmethod
        def get_meter(name: str) -> FakeMeter:
            assert name == "signal_gating"
            return fake_meter

    def fake_import_module(name: str) -> Any:
        if name == "opentelemetry.metrics":
            return FakeOtelMetrics
        return import_module(name)

    monkeypatch.setattr("signal_gating.tracing.import_module", fake_import_module)
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main([
        "auth",
        str(receipts_path),
        "--otel",
        "--otel-include-paths",
        "--otel-max-paths",
        "1",
    ])

    summary = json.loads(stdout.getvalue())
    count_calls = fake_meter.counters["sgp.receipts.count"].add_calls
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert summary["counts"]["paths"] == {"/mcp/a": 1, "/mcp/b": 1}
    assert (
        1,
        {
            "sgp.schema": "signal-gating.receipt_metrics.v1",
            "sgp.verified": True,
            "sgp.filter.event_kinds": ("claude_mcp_http",),
            "sgp.filter.signal_types": (AUTH_SIGNAL_TYPE,),
            "sgp.dimension": "paths",
            "sgp.value": "/mcp/a",
        },
    ) in count_calls
    assert (
        1,
        {
            "sgp.schema": "signal-gating.receipt_metrics.v1",
            "sgp.verified": True,
            "sgp.filter.event_kinds": ("claude_mcp_http",),
            "sgp.filter.signal_types": (AUTH_SIGNAL_TYPE,),
            "sgp.dimension": "paths",
            "sgp.value": "__other__",
        },
    ) in count_calls
    exported_path_values = {
        str(attributes["sgp.value"])
        for _, attributes in count_calls
        if attributes.get("sgp.dimension") == "paths"
    }
    assert exported_path_values == {"/mcp/a", "__other__"}


def test_receipts_cli_rejects_path_labels_without_otel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
        ],
    )
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as raised:
        receipts_main(["auth", str(receipts_path), "--otel-include-paths"])

    assert raised.value.code == 2
    assert "--otel-include-paths requires --otel" in stderr.getvalue()


def test_receipts_cli_rejects_path_cap_without_path_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
        ],
    )
    stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as raised:
        receipts_main(["auth", str(receipts_path), "--otel", "--otel-max-paths", "1"])

    assert raised.value.code == 2
    assert "--otel-max-paths requires --otel-include-paths" in stderr.getvalue()


def test_receipts_cli_otel_failure_keeps_stdout_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts_path = tmp_path / "receipts.jsonl"
    _write_receipts(
        receipts_path,
        [
            _auth_receipt(
                action="claude_mcp_http_auth_allowed",
                outcome="allowed",
                status_code=200,
                reason="allowed",
                principal_present=True,
            ),
        ],
    )

    class BrokenExporter:
        def __init__(
            self,
            *,
            include_path_values: bool = False,
            max_path_values: int | None = None,
        ) -> None:
            raise ImportError("Install it with: pip install 'signal-gating[otel]'")

    monkeypatch.setattr(
        receipts_cli_module,
        "OpenTelemetryReceiptMetricsExporter",
        BrokenExporter,
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main(["auth", str(receipts_path), "--otel"])

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "ImportError" in stderr.getvalue()
    assert "signal-gating[otel]" in stderr.getvalue()


def test_receipts_cli_verifies_by_default_and_can_load_without_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _auth_receipt(
        action="claude_mcp_http_auth_allowed",
        outcome="allowed",
        status_code=200,
        reason="allowed",
        principal_present=True,
    )
    tampered = receipt.to_dict()
    tampered["payload"] = {**receipt.payload, "status_code": 500}
    receipts_path = tmp_path / "tampered.jsonl"
    receipts_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = receipts_main(["auth", str(receipts_path)])

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "receipt digest mismatch" in stderr.getvalue()

    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    exit_code = receipts_main(["auth", str(receipts_path), "--no-verify"])

    summary = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert summary["verified"] is False
    assert summary["matched"] == 1
    assert summary["counts"]["status_codes"] == {"500": 1}


def test_mcp_cli_main_exits_zero_after_successful_mesh_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
from signal_gating import Agent, Mesh

def make_mesh():
    worker = Agent("worker")

    @worker.tool(description="Echo text")
    async def echo(text: str) -> dict[str, str]:
        return {"echo": text}

    return Mesh([worker])
""",
    )
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdin", _input_stream(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        },
    ))
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    exit_code = main([f"{module_name}:make_mesh", "--server-name", "sgp"])

    decoded = _decode_jsonl(stdout)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert [message["id"] for message in decoded] == [1, 2]
    assert decoded[0]["result"]["serverInfo"]["name"] == "sgp"
    assert decoded[1]["result"]["structuredContent"] == {"echo": "hello"}


async def test_mcp_cli_keeps_stdout_jsonrpc_pure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
print("module import noise")

from signal_gating import Agent, Mesh

class LoudMesh(Mesh):
    async def start(self) -> None:
        print("mesh start noise")
        await super().start()

    async def stop(self, drain: bool = False, drain_timeout: float = 10.0) -> None:
        print("mesh stop noise")
        await super().stop(drain=drain, drain_timeout=drain_timeout)

def make_mesh():
    print("factory noise")
    worker = Agent("worker")

    @worker.tool(description="Noisy echo")
    async def echo(text: str) -> dict[str, str]:
        print("tool noise")
        return {"echo": text}

    return LoudMesh([worker])
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_mesh",
        input_stream=_input_stream(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "clean"}},
            },
        ),
        output_stream=stdout,
        error_stream=stderr,
    )

    decoded = _decode_jsonl(stdout)
    assert exit_code == 0
    assert [message["id"] for message in decoded] == [1, 2]
    assert decoded[1]["result"]["structuredContent"] == {"echo": "clean"}
    assert "noise" not in stdout.getvalue()
    assert stderr.getvalue().splitlines() == [
        "module import noise",
        "factory noise",
        "mesh start noise",
        "tool noise",
        "mesh stop noise",
    ]


async def test_mcp_cli_mesh_factories_are_started_and_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
from signal_gating import Agent, Mesh

events = []

class RecordingMesh(Mesh):
    async def start(self) -> None:
        events.append("start")
        await super().start()

    async def stop(self, drain: bool = False, drain_timeout: float = 10.0) -> None:
        events.append("stop")
        await super().stop(drain=drain, drain_timeout=drain_timeout)

def make_mesh():
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        events.append("tool")
        return "pong"

    return RecordingMesh([worker])
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_mesh",
        input_stream=_input_stream(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping"},
            },
        ),
        output_stream=stdout,
        error_stream=stderr,
    )

    fixture = import_module(module_name)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert fixture.events == ["start", "tool", "stop"]


async def test_mcp_cli_accepts_async_mesh_factories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
from signal_gating import Agent, Mesh

async def make_mesh():
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        return "pong"

    return Mesh([worker])
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_mesh",
        input_stream=_input_stream(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping"},
            },
        ),
        output_stream=stdout,
        error_stream=stderr,
    )

    decoded = _decode_jsonl(stdout)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert decoded[1]["result"] == {
        "content": [{"type": "text", "text": "pong"}],
        "isError": False,
    }


async def test_mcp_cli_reports_loader_failures_on_stderr_only() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        "not-a-factory",
        input_stream=_input_stream({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        output_stream=stdout,
        error_stream=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "ValueError" in stderr.getvalue()
    assert "module:attribute" in stderr.getvalue()


async def test_mcp_cli_reports_bad_factory_return_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
def make_bad():
    return object()
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_bad",
        input_stream=_input_stream({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        output_stream=stdout,
        error_stream=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "TypeError" in stderr.getvalue()
    assert "Mesh or ClaudeMeshMCPAdapter" in stderr.getvalue()


async def test_mcp_cli_adapter_factories_are_served_without_mesh_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
from signal_gating import Agent, ClaudeMeshMCPAdapter, Mesh

def make_adapter():
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        return "pong"

    return ClaudeMeshMCPAdapter(Mesh([worker]))
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_adapter",
        input_stream=_input_stream(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping"},
            },
        ),
        output_stream=stdout,
        error_stream=stderr,
    )

    decoded = _decode_jsonl(stdout)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert decoded[1]["error"]["code"] == -32000
    assert "mesh is not running" in decoded[1]["error"]["message"]


async def test_mcp_cli_adapter_factories_can_own_running_mesh_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_module(
        tmp_path,
        monkeypatch,
        """
from signal_gating import Agent, ClaudeMeshMCPAdapter, Mesh

events = []
mesh_ref = None

class RecordingMesh(Mesh):
    async def stop(self, drain: bool = False, drain_timeout: float = 10.0) -> None:
        events.append("stop")
        await super().stop(drain=drain, drain_timeout=drain_timeout)

async def make_adapter():
    global mesh_ref
    worker = Agent("worker")

    @worker.tool(description="Ping")
    async def ping() -> str:
        events.append("tool")
        return "pong"

    mesh_ref = RecordingMesh([worker])
    await mesh_ref.start()
    events.append("factory_started")
    return ClaudeMeshMCPAdapter(mesh_ref)
""",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = await run_mcp_stdio(
        f"{module_name}:make_adapter",
        input_stream=_input_stream(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "ping"},
            },
        ),
        output_stream=stdout,
        error_stream=stderr,
    )

    fixture = import_module(module_name)
    decoded = _decode_jsonl(stdout)
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert decoded[1]["result"] == {
        "content": [{"type": "text", "text": "pong"}],
        "isError": False,
    }
    assert fixture.events == ["factory_started", "tool"]
    assert fixture.mesh_ref is not None
    assert fixture.mesh_ref._running is True
    await fixture.mesh_ref.stop()
