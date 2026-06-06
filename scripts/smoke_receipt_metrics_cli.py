"""Smoke-test the installed receipt metrics console script.

This intentionally shells out to ``signal-gating-receipts`` so CI proves the
packaged entrypoint, optional OpenTelemetry import path, JSONL loading, auth
filtering, and high-cardinality path controls work together.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from signal_gating import ClaudeMCPHTTPAuthorizationSignal, Receipt, Signal

AUTH_SIGNAL_TYPE = "sgp.integrations.claude.mcp_http_authorization.v1"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        receipts_path = Path(tmp) / "auth-receipts.jsonl"
        otel_calls_path = Path(tmp) / "otel-calls.jsonl"
        hooks_dir = Path(tmp) / "pythonpath"
        hooks_dir.mkdir()
        _write_receipts(receipts_path)
        _write_otel_sitecustomize(hooks_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(hooks_dir)
        env["SGP_SMOKE_OTEL_JSONL"] = str(otel_calls_path)
        result = subprocess.run(
            [
                "signal-gating-receipts",
                "auth",
                str(receipts_path),
                "--otel",
                "--otel-include-paths",
                "--otel-max-paths",
                "1",
                "--pretty",
            ],
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=10,
        )
        otel_calls = _read_jsonl(otel_calls_path)

    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    if result.stderr:
        sys.stderr.write(result.stderr)
        return 1
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"receipt CLI emitted invalid JSON: {exc}\n{result.stdout}")
        return 1

    _assert_summary(summary)
    _assert_otel_calls(otel_calls)
    if "ordinary-trajectory-secret" in result.stdout:
        raise AssertionError("ordinary receipt metadata leaked into auth metrics output")
    return 0


def _write_receipts(path: Path) -> None:
    receipts = [
        Receipt.from_signal(
            Signal(
                id="ordinary-signal",
                timestamp=100.0,
                trace_id="trace-ordinary",
                metadata={"secret": "ordinary-trajectory-secret"},
            ),
            source="mesh",
            target="worker",
            event_kind="mesh",
            action="inject",
            timestamp=100.5,
        ),
        _auth_receipt(
            signal_id="auth-allowed",
            trace_id="trace-auth",
            timestamp=101.0,
            receipt_timestamp=101.5,
            action="claude_mcp_http_auth_allowed",
            outcome="allowed",
            status_code=200,
            path="/mcp/a",
            reason="allowed",
            principal_present=True,
        ),
        _auth_receipt(
            signal_id="auth-denied",
            trace_id="trace-auth",
            timestamp=102.0,
            receipt_timestamp=102.5,
            action="claude_mcp_http_auth_denied",
            outcome="denied",
            status_code=403,
            path="/mcp/b",
            reason="insufficient_scope",
            principal_present=True,
        ),
    ]
    path.write_text(
        "".join(json.dumps(receipt.to_dict(), default=str) + "\n" for receipt in receipts),
        encoding="utf-8",
    )


def _auth_receipt(
    *,
    signal_id: str,
    trace_id: str,
    timestamp: float,
    receipt_timestamp: float,
    action: str,
    outcome: str,
    status_code: int,
    path: str,
    reason: str,
    principal_present: bool,
) -> Receipt:
    return Receipt.from_signal(
        ClaudeMCPHTTPAuthorizationSignal(
            id=signal_id,
            timestamp=timestamp,
            trace_id=trace_id,
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
        timestamp=receipt_timestamp,
    )


def _write_otel_sitecustomize(path: Path) -> None:
    (path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import json
            import os

            from opentelemetry import metrics

            _calls_path = os.environ["SGP_SMOKE_OTEL_JSONL"]


            def _write_call(kind, name, amount, attributes):
                with open(_calls_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "kind": kind,
                                "name": name,
                                "amount": amount,
                                "attributes": dict(attributes or {}),
                            },
                            sort_keys=True,
                        )
                        + "\\n"
                    )


            class _Instrument:
                def __init__(self, kind, name):
                    self._kind = kind
                    self._name = name

                def add(self, amount, *, attributes=None):
                    _write_call(self._kind, self._name, amount, attributes)

                def record(self, amount, *, attributes=None):
                    _write_call(self._kind, self._name, amount, attributes)


            class _Meter:
                def create_counter(self, name, *, unit, description):
                    return _Instrument("counter", name)

                def create_histogram(self, name, *, unit, description):
                    return _Instrument("histogram", name)


            def _get_meter(name):
                if name != "signal_gating":
                    raise AssertionError(name)
                return _Meter()


            metrics.get_meter = _get_meter
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            _require(isinstance(row, dict), "OpenTelemetry JSONL row is not an object")
            rows.append(row)
    return rows


def _assert_summary(summary: dict[str, Any]) -> None:
    _require_equal(summary["schema"], "signal-gating.receipt_metrics.v1", "schema")
    _require_equal(summary["loaded_receipts"], 3, "loaded receipt count")
    _require_equal(summary["matched_receipts"], 2, "matched receipt count")
    _require_equal(summary["trace_count"], 1, "trace count")
    _require_equal(summary["verified"], True, "verification flag")
    _require_equal(
        summary["filters"],
        {
            "event_kinds": ["claude_mcp_http"],
            "actions": [],
            "signal_types": [AUTH_SIGNAL_TYPE],
        },
        "filters",
    )
    _require_equal(
        summary["counts"]["actions"],
        {
            "claude_mcp_http_auth_allowed": 1,
            "claude_mcp_http_auth_denied": 1,
        },
        "action counts",
    )
    _require_equal(summary["counts"]["outcomes"], {"allowed": 1, "denied": 1}, "outcomes")
    _require_equal(summary["counts"]["paths"], {"/mcp/a": 1, "/mcp/b": 1}, "paths")
    _require_equal(summary["presence"]["bearer_token_present"], 2, "bearer presence")
    _require_equal(summary["presence"]["principal_present"], 2, "principal presence")


def _assert_otel_calls(calls: list[dict[str, Any]]) -> None:
    _require(calls, "OpenTelemetry smoke did not capture any metric calls")
    path_calls = sorted(
        (call["amount"], call["attributes"]["sgp.value"])
        for call in calls
        if call.get("kind") == "counter"
        and call.get("name") == "sgp.receipts.count"
        and call.get("attributes", {}).get("sgp.dimension") == "paths"
    )
    _require_equal(
        path_calls,
        [(1, "/mcp/a"), (1, "__other__")],
        "OpenTelemetry path calls",
    )
    all_call_text = repr(calls)
    _require("/mcp/b" not in all_call_text, "folded path leaked into OTel calls")
    _require(
        "ordinary-trajectory-secret" not in all_call_text,
        "ordinary trajectory metadata leaked into OTel calls",
    )


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
