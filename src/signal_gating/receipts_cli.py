"""Receipt inspection CLI for trajectory JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from signal_gating.tracing import OpenTelemetryReceiptMetricsExporter
from signal_gating.trajectory import Receipt, TrajectoryRecorder

AUTHORIZATION_SIGNAL_TYPE = "sgp.integrations.claude.mcp_http_authorization.v1"


def main(argv: Sequence[str] | None = None) -> int:
    """Console entrypoint for ``signal-gating-receipts``."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.otel_include_paths and not args.otel:
        parser.error("--otel-include-paths requires --otel")
    if args.otel_max_paths is not None and not args.otel_include_paths:
        parser.error("--otel-max-paths requires --otel-include-paths")
    event_kinds = ["claude_mcp_http"] if args.command == "auth" else args.event_kind
    signal_types = [AUTHORIZATION_SIGNAL_TYPE] if args.command == "auth" else args.signal_type
    try:
        metrics = build_receipt_metrics(
            args.jsonl,
            event_kinds=event_kinds,
            actions=args.action,
            signal_types=signal_types,
            verify=not args.no_verify,
        )
        if args.otel:
            emit_otel_metrics(
                metrics,
                include_path_values=args.otel_include_paths,
                max_path_values=args.otel_max_paths,
            )
    except Exception as e:
        print(f"signal-gating-receipts: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    indent = 2 if args.pretty else None
    print(json.dumps(metrics, indent=indent, sort_keys=True))
    return 0


def build_receipt_metrics(
    path: str | Path,
    *,
    event_kinds: Iterable[str] | None = None,
    actions: Iterable[str] | None = None,
    signal_types: Iterable[str] | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Load trajectory JSONL and summarize selected receipts without raw payloads."""
    recorder = TrajectoryRecorder()
    event_kind_filter = _filter_tuple(event_kinds)
    action_filter = _filter_tuple(actions)
    signal_type_filter = _filter_tuple(signal_types)
    loaded = recorder.load_jsonl(path, verify=False)
    receipts = recorder.filter_receipts(
        event_kinds=event_kind_filter,
        actions=action_filter,
        signal_types=signal_type_filter,
        verify=verify,
    )
    timestamps = [receipt.timestamp for receipt in receipts]
    first_timestamp = min(timestamps) if timestamps else None
    last_timestamp = max(timestamps) if timestamps else None
    return {
        "schema": "signal-gating.receipt_metrics.v1",
        "path": str(path),
        "loaded": loaded,
        "loaded_receipts": loaded,
        "matched": len(receipts),
        "matched_receipts": len(receipts),
        "trace_count": len({receipt.trace_id for receipt in receipts}),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "duration_seconds": (
            last_timestamp - first_timestamp
            if first_timestamp is not None and last_timestamp is not None
            else None
        ),
        "verified": verify,
        "filters": {
            "event_kinds": list(event_kind_filter or ()),
            "actions": list(action_filter or ()),
            "signal_types": list(signal_type_filter or ()),
        },
        "counts": {
            "event_kinds": _counter(receipt.event_kind for receipt in receipts),
            "actions": _counter(receipt.action for receipt in receipts),
            "signal_types": _counter(receipt.signal_type for receipt in receipts),
            "outcomes": _payload_counter(receipts, "outcome"),
            "status_codes": _payload_counter(receipts, "status_code"),
            "reasons": _payload_counter(receipts, "reason"),
            "methods": _payload_counter(receipts, "method"),
            "paths": _payload_counter(receipts, "path"),
            "jsonrpc_methods": _payload_counter(receipts, "jsonrpc_method"),
            "identity_binding_kinds": _payload_counter(receipts, "identity_binding_kind"),
            "scope_counts": _payload_counter(receipts, "scope_count"),
        },
        "presence": {
            "bearer_token_present": _payload_present_count(receipts, "bearer_token_present"),
            "principal_present": _payload_present_count(receipts, "principal_present"),
            "audience_present": _payload_present_count(receipts, "audience_present"),
            "resource_present": _payload_present_count(receipts, "resource_present"),
            "mcp_session_present": _payload_present_count(receipts, "mcp_session_present"),
            "protected_resource_metadata_advertised": _payload_present_count(
                receipts,
                "protected_resource_metadata_advertised",
            ),
        },
    }


def emit_otel_metrics(
    metrics: Mapping[str, Any],
    *,
    include_path_values: bool = False,
    max_path_values: int | None = None,
) -> None:
    """Export aggregate receipt metrics through the configured OpenTelemetry meter."""
    OpenTelemetryReceiptMetricsExporter(
        include_path_values=include_path_values,
        max_path_values=100 if max_path_values is None else max_path_values,
    )(metrics)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-gating-receipts",
        description="Summarize verifiable SGP trajectory receipt JSONL files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser(
        "summary",
        help="Summarize selected receipts from a trajectory JSONL file.",
    )
    _add_common_arguments(summary)
    summary.add_argument(
        "--event-kind",
        action="append",
        default=[],
        help="Receipt event_kind to include. May be repeated.",
    )
    summary.add_argument(
        "--signal-type",
        action="append",
        default=[],
        help="Stable signal wire type to include. May be repeated.",
    )
    auth = subparsers.add_parser(
        "auth",
        help="Summarize Claude MCP HTTP authorization receipts.",
    )
    _add_common_arguments(auth)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("jsonl", help="Path to a TrajectoryRecorder JSONL export.")
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        help="Receipt action to include. May be repeated.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Load receipts without digest verification.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output (the default).")
    parser.add_argument(
        "--otel",
        action="store_true",
        help="Also export aggregate metrics through OpenTelemetry.",
    )
    parser.add_argument(
        "--otel-include-paths",
        action="store_true",
        help=(
            "Include raw receipt path values as OpenTelemetry dimensions; "
            "explicit opt-in due cardinality/privacy."
        ),
    )
    parser.add_argument(
        "--otel-max-paths",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help="Maximum distinct path labels to export before an __other__ overflow bucket.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")


def _filter_tuple(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized = tuple(values)
    return normalized or None


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(value) for value in values if value not in (None, "")))


def _payload_counter(receipts: Iterable[Receipt], field: str) -> dict[str, int]:
    return _counter(receipt.payload.get(field) for receipt in receipts)


def _payload_present_count(receipts: Iterable[Receipt], field: str) -> int:
    return sum(1 for receipt in receipts if bool(receipt.payload.get(field)))


__all__ = ["build_receipt_metrics", "emit_otel_metrics", "main"]
