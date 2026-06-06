"""Tests for exposing SGP mesh tools through MCP Streamable HTTP."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from signal_gating import (
    Agent,
    ClaudeMCPHTTPAuthorizationContext,
    ClaudeMCPHTTPAuthorizationDecision,
    ClaudeMCPHTTPAuthorizationResult,
    ClaudeMCPHTTPAuthorizationSignal,
    ClaudeMCPProtectedResourceMetadata,
    ClaudeMeshMCPAdapter,
    ClaudeMeshMCPHTTPApp,
    Mesh,
    Receipt,
    Signal,
    SignalSerializationError,
    TrajectoryRecorder,
    TrajectoryReplayRunner,
    protected_resource_metadata_url,
)

AUTH_SIGNAL_TYPE = "sgp.integrations.claude.mcp_http_authorization.v1"

POST_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


@dataclass(slots=True)
class ASGIResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        decoded = json.loads(self.body.decode("utf-8"))
        assert isinstance(decoded, dict)
        return decoded


def _json_body(message: Mapping[str, Any]) -> bytes:
    return json.dumps(message).encode("utf-8")


def _auth_receipts(recorder: TrajectoryRecorder) -> list[Receipt]:
    return recorder.filter_receipts(
        event_kinds="claude_mcp_http",
        signal_types=AUTH_SIGNAL_TYPE,
    )


def _serialized_receipts(receipts: list[Receipt]) -> str:
    return json.dumps([receipt.to_dict() for receipt in receipts], default=str)


def _assert_receipts_exclude(receipts: list[Receipt], *values: str) -> None:
    serialized = _serialized_receipts(receipts)
    for value in values:
        assert value not in serialized


async def _call_http_app(
    app: ClaudeMeshMCPHTTPApp,
    *,
    method: str = "POST",
    path: str = "/mcp",
    headers: Mapping[str, str] | None = None,
    extra_headers: list[tuple[str, str]] | None = None,
    query_string: str = "",
    body: bytes = b"",
) -> ASGIResponse:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    raw_headers.extend(
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (extra_headers or [])
    )
    messages = [{
        "type": "http.request",
        "body": body,
        "more_body": False,
    }]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": query_string.encode("latin-1"),
            "headers": raw_headers,
        },
        receive,
        send,
    )

    assert len(sent) == 2
    start = sent[0]
    body_message = sent[1]
    response_headers = {
        bytes(name).decode("latin-1"): bytes(value).decode("latin-1")
        for name, value in start["headers"]
    }
    response_body = body_message.get("body", b"")
    assert isinstance(response_body, bytes)
    return ASGIResponse(
        status=int(start["status"]),
        headers=response_headers,
        body=response_body,
    )


def test_protected_resource_metadata_url_inserts_well_known_before_resource_path() -> None:
    assert protected_resource_metadata_url("https://example.com/mcp") == (
        "https://example.com/.well-known/oauth-protected-resource/mcp"
    )
    assert protected_resource_metadata_url("https://example.com") == (
        "https://example.com/.well-known/oauth-protected-resource"
    )
    assert protected_resource_metadata_url("https://example.com/mcp?tenant=alpha") == (
        "https://example.com/.well-known/oauth-protected-resource/mcp?tenant=alpha"
    )


def test_protected_resource_metadata_shapes_rfc9728_json() -> None:
    metadata = ClaudeMCPProtectedResourceMetadata(
        resource="https://example.com/mcp",
        authorization_servers=("https://auth.example.com",),
        scopes_supported=("tools.read", "tools.call"),
        resource_documentation="https://example.com/docs",
        extra={
            "access_token": "raw",
            "bearer_token": "raw",
            "client_secret": "raw",
            "refresh_token": "raw",
            "service_tier": "prod",
        },
    )

    assert metadata.metadata_url() == "https://example.com/.well-known/oauth-protected-resource/mcp"
    assert metadata.to_dict() == {
        "resource": "https://example.com/mcp",
        "authorization_servers": ["https://auth.example.com"],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["tools.read", "tools.call"],
        "resource_name": "Signal Gating Protocol MCP",
        "resource_documentation": "https://example.com/docs",
        "service_tier": "prod",
    }
    assert "raw" not in json.dumps(metadata.to_dict())


def test_protected_resource_metadata_rejects_non_https_resource_identifiers() -> None:
    with pytest.raises(ValueError, match="https URL"):
        ClaudeMCPProtectedResourceMetadata(
            resource="http://example.com/mcp",
            authorization_servers=("https://auth.example.com",),
        )

    with pytest.raises(ValueError, match="https URL"):
        protected_resource_metadata_url("https://example.com/mcp#fragment")

    with pytest.raises(ValueError, match="authorization_servers"):
        ClaudeMCPProtectedResourceMetadata(
            resource="https://example.com/mcp",
            authorization_servers=(),
        )


def test_protected_resource_metadata_mapping_coercion_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="resource"):
        ClaudeMeshMCPHTTPApp(
            ClaudeMeshMCPAdapter(Mesh()),
            protected_resource_metadata={"authorization_servers": ["https://auth.example.com"]},
        )

    with pytest.raises(ValueError, match="authorization_servers"):
        ClaudeMeshMCPHTTPApp(
            ClaudeMeshMCPAdapter(Mesh()),
            protected_resource_metadata={
                "resource": "https://example.com/mcp",
                "authorization_servers": "https://auth.example.com",
            },
        )


async def test_claude_mesh_mcp_http_app_serves_protected_resource_metadata() -> None:
    metadata = ClaudeMCPProtectedResourceMetadata(
        resource="https://example.com/mcp",
        authorization_servers=("https://auth.example.com",),
        scopes_supported=("tools.call",),
    )
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        authorize_http=lambda context: True,
        protected_resource_metadata=metadata,
    )

    response = await _call_http_app(
        app,
        method="GET",
        path="/.well-known/oauth-protected-resource/mcp",
    )
    challenge = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["resource"] == "https://example.com/mcp"
    assert response.json()["authorization_servers"] == ["https://auth.example.com"]
    assert response.json()["bearer_methods_supported"] == ["header"]
    assert response.json()["scopes_supported"] == ["tools.call"]
    assert challenge.status == 401
    assert challenge.headers["www-authenticate"] == (
        'Bearer realm="mcp", '
        'resource_metadata="https://example.com/.well-known/oauth-protected-resource/mcp"'
    )


async def test_claude_mesh_mcp_http_app_metadata_path_allows_get_only() -> None:
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        protected_resource_metadata={
            "resource": "https://example.com/mcp",
            "authorization_servers": ["https://auth.example.com"],
        },
    )

    response = await _call_http_app(
        app,
        method="POST",
        path="/.well-known/oauth-protected-resource/mcp",
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 405
    assert response.headers["allow"] == "GET"
    assert response.body == b""


def test_claude_mesh_mcp_http_app_rejects_unserved_metadata_url_path() -> None:
    metadata = ClaudeMCPProtectedResourceMetadata(
        resource="https://example.com/mcp",
        authorization_servers=("https://auth.example.com",),
    )

    with pytest.raises(ValueError, match="must be served"):
        ClaudeMeshMCPHTTPApp(
            ClaudeMeshMCPAdapter(Mesh()),
            protected_resource_metadata=metadata,
            protected_resource_metadata_url="https://example.com/.well-known/oauth-protected-resource/mcp",
            protected_resource_metadata_path="/wrong",
        )


def test_claude_mesh_mcp_http_app_rejects_metadata_path_collision() -> None:
    metadata = ClaudeMCPProtectedResourceMetadata(
        resource="https://example.com/mcp",
        authorization_servers=("https://auth.example.com",),
    )

    with pytest.raises(ValueError, match="must not equal"):
        ClaudeMeshMCPHTTPApp(
            ClaudeMeshMCPAdapter(Mesh()),
            protected_resource_metadata=metadata,
            protected_resource_metadata_path="/mcp",
        )


async def test_claude_mesh_mcp_http_app_returns_initialize_json_response() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh(), server_name="sgp"))

    response = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    payload = response.json()
    assert response.status == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.headers["mcp-session-id"].isascii()
    assert payload["id"] == 1
    assert payload["result"]["protocolVersion"] == "2025-06-18"
    assert payload["result"]["serverInfo"]["name"] == "sgp"


async def test_claude_mesh_mcp_http_app_notifications_return_accepted() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_headers = {**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]}
    initialized = await _call_http_app(
        app,
        headers=session_headers,
        body=_json_body({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    tools = await _call_http_app(
        app,
        headers=session_headers,
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )

    assert initialized.status == 202
    assert initialized.body == b""
    assert tools.status == 200
    assert tools.json()["result"] == {"tools": []}


async def test_claude_mesh_mcp_http_app_calls_running_mesh_tool() -> None:
    worker = Agent("worker")

    @worker.tool(description="Echo text")
    async def echo(text: str) -> dict[str, str]:
        return {"echo": text}

    mesh = Mesh([worker])
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh))

    async with mesh:
        initialize = await _call_http_app(
            app,
            headers=POST_HEADERS,
            body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        )
        session_headers = {**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]}
        await _call_http_app(
            app,
            headers=session_headers,
            body=_json_body({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        )
        response = await _call_http_app(
            app,
            headers=session_headers,
            body=_json_body({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hello"}},
            }),
        )

    payload = response.json()
    assert response.status == 200
    assert payload["result"]["structuredContent"] == {"echo": "hello"}
    assert payload["result"]["isError"] is False


async def test_claude_mesh_mcp_http_app_rejects_get_sse_streams() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))
    initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    response = await _call_http_app(
        app,
        method="GET",
        headers={**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]},
    )

    assert response.status == 405
    assert response.headers["allow"] == "POST"
    assert response.body == b""


async def test_claude_mesh_mcp_http_app_deletes_sessions() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))
    initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_headers = {**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]}

    deleted = await _call_http_app(app, method="DELETE", headers=session_headers)
    after_delete = await _call_http_app(
        app,
        headers=session_headers,
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )

    assert deleted.status == 204
    assert deleted.body == b""
    assert after_delete.status == 404
    assert after_delete.json()["error"]["message"] == "Unknown MCP session"


async def test_claude_mesh_mcp_http_app_requires_sessions_after_initialize() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    missing = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    unknown = await _call_http_app(
        app,
        headers={**POST_HEADERS, "mcp-session-id": "missing"},
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )

    assert missing.status == 400
    assert missing.json()["error"]["message"] == "Missing Mcp-Session-Id"
    assert unknown.status == 404
    assert unknown.json()["error"]["message"] == "Unknown MCP session"


async def test_claude_mesh_mcp_http_app_isolates_adapter_state_per_session() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    first_initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    second_initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "initialize"}),
    )
    first_headers = {**POST_HEADERS, "mcp-session-id": first_initialize.headers["mcp-session-id"]}
    second_headers = {
        **POST_HEADERS,
        "mcp-session-id": second_initialize.headers["mcp-session-id"],
    }

    await _call_http_app(
        app,
        headers=first_headers,
        body=_json_body({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    first_tools = await _call_http_app(
        app,
        headers=first_headers,
        body=_json_body({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    )
    second_tools = await _call_http_app(
        app,
        headers=second_headers,
        body=_json_body({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
    )

    assert first_initialize.headers["mcp-session-id"] != second_initialize.headers["mcp-session-id"]
    assert first_tools.json()["result"] == {"tools": []}
    assert second_tools.json()["error"]["code"] == -32002
    assert second_tools.json()["error"]["message"] == "MCP server is not initialized"


async def test_claude_mesh_mcp_http_app_rejects_unsupported_non_post_methods() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    response = await _call_http_app(app, method="PUT", headers=POST_HEADERS)

    assert response.status == 405
    assert response.headers["allow"] == "POST"
    assert response.body == b""


async def test_claude_mesh_mcp_http_app_rejects_bad_http_envelopes() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))
    initialize = _json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    wrong_path = await _call_http_app(
        app,
        path="/wrong",
        headers=POST_HEADERS,
        body=initialize,
    )
    missing_accept = await _call_http_app(
        app,
        headers={"content-type": "application/json"},
        body=initialize,
    )
    wrong_content_type = await _call_http_app(
        app,
        headers={"accept": "application/json, text/event-stream", "content-type": "text/plain"},
        body=initialize,
    )
    bad_protocol = await _call_http_app(
        app,
        headers={**POST_HEADERS, "mcp-protocol-version": "2024-11-05"},
        body=initialize,
    )
    bad_origin = await _call_http_app(
        app,
        headers={**POST_HEADERS, "origin": "https://attacker.example"},
        body=initialize,
    )

    assert wrong_path.status == 404
    assert missing_accept.status == 406
    assert missing_accept.json()["error"]["message"] == "Unsupported Accept header"
    assert wrong_content_type.status == 415
    assert bad_protocol.status == 400
    assert bad_protocol.json()["error"]["message"] == "Unsupported MCP protocol version"
    assert bad_origin.status == 403
    assert bad_origin.json()["error"]["message"] == "Forbidden origin"


async def test_claude_mesh_mcp_http_app_allows_localhost_origin_with_port() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "origin": "http://localhost:5173"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 200


async def test_claude_mesh_mcp_http_app_parse_shape_and_response_messages() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()))

    parse_error = await _call_http_app(app, headers=POST_HEADERS, body=b"{not-json}")
    invalid_shape = await _call_http_app(app, headers=POST_HEADERS, body=b"[]")
    initialize = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    jsonrpc_response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}),
    )

    assert parse_error.status == 400
    assert parse_error.json()["error"]["code"] == -32700
    assert invalid_shape.status == 400
    assert invalid_shape.json()["error"]["code"] == -32600
    assert jsonrpc_response.status == 202
    assert jsonrpc_response.body == b""


async def test_claude_mesh_mcp_http_app_auth_requires_bearer_token() -> None:
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        authorize_http=lambda context: True,
        protected_resource_metadata_url="https://example.com/.well-known/oauth-protected-resource",
    )

    response = await _call_http_app(
        app,
        headers=POST_HEADERS,
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 401
    assert response.json()["error"]["message"] == "Authorization required"
    assert response.headers["www-authenticate"] == (
        'Bearer realm="mcp", '
        'resource_metadata="https://example.com/.well-known/oauth-protected-resource"'
    )


async def test_claude_mesh_mcp_http_app_auth_rejects_query_string_tokens() -> None:
    seen: list[ClaudeMCPHTTPAuthorizationContext] = []

    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> bool:
        seen.append(context)
        return True

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer header-token"},
        query_string="access_token=query-token",
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 400
    assert response.json()["error"]["message"] == (
        "Access tokens must not be sent in the URI query string"
    )
    assert "query-token" not in response.body.decode("utf-8")
    assert all("query-token" not in value for value in response.headers.values())
    assert seen == []


async def test_claude_mesh_mcp_http_app_auth_rejects_malformed_authorization() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=lambda context: True)

    basic = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Basic abc"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    duplicate = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer first"},
        extra_headers=[("authorization", "Bearer second")],
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "initialize"}),
    )

    assert basic.status == 401
    assert basic.headers["www-authenticate"] == 'Bearer realm="mcp"'
    assert duplicate.status == 401
    assert duplicate.headers["www-authenticate"] == 'Bearer realm="mcp"'


async def test_claude_mesh_mcp_http_app_auth_propagates_custom_challenge() -> None:
    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=False,
            status_code=401,
            message="invalid token",
            www_authenticate='Bearer realm="custom", error="invalid_token"',
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer expired"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 401
    assert response.json()["error"]["message"] == "invalid token"
    assert response.headers["www-authenticate"] == 'Bearer realm="custom", error="invalid_token"'


async def test_claude_mesh_mcp_http_app_auth_empty_principal_denies() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=lambda context: "")

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-1"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 401
    assert response.json()["error"]["message"] == "Unauthorized"
    assert "token-1" not in response.body.decode("utf-8")


async def test_claude_mesh_mcp_http_app_auth_context_sanitizes_headers() -> None:
    seen: list[ClaudeMCPHTTPAuthorizationContext] = []

    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> str:
        seen.append(context)
        return "analyst"

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-1"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 200
    assert seen[0].authorization_scheme == "Bearer"
    assert seen[0].bearer_token == "token-1"
    assert "token-1" not in repr(seen[0])
    assert "authorization" not in seen[0].header_names
    assert seen[0].mcp_session_id == ""


async def test_claude_mesh_mcp_http_app_auth_records_allowed_receipt() -> None:
    mesh = Mesh()
    recorder = TrajectoryRecorder()
    mesh.record(recorder)

    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        assert context.bearer_token == "token-secret"
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal="alice",
            scopes=("tools.call", "tools.read"),
            audience="sgp-api",
            resource="https://example.com/mcp",
        )

    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(mesh),
        authorize_http=authorize,
        protected_resource_metadata_url=(
            "https://example.com/.well-known/oauth-protected-resource/mcp"
        ),
    )

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 200
    session_id = response.headers["mcp-session-id"]
    receipts = _auth_receipts(recorder)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.action == "claude_mcp_http_auth_allowed"
    assert receipt.event_kind == "claude_mcp_http"
    assert receipt.source == "claude_mcp_http"
    assert receipt.verify() is True
    signal = receipt.to_signal()
    assert isinstance(signal, ClaudeMCPHTTPAuthorizationSignal)
    assert receipt.payload == {
        "outcome": "allowed",
        "status_code": 200,
        "method": "POST",
        "path": "/mcp",
        "reason": "allowed",
        "jsonrpc_method": "",
        "authorization_scheme": "Bearer",
        "bearer_token_present": True,
        "principal_hash": signal.principal_hash,
        "principal_present": True,
        "audience_present": True,
        "resource_present": True,
        "scope_count": 2,
        "identity_binding_kind": "claims",
        "mcp_session_id_hash": "",
        "mcp_session_present": False,
        "protected_resource_metadata_advertised": True,
    }
    assert signal.principal_hash.startswith("sha256:")
    assert receipt.metadata["principal_hash"] == signal.principal_hash
    assert receipt.metadata["protected_resource_metadata_advertised"] is True
    _assert_receipts_exclude(
        receipts,
        "token-secret",
        "alice",
        "tools.call",
        "tools.read",
        "sgp-api",
        "https://example.com/mcp",
        session_id,
    )


async def test_claude_mesh_mcp_http_app_auth_records_query_token_rejection() -> None:
    seen: list[ClaudeMCPHTTPAuthorizationContext] = []
    mesh = Mesh()
    recorder = TrajectoryRecorder()
    mesh.record(recorder)

    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> bool:
        seen.append(context)
        return True

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), authorize_http=authorize)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer header-token"},
        query_string="access_token=query-secret",
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    receipts = _auth_receipts(recorder)
    assert response.status == 400
    assert seen == []
    assert len(receipts) == 1
    assert receipts[0].action == "claude_mcp_http_auth_query_token_rejected"
    assert receipts[0].payload["outcome"] == "denied"
    assert receipts[0].payload["status_code"] == 400
    assert receipts[0].payload["reason"] == "query_token_rejected"
    assert receipts[0].payload["authorization_scheme"] == "Bearer"
    assert receipts[0].payload["bearer_token_present"] is True
    assert receipts[0].payload["identity_binding_kind"] == "none"
    assert receipts[0].verify() is True
    _assert_receipts_exclude(receipts, "query-secret", "header-token")


async def test_claude_mesh_mcp_http_app_auth_records_denied_receipt() -> None:
    mesh = Mesh()
    recorder = TrajectoryRecorder()
    mesh.record(recorder)

    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=False,
            status_code=403,
            message="raw denial alice token-secret",
            www_authenticate='Bearer error="insufficient_scope", scope="tools.call"',
            principal="alice",
            scopes=("tools.read",),
            audience="sgp-api",
            resource="https://example.com/mcp",
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), authorize_http=authorize)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    receipts = _auth_receipts(recorder)
    assert response.status == 403
    assert len(receipts) == 1
    assert receipts[0].action == "claude_mcp_http_auth_denied"
    assert receipts[0].payload["outcome"] == "denied"
    assert receipts[0].payload["status_code"] == 403
    assert receipts[0].payload["reason"] == "insufficient_scope"
    assert receipts[0].payload["principal_present"] is True
    assert receipts[0].payload["principal_hash"].startswith("sha256:")
    assert receipts[0].payload["scope_count"] == 1
    assert receipts[0].verify() is True
    _assert_receipts_exclude(
        receipts,
        "raw denial",
        "alice",
        "token-secret",
        "tools.call",
        "tools.read",
        "sgp-api",
        "https://example.com/mcp",
    )


async def test_claude_mesh_mcp_http_app_auth_records_session_mismatch_receipt() -> None:
    mesh = Mesh()
    recorder = TrajectoryRecorder()
    mesh.record(recorder)

    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> str:
        if context.bearer_token == "alice-token":
            return "alice"
        return "bob"

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer alice-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]

    response = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer bob-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )

    receipts = _auth_receipts(recorder)
    assert response.status == 403
    assert [receipt.action for receipt in receipts] == [
        "claude_mcp_http_auth_allowed",
        "claude_mcp_http_auth_allowed",
        "claude_mcp_http_auth_session_mismatch",
    ]
    mismatch = receipts[-1]
    assert mismatch.payload["outcome"] == "denied"
    assert mismatch.payload["status_code"] == 403
    assert mismatch.payload["reason"] == "session_authorization_mismatch"
    assert mismatch.payload["jsonrpc_method"] == "tools/list"
    assert mismatch.payload["authorization_scheme"] == "Bearer"
    assert mismatch.payload["bearer_token_present"] is True
    assert mismatch.payload["mcp_session_present"] is True
    assert mismatch.payload["mcp_session_id_hash"].startswith("sha256:")
    assert mismatch.payload["identity_binding_kind"] == "claims"
    assert mismatch.payload["principal_present"] is True
    assert mismatch.payload["principal_hash"].startswith("sha256:")
    assert mismatch.metadata["jsonrpc_method"] == "tools/list"
    assert mismatch.metadata["bearer_token_present"] is True
    assert mismatch.verify() is True
    _assert_receipts_exclude(
        receipts,
        "alice-token",
        "bob-token",
        "alice",
        "bob",
        session_id,
    )


async def test_claude_mesh_mcp_http_app_auth_receipts_export_and_replay(
    tmp_path: Path,
) -> None:
    mesh = Mesh()
    recorder = TrajectoryRecorder()
    mesh.record(recorder)

    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal="alice",
            scopes=("tools.call",),
            audience="sgp-api",
            resource="https://example.com/mcp",
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), authorize_http=authorize)
    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = response.headers["mcp-session-id"]
    receipts = _auth_receipts(recorder)
    assert len(receipts) == 1

    out = tmp_path / "auth-receipts.jsonl"
    assert recorder.export_jsonl(out) == 1
    text = out.read_text(encoding="utf-8")
    for raw in (
        "token-secret",
        "alice",
        "tools.call",
        "sgp-api",
        "https://example.com/mcp",
        session_id,
    ):
        assert raw not in text

    reloaded = TrajectoryRecorder()
    assert reloaded.load_jsonl(out) == 1
    reloaded_receipts = _auth_receipts(reloaded)
    assert all(receipt.verify() for receipt in reloaded_receipts)
    signals = reloaded.replay(
        event_kinds="claude_mcp_http",
        signal_types=AUTH_SIGNAL_TYPE,
    )
    assert len(signals) == 1
    assert isinstance(signals[0], ClaudeMCPHTTPAuthorizationSignal)
    assert signals[0].outcome == "allowed"

    tampered_data = reloaded_receipts[0].to_dict()
    tampered_data["payload"] = {**reloaded_receipts[0].payload, "status_code": 500}
    tampered = Receipt.from_dict(tampered_data)
    assert tampered.verify() is False
    tampered_out = tmp_path / "tampered-auth-receipts.jsonl"
    tampered_out.write_text(json.dumps(tampered_data) + "\n", encoding="utf-8")
    rejected = TrajectoryRecorder()
    with pytest.raises(SignalSerializationError, match="receipt digest mismatch"):
        rejected.load_jsonl(tampered_out)
    assert rejected.receipts == []


async def test_claude_mesh_mcp_http_app_filters_auth_receipts_from_mixed_recorder(
    tmp_path: Path,
) -> None:
    worker = Agent("worker")
    mesh = Mesh([worker])
    recorder = TrajectoryRecorder()
    mesh.record(recorder)
    seen_tokens: list[str] = []

    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        seen_tokens.append(context.bearer_token)
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal="alice",
            scopes=("tools.call",),
            audience="sgp-api",
            resource="https://example.com/mcp",
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), authorize_http=authorize)

    async with mesh:
        await mesh.inject(
            worker,
            Signal(metadata={"mixed_recorder_marker": "ordinary-trajectory-secret"}),
        )
        allowed = await _call_http_app(
            app,
            headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
            body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        )
        rejected = await _call_http_app(
            app,
            headers={**POST_HEADERS, "authorization": "Bearer header-token"},
            query_string="access_token=query-secret",
            body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "initialize"}),
        )

    session_id = allowed.headers["mcp-session-id"]
    all_receipts = recorder.receipts
    auth_receipts = _auth_receipts(recorder)

    assert allowed.status == 200
    assert rejected.status == 400
    assert seen_tokens == ["token-secret"]
    assert [receipt.action for receipt in all_receipts] == [
        "inject",
        "claude_mcp_http_auth_allowed",
        "claude_mcp_http_auth_query_token_rejected",
    ]
    assert [receipt.signal_type for receipt in auth_receipts] == [
        AUTH_SIGNAL_TYPE,
        AUTH_SIGNAL_TYPE,
    ]
    assert "ordinary-trajectory-secret" in _serialized_receipts(all_receipts)

    out = tmp_path / "filtered-auth-receipts.jsonl"
    assert recorder.export_jsonl(
        out,
        event_kinds="claude_mcp_http",
        signal_types=AUTH_SIGNAL_TYPE,
        verify=True,
    ) == len(auth_receipts)
    text = out.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.strip().splitlines()]
    assert len(text.strip().splitlines()) == 2
    assert AUTH_SIGNAL_TYPE in text
    assert '"signal_type": "Signal"' not in text
    assert [row["event_kind"] for row in rows] == [
        "claude_mcp_http",
        "claude_mcp_http",
    ]
    assert [row["source"] for row in rows] == [
        "claude_mcp_http",
        "claude_mcp_http",
    ]
    assert [row["signal_type"] for row in rows] == [AUTH_SIGNAL_TYPE, AUTH_SIGNAL_TYPE]
    assert [row["action"] for row in rows] == [
        "claude_mcp_http_auth_allowed",
        "claude_mcp_http_auth_query_token_rejected",
    ]
    assert [row["payload"]["bearer_token_present"] for row in rows] == [True, True]
    assert [row["payload"]["principal_present"] for row in rows] == [True, False]
    assert all(Receipt.from_dict(row).verify() for row in rows)
    for raw in (
        "token-secret",
        "header-token",
        "query-secret",
        "alice",
        "tools.call",
        "sgp-api",
        "https://example.com/mcp",
        session_id,
        "ordinary-trajectory-secret",
    ):
        assert raw not in text

    reloaded = TrajectoryRecorder()
    assert reloaded.load_jsonl(out) == 2
    assert [receipt.action for receipt in reloaded.receipts] == [
        "claude_mcp_http_auth_allowed",
        "claude_mcp_http_auth_query_token_rejected",
    ]
    assert all(receipt.verify() for receipt in reloaded.receipts)
    assert TrajectoryReplayRunner.from_recorder(reloaded).replayable_receipts() == []

    signals = reloaded.replay()
    assert all(isinstance(signal, ClaudeMCPHTTPAuthorizationSignal) for signal in signals)
    auth_signals = [
        signal
        for signal in signals
        if isinstance(signal, ClaudeMCPHTTPAuthorizationSignal)
    ]
    assert [signal.outcome for signal in auth_signals] == ["allowed", "denied"]
    assert [signal.status_code for signal in auth_signals] == [200, 400]
    assert [signal.reason for signal in auth_signals] == [
        "allowed",
        "query_token_rejected",
    ]

    tampered_lines = [json.loads(line) for line in text.strip().splitlines()]
    tampered_lines[0]["wire"]["data"]["status_code"] = 500
    assert Receipt.from_dict(tampered_lines[0]).verify() is False
    tampered_out = tmp_path / "tampered-filtered-auth-receipts.jsonl"
    tampered_out.write_text(
        "".join(json.dumps(line) + "\n" for line in tampered_lines),
        encoding="utf-8",
    )
    rejected_reload = TrajectoryRecorder()
    with pytest.raises(SignalSerializationError, match="receipt digest mismatch"):
        rejected_reload.load_jsonl(tampered_out)
    assert rejected_reload.receipts == []


async def test_claude_mesh_mcp_http_app_auth_denials_use_status_codes() -> None:
    def deny_scope(context: ClaudeMCPHTTPAuthorizationContext) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=False,
            status_code=403,
            message=f"insufficient scope for {context.method}",
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=deny_scope)

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer scoped-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 403
    assert response.json()["error"]["message"] == "insufficient scope for POST"
    assert "www-authenticate" not in response.headers


async def test_claude_mesh_mcp_http_app_auth_binds_sessions_to_principal() -> None:
    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> str:
        if context.bearer_token == "alice-token":
            return "alice"
        return "bob"

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer alice-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]

    await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer alice-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    same_principal = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer alice-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    different_principal = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer bob-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    )

    assert same_principal.status == 200
    assert same_principal.json()["result"] == {"tools": []}
    assert different_principal.status == 403
    assert different_principal.json()["error"]["message"] == "Session authorization mismatch"


async def test_claude_mesh_mcp_http_app_auth_binds_sessions_to_claims() -> None:
    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        if context.bearer_token == "wide-token":
            scopes: Any = ("tools.call", "tools.read")
            audience = "sgp-api"
            resource = "https://example.com/mcp"
        elif context.bearer_token == "wide-reordered-token":
            scopes = ("tools.read", "tools.call", "tools.call")
            audience = "sgp-api"
            resource = "https://example.com/mcp"
        elif context.bearer_token == "wide-string-token":
            scopes = "tools.read tools.call"
            audience = "sgp-api"
            resource = "https://example.com/mcp"
        elif context.bearer_token == "narrow-token":
            scopes = ("tools.read",)
            audience = "sgp-api"
            resource = "https://example.com/mcp"
        elif context.bearer_token == "other-audience-token":
            scopes = ("tools.call", "tools.read")
            audience = "other-api"
            resource = "https://example.com/mcp"
        else:
            scopes = ("tools.call", "tools.read")
            audience = "sgp-api"
            resource = "https://other.example.com/mcp"
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal="alice",
            scopes=scopes,
            audience=audience,
            resource=resource,
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer wide-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]
    stored_identity = app._session_authorizations[session_id]

    await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer wide-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    reordered_scopes = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer wide-reordered-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    string_scopes = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer wide-string-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    )
    reduced_scopes = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer narrow-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
    )
    changed_audience = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer other-audience-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}),
    )
    changed_resource = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer other-resource-token",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 6, "method": "tools/list"}),
    )

    assert stored_identity.startswith("claims:")
    assert "wide-token" not in stored_identity
    assert json.loads(stored_identity.removeprefix("claims:")) == {
        "audience": "sgp-api",
        "principal": "alice",
        "resource": "https://example.com/mcp",
        "scopes": ["tools.call", "tools.read"],
        "version": 1,
    }
    assert reordered_scopes.status == 200
    assert reordered_scopes.json()["result"] == {"tools": []}
    assert string_scopes.status == 200
    assert string_scopes.json()["result"] == {"tools": []}
    assert reduced_scopes.status == 403
    assert changed_audience.status == 403
    assert changed_resource.status == 403
    for response in (reduced_scopes, changed_audience, changed_resource):
        assert response.json()["error"]["message"] == "Session authorization mismatch"
        assert "token" not in response.body.decode("utf-8")


async def test_claude_mesh_mcp_http_app_auth_claim_identity_is_collision_safe() -> None:
    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        if context.bearer_token == "plain":
            return ClaudeMCPHTTPAuthorizationResult(
                allowed=True,
                principal="alice",
                audience="audience=x",
                resource="https://example.com/mcp",
                scopes=("tools.read",),
            )
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal='alice","audience":"audience=x',
            audience="",
            resource="https://example.com/mcp",
            scopes=("tools.read",),
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer plain"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]

    forged = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer forged",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )

    assert forged.status == 403
    assert forged.json()["error"]["message"] == "Session authorization mismatch"


async def test_claude_mesh_mcp_http_app_auth_binds_get_and_delete_sessions() -> None:
    def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> str:
        if context.bearer_token == "alice-token":
            return "alice"
        return "bob"

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer alice-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]
    mismatched_get = await _call_http_app(
        app,
        method="GET",
        headers={
            "authorization": "Bearer bob-token",
            "mcp-session-id": session_id,
        },
    )
    mismatched_delete = await _call_http_app(
        app,
        method="DELETE",
        headers={
            "authorization": "Bearer bob-token",
            "mcp-session-id": session_id,
        },
    )
    original_principal = await _call_http_app(
        app,
        method="DELETE",
        headers={
            "authorization": "Bearer alice-token",
            "mcp-session-id": session_id,
        },
    )

    assert mismatched_get.status == 403
    assert mismatched_get.json()["error"]["message"] == "Session authorization mismatch"
    assert mismatched_delete.status == 403
    assert mismatched_delete.json()["error"]["message"] == "Session authorization mismatch"
    assert original_principal.status == 204


async def test_claude_mesh_mcp_http_app_auth_protects_get_and_delete() -> None:
    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=lambda context: "user")
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-1"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_headers = {**POST_HEADERS, "mcp-session-id": initialize.headers["mcp-session-id"]}

    get_without_auth = await _call_http_app(app, method="GET", headers=session_headers)
    delete_without_auth = await _call_http_app(app, method="DELETE", headers=session_headers)

    assert get_without_auth.status == 401
    assert get_without_auth.headers["www-authenticate"] == 'Bearer realm="mcp"'
    assert delete_without_auth.status == 401
    assert delete_without_auth.headers["www-authenticate"] == 'Bearer realm="mcp"'


async def test_claude_mesh_mcp_http_app_auth_fingerprints_tokens_without_storing_raw() -> None:
    async def authorize(context: ClaudeMCPHTTPAuthorizationContext) -> bool:
        return context.bearer_token.startswith("token-")

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]
    stored_identity = app._session_authorizations[session_id]
    same_token = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer token-secret",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    different_token = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer token-other",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    )

    assert stored_identity.startswith("sha256:")
    assert "token-secret" not in stored_identity
    assert same_token.json()["error"]["code"] == -32002
    assert different_token.status == 403


async def test_claude_mesh_mcp_http_app_auth_fingerprints_tokens_without_principal() -> None:
    def authorize(
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=context.bearer_token.startswith("token-"),
            audience="sgp-api",
            resource="https://example.com/mcp",
            scopes=("tools.call",),
        )

    app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(Mesh()), authorize_http=authorize)
    initialize = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer token-secret"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    session_id = initialize.headers["mcp-session-id"]
    stored_identity = app._session_authorizations[session_id]
    same_token = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer token-secret",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    different_token = await _call_http_app(
        app,
        headers={
            **POST_HEADERS,
            "authorization": "Bearer token-other",
            "mcp-session-id": session_id,
        },
        body=_json_body({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    )

    assert stored_identity.startswith("sha256:")
    assert "token-secret" not in stored_identity
    assert same_token.json()["error"]["code"] == -32002
    assert different_token.status == 403


def test_claude_mesh_mcp_http_app_exports_from_package_root() -> None:
    import signal_gating
    from signal_gating.integrations import (
        ClaudeMCPHTTPAuthorizationSignal as IntegrationAuthorizationSignal,
    )
    from signal_gating.integrations import (
        ClaudeMCPProtectedResourceMetadata as IntegrationMetadata,
    )
    from signal_gating.integrations import (
        claude,
    )
    from signal_gating.integrations import (
        protected_resource_metadata_url as integration_url,
    )

    assert hasattr(signal_gating, "ClaudeMeshMCPHTTPApp")
    assert hasattr(signal_gating, "ClaudeMCPBearerTokenValidator")
    assert hasattr(signal_gating, "ClaudeMCPHTTPAuthorizationContext")
    assert hasattr(signal_gating, "ClaudeMCPHTTPAuthorizationDecision")
    assert hasattr(signal_gating, "ClaudeMCPHTTPAuthorizationResult")
    assert hasattr(signal_gating, "ClaudeMCPHTTPAuthorizationSignal")
    assert hasattr(signal_gating, "ClaudeMCPHTTPAuthorizeFn")
    assert hasattr(signal_gating, "ClaudeMCPJWKSCache")
    assert hasattr(signal_gating, "ClaudeMCPJWTBearerAuthorizer")
    assert hasattr(signal_gating, "ClaudeMCPProtectedResourceMetadata")
    assert hasattr(signal_gating, "ClaudeMCPTokenClaims")
    assert hasattr(signal_gating, "ClaudeMCPTokenDecodeFn")
    assert hasattr(signal_gating, "protected_resource_metadata_url")
    assert hasattr(claude, "ClaudeMeshMCPHTTPApp")
    assert hasattr(claude, "ClaudeMCPBearerTokenValidator")
    assert hasattr(claude, "ClaudeMCPHTTPAuthorizationContext")
    assert hasattr(claude, "ClaudeMCPHTTPAuthorizationDecision")
    assert hasattr(claude, "ClaudeMCPHTTPAuthorizationResult")
    assert hasattr(claude, "ClaudeMCPHTTPAuthorizationSignal")
    assert hasattr(claude, "ClaudeMCPHTTPAuthorizeFn")
    assert hasattr(claude, "ClaudeMCPJWKSCache")
    assert hasattr(claude, "ClaudeMCPJWTBearerAuthorizer")
    assert hasattr(claude, "ClaudeMCPProtectedResourceMetadata")
    assert hasattr(claude, "ClaudeMCPTokenClaims")
    assert hasattr(claude, "ClaudeMCPTokenDecodeFn")
    assert hasattr(claude, "protected_resource_metadata_url")
    assert "ClaudeMeshMCPHTTPApp" in signal_gating.__all__
    assert "ClaudeMCPBearerTokenValidator" in signal_gating.__all__
    assert "ClaudeMCPHTTPAuthorizationContext" in signal_gating.__all__
    assert "ClaudeMCPHTTPAuthorizationDecision" in signal_gating.__all__
    assert "ClaudeMCPHTTPAuthorizationResult" in signal_gating.__all__
    assert "ClaudeMCPHTTPAuthorizationSignal" in signal_gating.__all__
    assert "ClaudeMCPHTTPAuthorizeFn" in signal_gating.__all__
    assert "ClaudeMCPJWKSCache" in signal_gating.__all__
    assert "ClaudeMCPJWTBearerAuthorizer" in signal_gating.__all__
    assert "ClaudeMCPProtectedResourceMetadata" in signal_gating.__all__
    assert "ClaudeMCPTokenClaims" in signal_gating.__all__
    assert "ClaudeMCPTokenDecodeFn" in signal_gating.__all__
    assert "protected_resource_metadata_url" in signal_gating.__all__
    assert "ClaudeMeshMCPHTTPApp" in claude.__all__
    assert "ClaudeMCPBearerTokenValidator" in claude.__all__
    assert "ClaudeMCPHTTPAuthorizationContext" in claude.__all__
    assert "ClaudeMCPHTTPAuthorizationDecision" in claude.__all__
    assert "ClaudeMCPHTTPAuthorizationResult" in claude.__all__
    assert "ClaudeMCPHTTPAuthorizationSignal" in claude.__all__
    assert "ClaudeMCPHTTPAuthorizeFn" in claude.__all__
    assert "ClaudeMCPJWKSCache" in claude.__all__
    assert "ClaudeMCPJWTBearerAuthorizer" in claude.__all__
    assert "ClaudeMCPProtectedResourceMetadata" in claude.__all__
    assert "ClaudeMCPTokenClaims" in claude.__all__
    assert "ClaudeMCPTokenDecodeFn" in claude.__all__
    assert "protected_resource_metadata_url" in claude.__all__

    decision: ClaudeMCPHTTPAuthorizationDecision = True
    assert decision is True
    assert IntegrationAuthorizationSignal is ClaudeMCPHTTPAuthorizationSignal
    assert IntegrationMetadata is ClaudeMCPProtectedResourceMetadata
    assert integration_url is protected_resource_metadata_url
