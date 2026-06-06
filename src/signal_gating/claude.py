"""Claude Agent SDK integration boundary.

This module keeps Claude Agent SDK sessions as an external runtime while making
their prompts, results, tool events, and permission decisions visible as typed
SGP signals. Core users do not pay the optional dependency cost unless they
instantiate ``ClaudeAgent`` without an injected query function.
"""

from __future__ import annotations

import inspect
import json
import secrets
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import parse_qsl, urlparse, urlunparse
from urllib.request import urlopen

from pydantic import Field

from signal_gating.agent import Agent
from signal_gating.errors import AgentError
from signal_gating.gate import Gate
from signal_gating.signal import Signal

ClaudeToolEventKind = Literal["tool_use", "tool_result"]
ClaudePermissionDecision = Literal["allowed", "denied", "prompted", "unknown"]
Render = Callable[[Signal], str]
ASGIMessage = dict[str, Any]
ASGIScope = Mapping[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ClaudeMCPTokenClaims = Mapping[str, Any]
ClaudeMCPTokenDecodeFn = Callable[
    [str],
    ClaudeMCPTokenClaims | Awaitable[ClaudeMCPTokenClaims],
]
ClaudeMCPJWKSLoader = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class ClaudeQueryFn(Protocol):
    """Structural type for ``claude_agent_sdk.query``."""

    def __call__(
        self,
        *,
        prompt: str,
        options: Any | None = None,
    ) -> AsyncIterator[Any]: ...


class ClaudeOptionsFactory(Protocol):
    """Structural type for ``claude_agent_sdk.ClaudeAgentOptions``."""

    def __call__(self, **kwargs: Any) -> Any: ...


class ClaudeClientFactory(Protocol):
    """Structural type for ``claude_agent_sdk.ClaudeSDKClient``."""

    def __call__(self, *, options: Any | None = None) -> Any: ...


class ClaudeAgentRunSignal(Signal):
    """Prompt an external Claude Agent SDK run from inside an SGP mesh."""

    __signal_type__ = "sgp.integrations.claude.run.v1"

    prompt: str
    session_id: str = ""
    continue_conversation: bool | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    permission_mode: str = ""
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    cwd: str = ""


class ClaudeAgentResultSignal(Signal):
    """Result of a Claude Agent SDK run, correlated to the external session."""

    __signal_type__ = "sgp.integrations.claude.result.v1"

    text: str = ""
    session_id: str = ""
    subtype: str = ""
    total_cost_usd: float | None = None
    message_count: int = 0
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    permission_mode: str = ""
    mcp_servers: list[str] = Field(default_factory=list)
    resumed_from_session_id: str = ""
    continued: bool = False


class ClaudeToolEventSignal(Signal):
    """Tool-use evidence surfaced from Claude Agent SDK stream messages."""

    __signal_type__ = "sgp.integrations.claude.tool_event.v1"

    event: ClaudeToolEventKind
    tool_name: str
    session_id: str = ""
    tool_call_id: str = ""
    parent_tool_use_id: str = ""
    mcp_server: str = ""
    status: str = ""
    tool_input_keys: list[str] = Field(default_factory=list)


class ClaudePermissionDecisionSignal(Signal):
    """A typed audit signal for external Claude tool permission decisions."""

    __signal_type__ = "sgp.integrations.claude.permission_decision.v1"

    tool_name: str
    decision: ClaudePermissionDecision = "unknown"
    session_id: str = ""
    permission_mode: str = ""
    reason: str = ""


class ClaudeMCPHTTPAuthorizationSignal(Signal):
    """Sanitized audit signal for HTTP MCP authorization decisions."""

    __signal_type__ = "sgp.integrations.claude.mcp_http_authorization.v1"

    outcome: str
    status_code: int
    method: str
    path: str
    reason: str
    jsonrpc_method: str = ""
    authorization_scheme: str = ""
    bearer_token_present: bool = False
    principal_hash: str = ""
    principal_present: bool = False
    audience_present: bool = False
    resource_present: bool = False
    scope_count: int = 0
    identity_binding_kind: str = "none"
    mcp_session_id_hash: str = ""
    mcp_session_present: bool = False
    protected_resource_metadata_advertised: bool = False


class ClaudeToolRequestSignal(Signal):
    """Sanitized permission request for a Claude tool call."""

    __signal_type__ = "sgp.integrations.claude.tool_request.v1"

    tool_name: str
    tool_input_keys: list[str] = Field(default_factory=list)
    mcp_server: str = ""
    permission_mode: str = ""
    blocked_path: str = ""
    decision_reason: str = ""
    display_name: str = ""


@dataclass(slots=True)
class ClaudeAgentSDKResult:
    """Summary returned by :class:`ClaudeAgentSDKRunner`."""

    session_id: str
    result: str
    subtype: str = ""
    total_cost_usd: float | None = None
    message_count: int = 0


@dataclass(slots=True)
class ClaudePermissionResultAllowFallback:
    """Fallback shape matching Claude SDK's ``PermissionResultAllow``."""

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[Any] | None = None


@dataclass(slots=True)
class ClaudePermissionResultDenyFallback:
    """Fallback shape matching Claude SDK's ``PermissionResultDeny``."""

    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False


@dataclass(frozen=True, slots=True)
class ClaudeMCPHTTPAuthorizationContext:
    """Sanitized HTTP authorization input for MCP Streamable HTTP requests."""

    method: str
    path: str
    origin: str = ""
    mcp_session_id: str = ""
    authorization_scheme: str = ""
    bearer_token: str = dataclass_field(default="", repr=False)
    header_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaudeMCPHTTPAuthorizationResult:
    """Decision returned by an MCP HTTP authorization hook."""

    allowed: bool
    principal: str = ""
    scopes: tuple[str, ...] = ()
    audience: str = ""
    resource: str = ""
    status_code: int = 401
    message: str = ""
    www_authenticate: str = ""


class ClaudeMCPJWKSCache:
    """Small async-friendly cache for OAuth authorization-server JWKS documents."""

    def __init__(
        self,
        url: str = "",
        *,
        ttl_seconds: float = 300,
        timeout_seconds: float = 5,
        loader: ClaudeMCPJWKSLoader | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.url = url
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._loader = loader
        self._now = now
        self._cached: dict[str, Any] | None = None
        self._expires_at = 0.0

    async def get(self) -> Mapping[str, Any]:
        """Return cached JWKS JSON, refreshing when the TTL has elapsed."""
        now = self._now()
        if self._cached is not None and now < self._expires_at:
            return self._cached
        if self._loader is not None:
            raw = self._loader()
            loaded = await raw if inspect.isawaitable(raw) else raw
        else:
            loaded = self._fetch()
        if not isinstance(loaded, Mapping):
            raise ValueError("JWKS loader must return a mapping")
        self._cached = dict(loaded)
        self._expires_at = now + max(self.ttl_seconds, 0)
        return self._cached

    def _fetch(self) -> Mapping[str, Any]:
        if not self.url:
            raise ValueError("jwks_url is required when no loader is configured")
        with urlopen(self.url, timeout=self.timeout_seconds) as response:
            loaded = json.load(response)
        if not isinstance(loaded, Mapping):
            raise ValueError("JWKS endpoint must return a JSON object")
        return loaded


class ClaudeMCPBearerTokenValidator:
    """Validate verified OAuth/JWT-style bearer claims for HTTP MCP auth hooks."""

    def __init__(
        self,
        *,
        decode: ClaudeMCPTokenDecodeFn,
        issuer: str | Sequence[str] = (),
        audience: str | Sequence[str] = (),
        resource: str = "",
        required_scopes: Sequence[str] = (),
        principal_claims: Sequence[str] = ("sub", "client_id", "azp"),
        scope_claims: Sequence[str] = ("scope", "scp"),
        resource_claims: Sequence[str] = ("resource", "aud"),
        leeway_seconds: float = 0,
        require_exp: bool = True,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._decode = decode
        self.issuers = _nonempty_string_tuple(issuer)
        self.audiences = _nonempty_string_tuple(audience)
        self.resource = resource
        self.required_scopes = _http_authorization_scopes(required_scopes)
        self.principal_claims = tuple(principal_claims)
        self.scope_claims = tuple(scope_claims)
        self.resource_claims = tuple(resource_claims)
        self.leeway_seconds = leeway_seconds
        self.require_exp = require_exp
        self._now = now

    @classmethod
    def pyjwt(
        cls,
        *,
        signing_key: Any | None = None,
        jwks_url: str = "",
        jwks_loader: ClaudeMCPJWKSLoader | None = None,
        jwks_ttl_seconds: float = 300,
        algorithms: Sequence[str] = ("RS256",),
        issuer: str | Sequence[str] = (),
        audience: str | Sequence[str] = (),
        resource: str = "",
        required_scopes: Sequence[str] = (),
        leeway_seconds: float = 0,
        allowed_token_types: Sequence[str] = ("at+jwt", "application/at+jwt"),
    ) -> ClaudeMCPBearerTokenValidator:
        """Build a validator that verifies JWT signatures with optional PyJWT."""
        try:
            jwt = import_module("jwt")
        except ImportError as e:
            raise AgentError(
                "ClaudeMCPBearerTokenValidator",
                "Install signal-gating[auth] to use PyJWT-backed token validation.",
            ) from e
        if signing_key is None and not jwks_url and jwks_loader is None:
            raise ValueError("signing_key, jwks_url, or jwks_loader is required")
        effective_audience: str | Sequence[str] = audience or resource
        jwks_cache = (
            ClaudeMCPJWKSCache(jwks_url, ttl_seconds=jwks_ttl_seconds, loader=jwks_loader)
            if signing_key is None
            else None
        )

        async def decode(token: str) -> ClaudeMCPTokenClaims:
            key = signing_key
            header = jwt.get_unverified_header(token)
            if not isinstance(header, Mapping):
                raise ValueError("JWT header must be a mapping")
            normalized_token_types = {
                str(token_type).lower()
                for token_type in allowed_token_types
                if str(token_type)
            }
            if normalized_token_types:
                token_type = str(header.get("typ", "")).lower()
                if token_type not in normalized_token_types:
                    raise ValueError("JWT access token typ header is invalid")
            if key is None:
                assert jwks_cache is not None
                jwk = _jwk_from_set(await jwks_cache.get(), str(header.get("kid", "")))
                key = jwt.PyJWK.from_dict(jwk).key
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(algorithms),
                options={"verify_aud": False, "verify_iss": False},
                leeway=leeway_seconds,
            )
            if not isinstance(claims, Mapping):
                raise ValueError("JWT claims must be a mapping")
            return claims

        return cls(
            decode=decode,
            issuer=issuer,
            audience=effective_audience,
            resource=resource,
            required_scopes=required_scopes,
            leeway_seconds=leeway_seconds,
        )

    async def __call__(
        self,
        context: ClaudeMCPHTTPAuthorizationContext,
    ) -> ClaudeMCPHTTPAuthorizationResult:
        try:
            raw_claims = self._decode(context.bearer_token)
            claims = await raw_claims if inspect.isawaitable(raw_claims) else raw_claims
            if not isinstance(claims, Mapping):
                return self._invalid_token()
            return self.validate_claims(claims)
        except Exception:
            return self._invalid_token()

    def validate_claims(
        self,
        claims: Mapping[str, Any],
    ) -> ClaudeMCPHTTPAuthorizationResult:
        """Validate already-verified token claims and return an HTTP auth decision."""
        now = self._now()
        if self.require_exp and claims.get("exp") is None:
            return self._invalid_token()
        if not _time_claim_valid(claims.get("exp"), now, self.leeway_seconds, upper_bound=True):
            return self._invalid_token()
        if not _time_claim_valid(claims.get("nbf"), now, self.leeway_seconds, upper_bound=False):
            return self._invalid_token()
        if self.issuers and str(claims.get("iss", "")) not in self.issuers:
            return self._invalid_token()
        matched_audience = (
            _matching_claim_value(claims, ("aud",), self.audiences)
            if self.audiences
            else ""
        )
        if self.audiences and not matched_audience:
            return self._invalid_token()
        if self.resource and not _resource_claim_matches(
            claims,
            self.resource_claims,
            self.resource,
        ):
            return self._invalid_token()

        scopes = _claims_scopes(claims, self.scope_claims)
        missing_scopes = tuple(scope for scope in self.required_scopes if scope not in scopes)
        if missing_scopes:
            return ClaudeMCPHTTPAuthorizationResult(
                allowed=False,
                status_code=403,
                message="insufficient scope",
                www_authenticate=_bearer_error_challenge(
                    "insufficient_scope",
                    scope=" ".join(self.required_scopes),
                ),
            )

        principal = _first_claim_value(claims, self.principal_claims)
        if not principal:
            return self._invalid_token()
        audience = matched_audience or _first_claim_value(claims, ("aud",))
        resource = self.resource or _first_claim_value(claims, self.resource_claims)
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=True,
            principal=principal,
            scopes=scopes,
            audience=audience,
            resource=resource,
        )

    def _invalid_token(self) -> ClaudeMCPHTTPAuthorizationResult:
        return ClaudeMCPHTTPAuthorizationResult(
            allowed=False,
            status_code=401,
            message="invalid token",
            www_authenticate=_bearer_error_challenge("invalid_token"),
        )


ClaudeMCPJWTBearerAuthorizer = ClaudeMCPBearerTokenValidator


@dataclass(frozen=True, slots=True)
class ClaudeMCPProtectedResourceMetadata:
    """OAuth Protected Resource Metadata for an HTTP MCP resource server."""

    resource: str
    authorization_servers: tuple[str, ...]
    scopes_supported: tuple[str, ...] = ()
    bearer_methods_supported: tuple[str, ...] = ("header",)
    resource_name: str = "Signal Gating Protocol MCP"
    resource_documentation: str = ""
    resource_policy_uri: str = ""
    resource_tos_uri: str = ""
    extra: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = urlparse(self.resource)
        if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
            raise ValueError("resource must be an https URL without a fragment")
        if not self.authorization_servers:
            raise ValueError("authorization_servers must contain at least one issuer URL")

    def to_dict(self) -> dict[str, Any]:
        """Return RFC 9728 JSON metadata with empty values omitted."""
        data: dict[str, Any] = {
            "resource": self.resource,
            "authorization_servers": list(self.authorization_servers),
            "bearer_methods_supported": list(self.bearer_methods_supported),
        }
        if self.scopes_supported:
            data["scopes_supported"] = list(self.scopes_supported)
        if self.resource_name:
            data["resource_name"] = self.resource_name
        if self.resource_documentation:
            data["resource_documentation"] = self.resource_documentation
        if self.resource_policy_uri:
            data["resource_policy_uri"] = self.resource_policy_uri
        if self.resource_tos_uri:
            data["resource_tos_uri"] = self.resource_tos_uri
        projected = _safe_metadata_projection(self.extra, depth=3)
        if isinstance(projected, Mapping):
            data.update(projected)
        return data

    def metadata_url(self) -> str:
        """Return the default RFC 9728 well-known metadata URL for ``resource``."""
        return protected_resource_metadata_url(self.resource)


ClaudeMCPHTTPAuthorizationDecision = bool | str | ClaudeMCPHTTPAuthorizationResult
ClaudeMCPHTTPAuthorizeFn = Callable[
    [ClaudeMCPHTTPAuthorizationContext],
    ClaudeMCPHTTPAuthorizationDecision | Awaitable[ClaudeMCPHTTPAuthorizationDecision],
]


class ClaudeToolPolicy:
    """Compile SGP gates into Claude Agent SDK tool permission controls."""

    def __init__(
        self,
        *,
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        gate: Gate | None = None,
        default: ClaudePermissionDecision = "denied",
        deny_message: str = "Rejected by Signal Gating Protocol tool policy.",
        interrupt_on_deny: bool = False,
    ) -> None:
        if default not in {"allowed", "denied", "prompted", "unknown"}:
            raise ValueError("default must be a ClaudePermissionDecision")
        self.allowed_tools = _dedupe(allowed_tools or [])
        self.disallowed_tools = _dedupe(disallowed_tools or [])
        self.gate = gate
        self.default = default
        self.deny_message = deny_message
        self.interrupt_on_deny = interrupt_on_deny

    @classmethod
    def read_only(cls, *, gate: Gate | None = None) -> ClaudeToolPolicy:
        """Auto-approve Claude's read-only filesystem/code-search tools."""
        return cls(
            allowed_tools=["Read", "Glob", "Grep"],
            disallowed_tools=["Write", "Edit", "MultiEdit", "NotebookEdit"],
            gate=gate,
        )

    @classmethod
    def mcp_tools(
        cls,
        server: str,
        tools: Sequence[str],
        *,
        gate: Gate | None = None,
    ) -> ClaudeToolPolicy:
        """Build a policy for a named MCP server's tools."""
        return cls(
            allowed_tools=[mcp_tool_name(server, tool) for tool in tools],
            gate=gate,
        )

    def claude_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``ClaudeAgentOptions`` construction."""
        kwargs: dict[str, Any] = {
            "can_use_tool": self.can_use_tool,
        }
        if self.allowed_tools:
            kwargs["allowed_tools"] = list(self.allowed_tools)
        if self.disallowed_tools:
            kwargs["disallowed_tools"] = list(self.disallowed_tools)
        return kwargs

    async def can_use_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: Any,
    ) -> Any:
        """Claude Agent SDK ``can_use_tool`` callback backed by an SGP gate."""
        request = _tool_request_signal(tool_name, input_data, context)
        if _tool_rule_matches_any(tool_name, self.disallowed_tools):
            return _permission_result_deny(
                self.deny_message,
                interrupt=self.interrupt_on_deny,
            )
        if self.gate is not None:
            if await self.gate.process(request) is None:
                return _permission_result_deny(
                    self.deny_message,
                    interrupt=self.interrupt_on_deny,
                )
            return _permission_result_allow()
        if _tool_rule_matches_any(tool_name, self.allowed_tools):
            return _permission_result_allow()
        if self.default == "denied":
            return _permission_result_deny(
                self.deny_message,
                interrupt=self.interrupt_on_deny,
            )
        return _permission_result_allow()


class ClaudeMeshMCPAdapter:
    """Expose SGP mesh tools through MCP-shaped JSON-RPC handlers.

    This is a transport-agnostic adapter. It implements the protocol payloads
    for ``initialize``, ``tools/list``, and ``tools/call`` over an existing
    :class:`~signal_gating.mesh.Mesh`; a stdio or HTTP transport can wrap
    ``handle_request`` without changing the mesh/tool semantics.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        server_name: str = "mesh",
        server_version: str = "0.1.0",
    ) -> None:
        self._mesh = mesh
        self.server_name = server_name
        self.server_version = server_version
        self._initialize_seen = False
        self._initialized = False

    def tool_names(self) -> list[str]:
        """Return unique mesh tool names in deterministic order."""
        return sorted(self._index())

    def claude_allowed_tools(self) -> list[str]:
        """Return Claude MCP tool names for all exposed mesh tools."""
        return [mcp_tool_name(self.server_name, name) for name in self.tool_names()]

    def tool_policy(self, *, gate: Gate | None = None) -> ClaudeToolPolicy:
        """Build a Claude tool policy that allows this adapter's MCP tools."""
        return ClaudeToolPolicy(
            allowed_tools=self.claude_allowed_tools(),
            gate=gate,
        )

    def initialize_result(self) -> dict[str, Any]:
        """Return an MCP initialize result declaring tool support."""
        return {
            "protocolVersion": "2025-06-18",
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
            "capabilities": {"tools": {"listChanged": False}},
        }

    def tools_list_result(self) -> dict[str, Any]:
        """Return the MCP ``tools/list`` result for the mesh tools."""
        index = self._index()
        return {
            "tools": [
                _mcp_tool_schema(name, owner, spec)
                for name, (owner, spec) in sorted(index.items())
            ]
        }

    async def tools_call_result(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call one mesh tool and return the MCP ``tools/call`` result."""
        index = self._index()
        if name not in index:
            raise ValueError(f"Unknown tool: {name}")
        owner, spec = index[name]
        tool_arguments = dict(arguments or {})
        argument_error = _tool_argument_error(spec, tool_arguments)
        if argument_error:
            raise ValueError(argument_error)
        if not getattr(self._mesh, "_running", False):
            raise RuntimeError("mesh is not running; use 'async with mesh:' before calling tools")
        try:
            result = await self._mesh.call_tool(owner, name, **tool_arguments)
        except Exception as e:
            return _mcp_error_result(f"{type(e).__name__}: {e}")
        return _mcp_success_result(result)

    async def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC request or notification."""
        invalid = self._jsonrpc_validation_error(request)
        if invalid is not None:
            return invalid

        method = request.get("method")
        request_id = request.get("id")
        if "id" not in request:
            if method == "notifications/initialized":
                if self._initialize_seen:
                    self._initialized = True
                return None
            return None
        try:
            if method == "initialize":
                result = self.initialize_result()
                self._initialize_seen = True
            elif method == "tools/list":
                not_initialized = self._not_initialized_error(request_id)
                if not_initialized is not None:
                    return not_initialized
                result = self.tools_list_result()
            elif method == "tools/call":
                not_initialized = self._not_initialized_error(request_id)
                if not_initialized is not None:
                    return not_initialized
                params = request.get("params")
                if not isinstance(params, Mapping):
                    return _jsonrpc_error_response(request_id, -32602, "Invalid params")
                name = params.get("name")
                if not isinstance(name, str):
                    return _jsonrpc_error_response(request_id, -32602, "Missing tool name")
                raw_arguments = params.get("arguments")
                if raw_arguments is not None and not isinstance(raw_arguments, Mapping):
                    return _jsonrpc_error_response(request_id, -32602, "Invalid arguments")
                result = await self.tools_call_result(name, raw_arguments)
            else:
                return _jsonrpc_error_response(
                    request_id,
                    -32601,
                    f"Method not found: {method}",
                )
        except ValueError as e:
            return _jsonrpc_error_response(request_id, -32602, str(e))
        except RuntimeError as e:
            return _jsonrpc_error_response(request_id, -32000, str(e))

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _jsonrpc_validation_error(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return _jsonrpc_error_response(request_id, -32600, "Invalid JSON-RPC version")
        if "id" in request and request_id is None:
            return _jsonrpc_error_response(None, -32600, "Invalid request id")
        if "method" not in request or not isinstance(request.get("method"), str):
            return _jsonrpc_error_response(request_id, -32600, "Invalid method")
        return None

    def _not_initialized_error(self, request_id: Any) -> dict[str, Any] | None:
        if self._initialized:
            return None
        return _jsonrpc_error_response(
            request_id,
            -32002,
            "MCP server is not initialized",
        )

    def _index(self) -> dict[str, tuple[str, Any]]:
        index: dict[str, tuple[str, Any]] = {}
        for owner, specs in self._mesh.discover_tools().items():
            for spec in specs:
                if spec.name in index:
                    raise ValueError(
                        f"ClaudeMeshMCPAdapter: duplicate tool name {spec.name!r} "
                        f"(agents {index[spec.name][0]!r} and {owner!r}); "
                        "MCP tool names must be unique."
                    )
                index[spec.name] = (owner, spec)
        return index


class ClaudeMeshMCPStdioServer:
    """Serve a :class:`ClaudeMeshMCPAdapter` over newline-delimited stdio JSON-RPC.

    The transport writes only JSON-RPC messages to ``output_stream``. Diagnostics
    should go to stderr in a process wrapper, not through this class.
    """

    def __init__(
        self,
        adapter: ClaudeMeshMCPAdapter,
        *,
        diagnostic_stream: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self._diagnostic_stream = diagnostic_stream

    async def serve(
        self,
        input_stream: Iterable[str],
        output_stream: Any,
    ) -> int:
        """Read JSON-RPC lines and write JSON-RPC responses.

        Returns the number of responses written. Notifications and blank lines
        do not produce output.
        """
        written = 0
        for raw_line in input_stream:
            response = await self.handle_line(raw_line)
            if response is None:
                continue
            output_stream.write(_jsonrpc_line(response))
            flush = getattr(output_stream, "flush", None)
            if flush is not None:
                flush()
            written += 1
        return written

    async def handle_line(self, raw_line: str) -> dict[str, Any] | None:
        """Handle one newline-delimited JSON-RPC line."""
        line = raw_line.strip()
        if not line:
            return None
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            return _jsonrpc_error_response(None, -32700, f"Parse error: {e.msg}")
        if not isinstance(request, Mapping):
            return _jsonrpc_error_response(None, -32600, "Invalid request")
        diagnostic_stream = self._diagnostic_stream or sys.stderr
        with redirect_stdout(diagnostic_stream):
            return await self.adapter.handle_request(request)


class ClaudeMeshMCPHTTPApp:
    """Minimal ASGI Streamable HTTP transport for :class:`ClaudeMeshMCPAdapter`.

    This transport implements the non-SSE JSON response path from MCP
    Streamable HTTP with per-session adapter state. ``GET`` returns ``405``
    because this transport does not expose a server-to-client SSE stream.
    """

    protocol_version = "2025-06-18"

    def __init__(
        self,
        adapter: ClaudeMeshMCPAdapter,
        *,
        path: str = "/mcp",
        allowed_origins: Sequence[str] | None = None,
        allow_localhost_origins: bool = True,
        max_body_bytes: int = 1_048_576,
        diagnostic_stream: Any | None = None,
        authorize_http: ClaudeMCPHTTPAuthorizeFn | None = None,
        auth_realm: str = "mcp",
        protected_resource_metadata_url: str | None = None,
        protected_resource_metadata: (
            ClaudeMCPProtectedResourceMetadata | Mapping[str, Any] | None
        ) = None,
        protected_resource_metadata_path: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.path = path
        self.allowed_origins = set(allowed_origins or [])
        self.allow_localhost_origins = allow_localhost_origins
        self.max_body_bytes = max_body_bytes
        self._diagnostic_stream = diagnostic_stream
        self._authorize_http = authorize_http
        self.auth_realm = auth_realm
        self._protected_resource_metadata = _coerce_protected_resource_metadata(
            protected_resource_metadata
        )
        derived_metadata_url = (
            self._protected_resource_metadata.metadata_url()
            if self._protected_resource_metadata is not None
            else None
        )
        self.protected_resource_metadata_url = (
            protected_resource_metadata_url or derived_metadata_url
        )
        metadata_url = self.protected_resource_metadata_url or derived_metadata_url
        self.protected_resource_metadata_path = (
            protected_resource_metadata_path
            or (
                _url_path(metadata_url)
                if metadata_url
                else "/.well-known/oauth-protected-resource"
            )
        )
        if (
            self._protected_resource_metadata is not None
            and self.protected_resource_metadata_path == self.path
        ):
            raise ValueError("protected_resource_metadata_path must not equal the MCP path")
        if (
            metadata_url
            and protected_resource_metadata_path
            and _url_path(metadata_url) != protected_resource_metadata_path
        ):
            raise ValueError("protected_resource_metadata_url path must be served by the app")
        self._sessions: dict[str, ClaudeMeshMCPAdapter] = {}
        self._terminated_sessions: set[str] = set()
        self._session_authorizations: dict[str, str] = {}

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        """Serve one ASGI HTTP request."""
        if scope.get("type") != "http":
            raise TypeError("ClaudeMeshMCPHTTPApp only supports ASGI HTTP scopes")
        method = str(scope.get("method", "")).upper()
        path = str(scope.get("path", ""))
        headers = _asgi_headers(scope)
        raw_query_string = scope.get("query_string", b"")
        if isinstance(raw_query_string, bytes):
            query_string = raw_query_string.decode("latin-1")
        else:
            query_string = str(raw_query_string)

        if self._is_protected_resource_metadata_path(path):
            await self._serve_protected_resource_metadata(method, send)
            return
        if path != self.path:
            await _send_http_response(send, 404, b"Not Found", "text/plain; charset=utf-8")
            return
        if not self._origin_allowed(headers.get("origin")):
            await _send_jsonrpc_http_error(send, 403, -32000, "Forbidden origin")
            return
        if not _mcp_http_protocol_version_supported(headers):
            await _send_jsonrpc_http_error(send, 400, -32000, "Unsupported MCP protocol version")
            return
        if self._authorize_http is not None and _query_contains_access_token(query_string):
            await self._record_authorization(
                method=method,
                path=path,
                headers=headers,
                allowed=False,
                status_code=400,
                reason="query_token_rejected",
            )
            await _send_jsonrpc_http_error(
                send,
                400,
                -32000,
                "Access tokens must not be sent in the URI query string",
            )
            return
        auth_identity = await self._authorize_request(method, path, headers, send)
        if auth_identity is None:
            return
        if method == "GET":
            checked_session_id = await self._require_session(headers, send)
            if checked_session_id is None:
                return
            if not await self._session_authorized(
                checked_session_id,
                auth_identity,
                headers,
                send,
                method=method,
                path=path,
            ):
                return
            await _send_http_response(
                send,
                405,
                b"",
                headers=[(b"allow", b"POST")],
            )
            return
        if method == "DELETE":
            deleted_session_id = await self._require_session(headers, send)
            if deleted_session_id is None:
                return
            if not await self._session_authorized(
                deleted_session_id,
                auth_identity,
                headers,
                send,
                method=method,
                path=path,
            ):
                return
            self._sessions.pop(deleted_session_id, None)
            self._terminated_sessions.add(deleted_session_id)
            self._session_authorizations.pop(deleted_session_id, None)
            await _send_http_response(
                send,
                204,
                b"",
            )
            return
        if method != "POST":
            await _send_http_response(
                send,
                405,
                b"",
                headers=[(b"allow", b"POST")],
            )
            return
        if not _http_accepts(headers, "application/json") or not _http_accepts(
            headers,
            "text/event-stream",
        ):
            await _send_jsonrpc_http_error(send, 406, -32000, "Unsupported Accept header")
            return
        if _http_content_type(headers) != "application/json":
            await _send_jsonrpc_http_error(send, 415, -32000, "Unsupported Content-Type")
            return

        try:
            body = await _read_asgi_body(receive, self.max_body_bytes)
            request = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError:
            await _send_jsonrpc_http_error(send, 400, -32700, "Parse error: invalid UTF-8")
            return
        except json.JSONDecodeError as e:
            await _send_jsonrpc_http_error(send, 400, -32700, f"Parse error: {e.msg}")
            return
        except ValueError as e:
            await _send_jsonrpc_http_error(send, 413, -32000, str(e))
            return
        if not isinstance(request, Mapping):
            await _send_jsonrpc_http_error(send, 400, -32600, "Invalid request")
            return

        is_initialize = request.get("method") == "initialize" and "id" in request
        session_id: str | None = None
        adapter = self.adapter
        extra_headers: list[tuple[bytes, bytes]] = []
        if is_initialize:
            session_id = self._new_session_id()
            adapter = self._new_session_adapter()
            self._sessions[session_id] = adapter
            if auth_identity:
                self._session_authorizations[session_id] = auth_identity
            extra_headers.append((b"mcp-session-id", session_id.encode("ascii")))
        else:
            session_id = await self._require_session(headers, send)
            if session_id is None:
                return
            if not await self._session_authorized(
                session_id,
                auth_identity,
                headers,
                send,
                method=method,
                path=path,
                jsonrpc_method=str(request.get("method") or ""),
            ):
                return
            adapter = self._sessions[session_id]

        if _is_jsonrpc_response_message(request):
            await _send_http_response(send, 202, b"")
            return

        diagnostic_stream = self._diagnostic_stream or sys.stderr
        with redirect_stdout(diagnostic_stream):
            response = await adapter.handle_request(request)
        if response is None:
            await _send_http_response(send, 202, b"")
            return
        if is_initialize and "error" in response and session_id is not None:
            self._sessions.pop(session_id, None)
            self._session_authorizations.pop(session_id, None)
            extra_headers = []
        await _send_jsonrpc_http_response(send, response, headers=extra_headers)

    def _origin_allowed(self, origin: str | None) -> bool:
        if origin is None or origin == "":
            return True
        if origin in self.allowed_origins:
            return True
        if not self.allow_localhost_origins:
            return False
        return _origin_host(origin) in {"localhost", "127.0.0.1", "::1"}

    async def _require_session(
        self,
        headers: Mapping[str, str],
        send: ASGISend,
    ) -> str | None:
        session_id = headers.get("mcp-session-id")
        if not session_id:
            await _send_jsonrpc_http_error(send, 400, -32000, "Missing Mcp-Session-Id")
            return None
        if session_id in self._terminated_sessions or session_id not in self._sessions:
            await _send_jsonrpc_http_error(send, 404, -32000, "Unknown MCP session")
            return None
        return session_id

    def _new_session_id(self) -> str:
        while True:
            session_id = secrets.token_urlsafe(32)
            if session_id not in self._sessions and session_id not in self._terminated_sessions:
                return session_id

    def _new_session_adapter(self) -> ClaudeMeshMCPAdapter:
        return ClaudeMeshMCPAdapter(
            self.adapter._mesh,
            server_name=self.adapter.server_name,
            server_version=self.adapter.server_version,
        )

    def _is_protected_resource_metadata_path(self, path: str) -> bool:
        return (
            self._protected_resource_metadata is not None
            and path == self.protected_resource_metadata_path
        )

    async def _serve_protected_resource_metadata(
        self,
        method: str,
        send: ASGISend,
    ) -> None:
        if self._protected_resource_metadata is None:
            await _send_http_response(send, 404, b"Not Found", "text/plain; charset=utf-8")
            return
        if method != "GET":
            await _send_http_response(
                send,
                405,
                b"",
                headers=[(b"allow", b"GET")],
            )
            return
        await _send_http_response(
            send,
            200,
            _jsonrpc_http_body(self._protected_resource_metadata.to_dict()),
            "application/json; charset=utf-8",
        )

    async def _authorize_request(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        send: ASGISend,
    ) -> str | None:
        if self._authorize_http is None:
            return ""
        scheme, token = _bearer_authorization(headers)
        if not token:
            await self._record_authorization(
                method=method,
                path=path,
                headers=headers,
                allowed=False,
                status_code=401,
                reason="authorization_required",
            )
            await self._send_authorization_error(
                send,
                401,
                "Authorization required",
                "",
            )
            return None
        context = ClaudeMCPHTTPAuthorizationContext(
            method=method,
            path=path,
            origin=headers.get("origin", ""),
            mcp_session_id=headers.get("mcp-session-id", ""),
            authorization_scheme=scheme,
            bearer_token=token,
            header_names=tuple(sorted(name for name in headers if name != "authorization")),
        )
        raw_decision = await _resolve_http_authorization_decision(self._authorize_http, context)
        decision = _coerce_http_authorization_result(raw_decision)
        if not decision.allowed:
            status = 403 if decision.status_code == 403 else 401
            default_message = "Forbidden" if status == 403 else "Unauthorized"
            await self._record_authorization(
                method=method,
                path=path,
                headers=headers,
                allowed=False,
                status_code=status,
                reason=_authorization_reason(decision, status),
                decision=decision,
            )
            await self._send_authorization_error(
                send,
                status,
                decision.message or default_message,
                decision.www_authenticate,
            )
            return None
        await self._record_authorization(
            method=method,
            path=path,
            headers=headers,
            allowed=True,
            status_code=200,
            reason="allowed",
            decision=decision,
        )
        return _http_authorization_identity(decision, token)

    async def _session_authorized(
        self,
        session_id: str,
        auth_identity: str,
        headers: Mapping[str, str],
        send: ASGISend,
        *,
        method: str,
        path: str,
        jsonrpc_method: str = "",
    ) -> bool:
        expected = self._session_authorizations.get(session_id, "")
        if not expected or not auth_identity or expected == auth_identity:
            return True
        await self._record_authorization(
            method=method,
            path=path,
            headers=headers,
            allowed=False,
            status_code=403,
            reason="session_authorization_mismatch",
            auth_identity=auth_identity,
            jsonrpc_method=jsonrpc_method,
        )
        await self._send_authorization_error(send, 403, "Session authorization mismatch", "")
        return False

    async def _record_authorization(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        allowed: bool,
        status_code: int,
        reason: str,
        decision: ClaudeMCPHTTPAuthorizationResult | None = None,
        auth_identity: str = "",
        jsonrpc_method: str = "",
    ) -> None:
        session_id = headers.get("mcp-session-id", "")
        components = _authorization_components(decision, auth_identity)
        scheme, bearer_token = _bearer_authorization(headers)
        outcome = "allowed" if allowed else "denied"
        action = _authorization_action(allowed, reason)
        signal = ClaudeMCPHTTPAuthorizationSignal(
            outcome=outcome,
            status_code=status_code,
            method=method,
            path=path,
            reason=reason,
            jsonrpc_method=jsonrpc_method,
            authorization_scheme=scheme,
            bearer_token_present=bool(bearer_token),
            principal_hash=components["principal_hash"],
            principal_present=components["principal_present"],
            audience_present=components["audience_present"],
            resource_present=components["resource_present"],
            scope_count=components["scope_count"],
            identity_binding_kind=components["identity_binding_kind"],
            mcp_session_id_hash=_token_fingerprint(session_id) if session_id else "",
            mcp_session_present=bool(session_id),
            protected_resource_metadata_advertised=bool(self.protected_resource_metadata_url),
        )
        await _record_claude_http_event(
            self.adapter._mesh,
            action,
            signal,
            outcome=outcome,
            status_code=status_code,
            reason=reason,
            jsonrpc_method=jsonrpc_method,
            authorization_scheme=scheme,
            bearer_token_present=bool(bearer_token),
            principal_present=components["principal_present"],
            principal_hash=components["principal_hash"],
            audience_present=components["audience_present"],
            resource_present=components["resource_present"],
            scope_count=components["scope_count"],
            identity_binding_kind=components["identity_binding_kind"],
            mcp_session_present=bool(session_id),
            mcp_session_id_hash=_token_fingerprint(session_id) if session_id else "",
            protected_resource_metadata_advertised=bool(self.protected_resource_metadata_url),
        )

    async def _send_authorization_error(
        self,
        send: ASGISend,
        status: int,
        message: str,
        www_authenticate: str,
    ) -> None:
        headers: list[tuple[bytes, bytes]] = []
        if www_authenticate and status in {401, 403}:
            challenge = _challenge_with_resource_metadata(
                www_authenticate,
                self.protected_resource_metadata_url,
            )
            headers.append((b"www-authenticate", challenge.encode("ascii")))
        elif status == 401:
            challenge = www_authenticate or _bearer_challenge(
                self.auth_realm,
                self.protected_resource_metadata_url,
            )
            headers.append((b"www-authenticate", challenge.encode("ascii")))
        await _send_jsonrpc_http_error(send, status, -32000, message, headers=headers)


class ClaudeAgent(Agent):
    """An SGP agent that delegates reasoning/action to Claude Agent SDK.

    The adapter emits typed result and tool-event signals. It records only SGP
    boundary evidence; Claude's session transcript remains owned by Claude Agent
    SDK and should be resumed with the returned ``session_id``.
    """

    def __init__(
        self,
        name: str,
        *,
        query_fn: ClaudeQueryFn | None = None,
        options_factory: ClaudeOptionsFactory | None = None,
        model: str = "",
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        permission_mode: str = "",
        can_use_tool: Any | None = None,
        tool_policy: ClaudeToolPolicy | None = None,
        mcp_servers: Mapping[str, Any] | None = None,
        cwd: str = "",
        resume: str = "",
        continue_conversation: bool = False,
        max_turns: int | None = None,
        on: type[Signal] = ClaudeAgentRunSignal,
        render: Render | None = None,
        emit_tool_events: bool = True,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(name, **agent_kwargs)
        self._query_fn = query_fn
        self._options_factory = options_factory
        self._model = model
        self._system_prompt = system_prompt
        policy_kwargs = tool_policy.claude_kwargs() if tool_policy is not None else {}
        policy_allowed = policy_kwargs.pop("allowed_tools", [])
        policy_disallowed = policy_kwargs.pop("disallowed_tools", [])
        if can_use_tool is None:
            can_use_tool = policy_kwargs.pop("can_use_tool", None)
        self._allowed_tools = _dedupe([*(allowed_tools or []), *policy_allowed])
        self._disallowed_tools = _dedupe([*(disallowed_tools or []), *policy_disallowed])
        self._permission_mode = permission_mode
        self._can_use_tool = can_use_tool
        self._mcp_servers = dict(mcp_servers or {})
        self._cwd = cwd
        self._resume = resume
        self._continue_conversation = continue_conversation
        self._max_turns = max_turns
        self._render: Render = render or _default_render
        self._emit_tool_events = emit_tool_events
        self.on(on)(self._handle)

    async def _handle(self, signal: Signal) -> None:
        prompt = self._render(signal)
        options_kwargs = self._options_kwargs(signal)
        query_fn, options_factory = self._sdk_bindings()
        options = options_factory(**options_kwargs) if options_kwargs else options_factory()

        session_id = ""
        result_text = ""
        subtype = ""
        total_cost_usd: float | None = None
        message_count = 0

        async for message in query_fn(prompt=prompt, options=options):
            message_count += 1
            session_id = _session_id_from_message(message) or session_id
            if self._emit_tool_events:
                for event in _tool_events_from_message(message, session_id):
                    await self.emit(
                        event.evolve(
                            trace_id=signal.trace_id,
                            parent_id=signal.id,
                            priority=signal.priority,
                        )
                    )
            result = _result_from_message(message)
            if result is not None:
                result_text = result["text"]
                subtype = result["subtype"]
                total_cost_usd = result["total_cost_usd"]

        if not result_text.strip():
            raise AgentError(self.name, "Claude Agent SDK returned no result text")

        await self.emit(
            ClaudeAgentResultSignal(
                text=result_text,
                session_id=session_id,
                subtype=subtype,
                total_cost_usd=total_cost_usd,
                message_count=message_count,
                allowed_tools=list(options_kwargs.get("allowed_tools", [])),
                disallowed_tools=list(options_kwargs.get("disallowed_tools", [])),
                permission_mode=str(options_kwargs.get("permission_mode", "")),
                mcp_servers=sorted(dict(options_kwargs.get("mcp_servers", {}))),
                resumed_from_session_id=str(options_kwargs.get("resume", "")),
                continued=bool(options_kwargs.get("continue_conversation", False)),
                trace_id=signal.trace_id,
                parent_id=signal.id,
                priority=signal.priority,
            )
        )

    def _sdk_bindings(self) -> tuple[ClaudeQueryFn, ClaudeOptionsFactory]:
        query_fn = self._query_fn
        options_factory = self._options_factory
        if query_fn is not None and options_factory is not None:
            return query_fn, options_factory

        try:
            sdk = import_module("claude_agent_sdk")
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "ClaudeAgent requires claude-agent-sdk. "
                "Install it with: pip install 'signal-gating[claude]'"
            ) from e

        query_fn = query_fn or getattr(sdk, "query")
        options_factory = options_factory or getattr(sdk, "ClaudeAgentOptions")
        self._query_fn = query_fn
        self._options_factory = options_factory
        return query_fn, options_factory

    def _options_kwargs(self, signal: Signal) -> dict[str, Any]:
        allowed_tools = list(self._allowed_tools)
        disallowed_tools = list(self._disallowed_tools)
        permission_mode = self._permission_mode
        mcp_servers = dict(self._mcp_servers)
        cwd = self._cwd
        resume = self._resume
        continue_conversation = self._continue_conversation

        if isinstance(signal, ClaudeAgentRunSignal):
            if signal.allowed_tools:
                allowed_tools = list(signal.allowed_tools)
            if signal.disallowed_tools:
                disallowed_tools = list(signal.disallowed_tools)
            if signal.permission_mode:
                permission_mode = signal.permission_mode
            if signal.mcp_servers:
                mcp_servers = dict(signal.mcp_servers)
            if signal.cwd:
                cwd = signal.cwd
            if signal.session_id:
                resume = signal.session_id
            if signal.continue_conversation is not None:
                continue_conversation = signal.continue_conversation

        kwargs: dict[str, Any] = {}
        if self._model:
            kwargs["model"] = self._model
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        if allowed_tools:
            kwargs["allowed_tools"] = allowed_tools
        if disallowed_tools:
            kwargs["disallowed_tools"] = disallowed_tools
        if permission_mode:
            kwargs["permission_mode"] = permission_mode
        if self._can_use_tool is not None:
            kwargs["can_use_tool"] = self._can_use_tool
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
        if cwd:
            kwargs["cwd"] = cwd
        if self._max_turns is not None:
            kwargs["max_turns"] = self._max_turns
        if resume:
            kwargs["resume"] = resume
        elif continue_conversation:
            kwargs["continue_conversation"] = True
        return kwargs


def _default_render(signal: Signal) -> str:
    if isinstance(signal, ClaudeAgentRunSignal):
        return signal.prompt
    text = getattr(signal, "text", "")
    return str(text) if text else repr(signal)


def mcp_tool_name(server: str, tool: str) -> str:
    """Return Claude Agent SDK's MCP tool name for a server/tool pair."""
    return f"mcp__{server}__{tool}"


def claude_options(
    *,
    allowed_tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    can_use_tool: Any | None = None,
    tool_policy: ClaudeToolPolicy | None = None,
    permission_mode: str | None = None,
    continue_conversation: bool | None = None,
    resume: str | None = None,
    mcp_servers: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
    session_store: Any | None = None,
    strict_mcp_config: bool | None = None,
    max_turns: int | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
) -> Any:
    """Build ``claude_agent_sdk.ClaudeAgentOptions`` with a lazy import."""
    _query_fn, options_factory = _load_sdk_bindings()
    policy_kwargs = tool_policy.claude_kwargs() if tool_policy is not None else {}
    policy_allowed = policy_kwargs.pop("allowed_tools", [])
    policy_disallowed = policy_kwargs.pop("disallowed_tools", [])
    if can_use_tool is None:
        can_use_tool = policy_kwargs.pop("can_use_tool", None)
    kwargs: dict[str, Any] = {}
    compiled_allowed = _dedupe([*(allowed_tools or []), *policy_allowed])
    compiled_disallowed = _dedupe([*(disallowed_tools or []), *policy_disallowed])
    if compiled_allowed:
        kwargs["allowed_tools"] = compiled_allowed
    if compiled_disallowed:
        kwargs["disallowed_tools"] = compiled_disallowed
    if can_use_tool is not None:
        kwargs["can_use_tool"] = can_use_tool
    if permission_mode is not None:
        kwargs["permission_mode"] = permission_mode
    if continue_conversation is not None:
        kwargs["continue_conversation"] = continue_conversation
    if resume is not None:
        kwargs["resume"] = resume
    if mcp_servers is not None:
        kwargs["mcp_servers"] = dict(mcp_servers)
    if cwd is not None:
        kwargs["cwd"] = cwd
    if session_store is not None:
        kwargs["session_store"] = session_store
    if strict_mcp_config is not None:
        kwargs["strict_mcp_config"] = strict_mcp_config
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    if model is not None:
        kwargs["model"] = model
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return options_factory(**kwargs)


class ClaudeAgentSDKRunner:
    """Run Claude Agent SDK directly and record audit-only mesh events."""

    def __init__(
        self,
        *,
        query: ClaudeQueryFn | None = None,
        options_factory: ClaudeOptionsFactory | None = None,
    ) -> None:
        self._query_fn = query
        self._options_factory = options_factory

    async def run(
        self,
        prompt: str,
        *,
        mesh: Any | None = None,
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        can_use_tool: Any | None = None,
        tool_policy: ClaudeToolPolicy | None = None,
        permission_mode: str = "",
        continue_conversation: bool = False,
        resume: str = "",
        mcp_servers: Mapping[str, Any] | None = None,
        cwd: str | Path | None = None,
        session_store: Any | None = None,
        strict_mcp_config: bool | None = None,
        max_turns: int | None = None,
        model: str = "",
        system_prompt: str = "",
    ) -> ClaudeAgentSDKResult:
        query_fn, options_factory = self._sdk_bindings()
        policy_kwargs = tool_policy.claude_kwargs() if tool_policy is not None else {}
        policy_allowed = policy_kwargs.pop("allowed_tools", [])
        policy_disallowed = policy_kwargs.pop("disallowed_tools", [])
        if can_use_tool is None:
            can_use_tool = policy_kwargs.pop("can_use_tool", None)
        allowed = _dedupe([*(allowed_tools or []), *policy_allowed])
        disallowed = _dedupe([*(disallowed_tools or []), *policy_disallowed])
        mcp_config = dict(mcp_servers or {})
        options_kwargs = _runner_options_kwargs(
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            can_use_tool=can_use_tool,
            permission_mode=permission_mode,
            continue_conversation=continue_conversation,
            resume=resume,
            mcp_servers=mcp_config,
            cwd=cwd,
            session_store=session_store,
            strict_mcp_config=strict_mcp_config,
            max_turns=max_turns,
            model=model,
            system_prompt=system_prompt,
        )
        options = options_factory(**options_kwargs)
        run_signal = ClaudeAgentRunSignal(
            prompt=prompt,
            session_id=resume,
            continue_conversation=continue_conversation,
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            permission_mode=permission_mode,
            mcp_servers={name: {} for name in mcp_config},
            cwd=str(cwd or ""),
        )
        await _record_claude_event(
            mesh,
            "claude_query_start",
            run_signal,
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            permission_mode=permission_mode,
            mcp_server_names=sorted(mcp_config),
            cwd=str(cwd or ""),
            resumed_from_session_id=resume,
            continued=continue_conversation,
        )

        session_id = ""
        result_text = ""
        subtype = ""
        total_cost_usd: float | None = None
        message_count = 0

        async for message in query_fn(prompt=prompt, options=options):
            message_count += 1
            session_id = _session_id_from_message(message) or session_id
            mcp_init = _mcp_init_metadata(message)
            if mcp_init:
                await _record_claude_event(
                    mesh,
                    "claude_mcp_init",
                    run_signal,
                    claude_session_id=session_id,
                    **mcp_init,
                )
            for event in _tool_events_from_message(message, session_id):
                await _record_claude_event(
                    mesh,
                    f"claude_{event.event}",
                    event,
                    claude_session_id=session_id,
                    tool_name=event.tool_name,
                    tool_use_id=event.tool_call_id,
                    tool_input_keys=event.tool_input_keys,
                    tool_allowed=_tool_is_allowed(event.tool_name, allowed, disallowed),
                    mcp_server=event.mcp_server,
                )
            result = _result_from_message(message)
            if result is not None:
                result_text = result["text"]
                subtype = result["subtype"]
                total_cost_usd = result["total_cost_usd"]
                final_signal = ClaudeAgentResultSignal(
                    text=result_text,
                    session_id=session_id,
                    subtype=subtype,
                    total_cost_usd=total_cost_usd,
                    message_count=message_count,
                    allowed_tools=allowed,
                    disallowed_tools=disallowed,
                    permission_mode=permission_mode,
                    mcp_servers=sorted(mcp_config),
                    resumed_from_session_id=resume,
                    continued=continue_conversation,
                )
                denials = _permission_denials_from_message(message)
                await _record_claude_event(
                    mesh,
                    "claude_result",
                    final_signal,
                    claude_session_id=session_id,
                    permission_denial_count=len(denials),
                    permission_denied_tools=[
                        item["tool_name"] for item in denials if item.get("tool_name")
                    ],
                )

        return ClaudeAgentSDKResult(
            session_id=session_id,
            result=result_text,
            subtype=subtype,
            total_cost_usd=total_cost_usd,
            message_count=message_count,
        )

    def _sdk_bindings(self) -> tuple[ClaudeQueryFn, ClaudeOptionsFactory]:
        query_fn = self._query_fn
        options_factory = self._options_factory
        if query_fn is not None and options_factory is not None:
            return query_fn, options_factory
        loaded_query, loaded_options = _load_sdk_bindings()
        query_fn = query_fn or loaded_query
        options_factory = options_factory or loaded_options
        self._query_fn = query_fn
        self._options_factory = options_factory
        return query_fn, options_factory


class ClaudeAgentSDKClientSession:
    """Stateful wrapper around ``ClaudeSDKClient`` for multi-turn sessions.

    ``ClaudeAgentSDKRunner`` maps to the SDK's one-shot ``query()`` helper.
    This class maps to ``ClaudeSDKClient``: one client, multiple prompts, shared
    Claude conversation context, and SGP receipts around each query boundary.
    """

    def __init__(
        self,
        *,
        client_factory: ClaudeClientFactory | None = None,
        options_factory: ClaudeOptionsFactory | None = None,
        options: Any | None = None,
        mesh: Any | None = None,
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        permission_mode: str = "",
        mcp_servers: Mapping[str, Any] | str | Path | None = None,
        cwd: str | Path | None = None,
        resume: str = "",
        continue_conversation: bool = False,
        session_store: Any | None = None,
        session_store_flush: str = "batched",
        strict_mcp_config: bool | None = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        model: str = "",
        fallback_model: str = "",
        system_prompt: str = "",
        can_use_tool: Any | None = None,
        tool_policy: ClaudeToolPolicy | None = None,
        permission_prompt_tool_name: str | None = None,
        hooks: Mapping[str, Any] | None = None,
        include_hook_events: bool = False,
        include_partial_messages: bool = False,
        fork_session: bool = False,
        agents: Mapping[str, Any] | None = None,
        setting_sources: Sequence[str] | None = None,
        enable_file_checkpointing: bool = False,
        tools: Any | None = None,
        skills: Sequence[str] | Literal["all"] | None = None,
        thinking: Any | None = None,
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        output_format: Mapping[str, Any] | None = None,
        add_dirs: Sequence[str | Path] | None = None,
        env: Mapping[str, str] | None = None,
        stderr: Callable[[str], None] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._options_factory = options_factory
        self._provided_options = options
        self._mesh = mesh
        self._client: Any | None = None
        self._connected = False
        policy_kwargs = tool_policy.claude_kwargs() if tool_policy is not None else {}
        policy_allowed = policy_kwargs.pop("allowed_tools", [])
        policy_disallowed = policy_kwargs.pop("disallowed_tools", [])
        if can_use_tool is None:
            can_use_tool = policy_kwargs.pop("can_use_tool", None)
        self._allowed_tools = _dedupe([*(allowed_tools or []), *policy_allowed])
        self._disallowed_tools = _dedupe([*(disallowed_tools or []), *policy_disallowed])
        self._permission_mode = permission_mode
        self._mcp_servers = mcp_servers
        self._cwd = cwd
        self._can_use_tool = can_use_tool
        wrapped_can_use_tool = self._wrap_can_use_tool(can_use_tool)
        self._options_kwargs = _runner_options_kwargs(
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            permission_mode=permission_mode,
            mcp_servers=self._mcp_servers,
            cwd=cwd,
            resume=resume,
            continue_conversation=continue_conversation,
            session_store=session_store,
            session_store_flush=session_store_flush,
            strict_mcp_config=strict_mcp_config,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            model=model,
            fallback_model=fallback_model,
            system_prompt=system_prompt,
            can_use_tool=wrapped_can_use_tool,
            permission_prompt_tool_name=permission_prompt_tool_name,
            hooks=dict(hooks or {}),
            include_hook_events=include_hook_events,
            include_partial_messages=include_partial_messages,
            fork_session=fork_session,
            agents=dict(agents or {}),
            setting_sources=list(setting_sources or []),
            enable_file_checkpointing=enable_file_checkpointing,
            tools=tools,
            skills=skills,
            thinking=thinking,
            effort=effort,
            output_format=dict(output_format or {}),
            add_dirs=list(add_dirs or []),
            env=dict(env or {}),
            stderr=stderr,
        )

    async def __aenter__(self) -> ClaudeAgentSDKClientSession:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        await self.disconnect()

    async def connect(self, prompt: str | None = None, *, mesh: Any | None = None) -> None:
        """Connect the underlying Claude SDK client once."""
        if mesh is not None:
            self._mesh = mesh
        if self._connected:
            return
        if self._client is None:
            client_factory, options_factory = self._sdk_bindings()
            options = self._provided_options
            if options is None:
                options = options_factory(**self._options_kwargs)
            self._client = client_factory(options=options)
        client = self._client
        connect = getattr(client, "connect", None)
        if connect is not None:
            if prompt is None:
                await _maybe_await(connect())
            else:
                await _maybe_await(connect(prompt))
        else:
            enter = getattr(client, "__aenter__", None)
            if enter is None:
                raise AgentError(
                    "ClaudeAgentSDKClientSession",
                    "ClaudeSDKClient exposes neither connect() nor __aenter__()",
                )
            self._client = await _maybe_await(enter())
        await _record_claude_event(
            self._mesh,
            "claude_client_connect",
            ClaudeAgentRunSignal(
                prompt=prompt if isinstance(prompt, str) else "",
                allowed_tools=list(self._allowed_tools),
                disallowed_tools=list(self._disallowed_tools),
                permission_mode=self._permission_mode,
                mcp_servers={name: {} for name in self._mcp_server_names()},
                cwd=str(self._cwd or ""),
            ),
            mcp_server_names=self._mcp_server_names(),
            cwd=str(self._cwd or ""),
            streaming_input=prompt is not None and not isinstance(prompt, str),
        )
        self._connected = True

    async def disconnect(self, *, mesh: Any | None = None) -> None:
        """Disconnect the underlying Claude SDK client if it is connected."""
        if mesh is not None:
            self._mesh = mesh
        if not self._connected or self._client is None:
            return
        client = self._client
        disconnect = getattr(client, "disconnect", None)
        if disconnect is not None:
            await _maybe_await(disconnect())
        else:
            exit_ = getattr(client, "__aexit__", None)
            if exit_ is not None:
                await _maybe_await(exit_(None, None, None))
        await _record_claude_event(
            self._mesh,
            "claude_client_disconnect",
            ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
            mcp_server_names=self._mcp_server_names(),
        )
        self._connected = False

    async def query(
        self,
        prompt: str,
        *,
        mesh: Any | None = None,
        session_id: str = "default",
        permission_mode: str = "",
        model: str = "",
    ) -> ClaudeAgentSDKResult:
        """Send one prompt through the continuous Claude client session."""
        await self.connect(mesh=mesh)
        if mesh is not None:
            self._mesh = mesh
        client = self._require_client()
        if permission_mode:
            await self.set_permission_mode(permission_mode, mesh=mesh)
        if model:
            await self.set_model(model, mesh=mesh)

        run_signal = ClaudeAgentRunSignal(
            prompt=prompt,
            allowed_tools=list(self._allowed_tools),
            disallowed_tools=list(self._disallowed_tools),
            permission_mode=permission_mode or self._permission_mode,
            mcp_servers={name: {} for name in self._mcp_server_names()},
            cwd=str(self._cwd or ""),
        )
        await _record_claude_event(
            mesh,
            "claude_client_query_start",
            run_signal,
            client_session_key=session_id,
            allowed_tools=list(self._allowed_tools),
            disallowed_tools=list(self._disallowed_tools),
            permission_mode=permission_mode or self._permission_mode,
            mcp_server_names=self._mcp_server_names(),
            cwd=str(self._cwd or ""),
        )

        await _maybe_await(client.query(prompt, session_id=session_id))

        result = await self.receive_response(
            mesh=mesh,
            run_signal=run_signal,
            allowed=self._allowed_tools,
            disallowed=self._disallowed_tools,
            mcp_servers=self._mcp_servers_mapping(),
            resumed_from_session_id="",
            continued=True,
        )
        return result

    async def ask(self, prompt: str, **kwargs: Any) -> ClaudeAgentSDKResult:
        """Alias for ``query`` that reads naturally in chat-like code."""
        return await self.query(prompt, **kwargs)

    async def run_turn(self, prompt: str, **kwargs: Any) -> ClaudeAgentSDKResult:
        """Alias for ``query`` that names the full send-and-drain lifecycle."""
        return await self.query(prompt, **kwargs)

    async def receive_messages(self, *, mesh: Any | None = None) -> AsyncIterator[Any]:
        """Yield raw SDK stream messages while recording sanitized stream events."""
        await self.connect(mesh=mesh)
        client = self._require_client()
        receive_messages = getattr(client, "receive_messages", None)
        if receive_messages is None:
            raise AgentError(
                "ClaudeAgentSDKClientSession",
                "ClaudeSDKClient does not expose receive_messages()",
            )
        async for message in receive_messages():
            await _record_claude_event(
                self._mesh,
                "claude_stream_event",
                ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
                message_type=type(message).__name__,
            )
            yield message

    async def receive_response(
        self,
        *,
        mesh: Any | None = None,
        run_signal: ClaudeAgentRunSignal | None = None,
        allowed: Sequence[str] | None = None,
        disallowed: Sequence[str] | None = None,
        mcp_servers: Mapping[str, Any] | None = None,
        resumed_from_session_id: str = "",
        continued: bool = True,
    ) -> ClaudeAgentSDKResult:
        """Drain the current SDK response until its result message."""
        await self.connect(mesh=mesh)
        return await _drain_client_response(
            self._require_client(),
            mesh=self._mesh,
            run_signal=run_signal or ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
            allowed=list(allowed if allowed is not None else self._allowed_tools),
            disallowed=list(disallowed if disallowed is not None else self._disallowed_tools),
            mcp_servers=mcp_servers or self._mcp_servers_mapping(),
            resumed_from_session_id=resumed_from_session_id,
            continued=continued,
        )

    async def interrupt(self, *, mesh: Any | None = None) -> None:
        """Interrupt the active Claude task; callers should drain its response."""
        await self._call_client_control("interrupt", "claude_interrupt", mesh=mesh)

    async def set_permission_mode(
        self,
        mode: str,
        *,
        mesh: Any | None = None,
    ) -> None:
        """Change Claude's permission mode for subsequent tool requests."""
        await self.connect(mesh=mesh)
        client = self._require_client()
        setter = getattr(client, "set_permission_mode", None)
        if setter is None:
            raise AgentError(
                "ClaudeAgentSDKClientSession",
                "ClaudeSDKClient does not expose set_permission_mode()",
            )
        await _maybe_await(setter(mode))
        self._permission_mode = mode
        signal = ClaudePermissionDecisionSignal(
            tool_name="*",
            decision="unknown",
            permission_mode=mode,
            reason="permission mode changed for continuous Claude session",
        )
        await _record_claude_event(
            self._mesh,
            "claude_permission_mode_changed",
            signal,
            permission_mode=mode,
        )

    async def set_model(
        self,
        model: str,
        *,
        mesh: Any | None = None,
    ) -> None:
        """Change Claude's model for subsequent messages when the SDK supports it."""
        await self.connect(mesh=mesh)
        client = self._require_client()
        setter = getattr(client, "set_model", None)
        if setter is None:
            raise AgentError(
                "ClaudeAgentSDKClientSession",
                "ClaudeSDKClient does not expose set_model()",
            )
        await _maybe_await(setter(model))
        await _record_claude_event(
            self._mesh,
            "claude_model_set",
            ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
            model=model,
        )

    async def get_mcp_status(self, *, mesh: Any | None = None) -> Any:
        """Return SDK MCP status and record a sanitized summary."""
        status = await self._call_client_control(
            "get_mcp_status",
            "claude_mcp_status",
            mesh=mesh,
            returns_value=True,
        )
        return status

    async def reconnect_mcp_server(
        self,
        server_name: str,
        *,
        mesh: Any | None = None,
    ) -> None:
        """Ask the SDK to reconnect one MCP server."""
        await self._call_client_control(
            "reconnect_mcp_server",
            "claude_mcp_reconnect",
            server_name,
            mesh=mesh,
            server_name=server_name,
        )

    async def toggle_mcp_server(
        self,
        server_name: str,
        enabled: bool,
        *,
        mesh: Any | None = None,
    ) -> None:
        """Enable or disable one MCP server mid-session."""
        await self._call_client_control(
            "toggle_mcp_server",
            "claude_mcp_toggle",
            server_name,
            enabled,
            mesh=mesh,
            server_name=server_name,
            enabled=enabled,
        )

    async def stop_task(self, task_id: str, *, mesh: Any | None = None) -> None:
        """Stop a Claude background task by SDK task id."""
        await self._call_client_control(
            "stop_task",
            "claude_task_stop",
            task_id,
            mesh=mesh,
            task_id=task_id,
        )

    async def rewind_files(self, user_message_id: str, *, mesh: Any | None = None) -> None:
        """Restore SDK file checkpoints to a prior user message."""
        await self._call_client_control(
            "rewind_files",
            "claude_rewind_files",
            user_message_id,
            mesh=mesh,
            user_message_id=user_message_id,
        )

    def _require_client(self) -> Any:
        if self._client is None:
            raise AgentError("ClaudeAgentSDKClientSession", "ClaudeSDKClient is not connected")
        return self._client

    def _sdk_bindings(self) -> tuple[ClaudeClientFactory, ClaudeOptionsFactory]:
        client_factory = self._client_factory
        options_factory = self._options_factory
        if client_factory is not None and options_factory is not None:
            return client_factory, options_factory
        loaded_client, loaded_options = _load_sdk_client_bindings()
        client_factory = client_factory or loaded_client
        options_factory = options_factory or loaded_options
        self._client_factory = client_factory
        self._options_factory = options_factory
        return client_factory, options_factory

    async def _call_client_control(
        self,
        method_name: str,
        action: str,
        *args: Any,
        mesh: Any | None = None,
        returns_value: bool = False,
        **metadata: Any,
    ) -> Any:
        await self.connect(mesh=mesh)
        client = self._require_client()
        method = getattr(client, method_name, None)
        if method is None:
            raise AgentError(
                "ClaudeAgentSDKClientSession",
                f"ClaudeSDKClient does not expose {method_name}()",
            )
        result = await _maybe_await(method(*args))
        if returns_value:
            metadata["status_summary"] = _safe_metadata_projection(result)
        await _record_claude_event(
            self._mesh,
            action,
            ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
            **metadata,
        )
        return result

    def _wrap_can_use_tool(self, handler: Any | None) -> Any | None:
        if handler is None:
            return None

        async def wrapped(tool_name: str, input_data: Any, context: Any) -> Any:
            result = await _maybe_await(handler(tool_name, input_data, context))
            input_keys = sorted(input_data) if isinstance(input_data, Mapping) else []
            decision = _permission_decision_from_result(result)
            await _record_claude_event(
                self._mesh,
                "claude_permission_decision",
                ClaudePermissionDecisionSignal(
                    tool_name=tool_name,
                    decision=decision,
                    permission_mode=self._permission_mode,
                    reason=_permission_reason_from_result(result),
                ),
                tool_name=tool_name,
                decision=decision,
                tool_input_keys=input_keys,
            )
            return result

        return wrapped

    def _mcp_server_names(self) -> list[str]:
        if isinstance(self._mcp_servers, Mapping):
            return sorted(str(name) for name in self._mcp_servers)
        if self._mcp_servers:
            return [str(self._mcp_servers)]
        return []

    def _mcp_servers_mapping(self) -> Mapping[str, Any]:
        if isinstance(self._mcp_servers, Mapping):
            return self._mcp_servers
        return {}


ClaudeAgentSDKSession = ClaudeAgentSDKClientSession


async def _drain_client_response(
    client: Any,
    *,
    mesh: Any | None,
    run_signal: ClaudeAgentRunSignal,
    allowed: Sequence[str],
    disallowed: Sequence[str],
    mcp_servers: Mapping[str, Any],
    resumed_from_session_id: str,
    continued: bool,
) -> ClaudeAgentSDKResult:
    receive_response = getattr(client, "receive_response", None)
    if receive_response is None:
        raise AgentError(
            "ClaudeAgentSDKClientSession",
            "ClaudeSDKClient does not expose receive_response()",
        )

    session_id = ""
    result_text = ""
    subtype = ""
    total_cost_usd: float | None = None
    message_count = 0

    async for message in receive_response():
        message_count += 1
        session_id = _session_id_from_message(message) or session_id
        mcp_init = _mcp_init_metadata(message)
        if mcp_init:
            await _record_claude_event(
                mesh,
                "claude_mcp_init",
                run_signal,
                claude_session_id=session_id,
                **mcp_init,
            )
        for event in _tool_events_from_message(message, session_id):
            await _record_claude_event(
                mesh,
                f"claude_{event.event}",
                event,
                claude_session_id=session_id,
                tool_name=event.tool_name,
                tool_use_id=event.tool_call_id,
                tool_input_keys=event.tool_input_keys,
                tool_allowed=_tool_is_allowed(event.tool_name, allowed, disallowed),
                mcp_server=event.mcp_server,
            )
        result = _result_from_message(message)
        if result is not None:
            result_text = result["text"]
            subtype = result["subtype"]
            total_cost_usd = result["total_cost_usd"]
            final_signal = ClaudeAgentResultSignal(
                text=result_text,
                session_id=session_id,
                subtype=subtype,
                total_cost_usd=total_cost_usd,
                message_count=message_count,
                allowed_tools=list(allowed),
                disallowed_tools=list(disallowed),
                mcp_servers=sorted(mcp_servers),
                resumed_from_session_id=resumed_from_session_id,
                continued=continued,
            )
            denials = _permission_denials_from_message(message)
            await _record_claude_event(
                mesh,
                "claude_result",
                final_signal,
                claude_session_id=session_id,
                permission_denial_count=len(denials),
                permission_denied_tools=[
                    item["tool_name"] for item in denials if item.get("tool_name")
                ],
            )

    return ClaudeAgentSDKResult(
        session_id=session_id,
        result=result_text,
        subtype=subtype,
        total_cost_usd=total_cost_usd,
        message_count=message_count,
    )


def _load_sdk_bindings() -> tuple[ClaudeQueryFn, ClaudeOptionsFactory]:
    try:
        sdk = import_module("claude_agent_sdk")
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Claude Agent SDK integration requires claude-agent-sdk. "
            "Install it with: pip install 'signal-gating[claude]'"
        ) from e
    return getattr(sdk, "query"), getattr(sdk, "ClaudeAgentOptions")


def _load_sdk_client_bindings() -> tuple[ClaudeClientFactory, ClaudeOptionsFactory]:
    try:
        sdk = import_module("claude_agent_sdk")
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Claude Agent SDK session integration requires claude-agent-sdk. "
            "Install it with: pip install 'signal-gating[claude]'"
        ) from e
    return getattr(sdk, "ClaudeSDKClient"), getattr(sdk, "ClaudeAgentOptions")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _runner_options_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if value not in (None, "", []) and value != {}
    }


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))


def _tool_rule_matches_any(tool_name: str, rules: Sequence[str]) -> bool:
    return any(_tool_rule_matches(tool_name, rule) for rule in rules)


def _tool_rule_matches(tool_name: str, rule: str) -> bool:
    if rule == tool_name:
        return True
    if rule.endswith("*"):
        return tool_name.startswith(rule[:-1])
    if "(" in rule:
        return rule.split("(", 1)[0] == tool_name
    return False


def _tool_request_signal(tool_name: str, input_data: Any, context: Any) -> ClaudeToolRequestSignal:
    input_keys = sorted(input_data) if isinstance(input_data, Mapping) else []
    return ClaudeToolRequestSignal(
        tool_name=tool_name,
        tool_input_keys=list(input_keys),
        mcp_server=_mcp_server_from_tool_name(tool_name),
        blocked_path=str(getattr(context, "blocked_path", "") or ""),
        decision_reason=str(getattr(context, "decision_reason", "") or ""),
        display_name=str(getattr(context, "display_name", "") or ""),
    )


def _permission_result_allow() -> Any:
    try:
        sdk = import_module("claude_agent_sdk")
        result_type = getattr(sdk, "PermissionResultAllow")
    except (ImportError, AttributeError):
        return ClaudePermissionResultAllowFallback()
    return result_type()


def _permission_result_deny(message: str, *, interrupt: bool = False) -> Any:
    try:
        sdk = import_module("claude_agent_sdk")
        result_type = getattr(sdk, "PermissionResultDeny")
    except (ImportError, AttributeError):
        return ClaudePermissionResultDenyFallback(message=message, interrupt=interrupt)
    return result_type(message=message, interrupt=interrupt)


def _mcp_tool_schema(name: str, owner: str, spec: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": spec.description,
        "inputSchema": {
            "type": "object",
            "properties": {
                param: _mcp_parameter_schema(meta)
                for param, meta in spec.parameters.items()
            },
            "required": [
                param for param, meta in spec.parameters.items() if meta.get("required")
            ],
            "additionalProperties": False,
        },
        "annotations": {
            "title": f"{owner}.{name}",
        },
    }


def _tool_argument_error(spec: Any, arguments: Mapping[str, Any]) -> str:
    parameters = getattr(spec, "parameters", {})
    for name in arguments:
        if name not in parameters:
            return f"unexpected argument {name!r}"

    for name, meta in parameters.items():
        if meta.get("required") and name not in arguments:
            return f"missing required argument {name!r}"
        if name not in arguments:
            continue
        expected = str(meta.get("type") or "")
        if expected and not _matches_tool_type(arguments[name], expected):
            got = type(arguments[name]).__name__
            return f"argument {name!r} expected {expected}, got {got}"
    return ""


def _matches_tool_type(value: Any, expected: str) -> bool:
    raw = expected.lower()
    if raw in {"str", "string"}:
        return isinstance(value, str)
    if raw in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if raw in {"float", "number"}:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if raw in {"bool", "boolean"}:
        return isinstance(value, bool)
    if raw in {"dict", "mapping", "object"}:
        return isinstance(value, dict)
    if raw in {"list", "sequence", "array"}:
        return isinstance(value, list)
    return True


def _mcp_parameter_schema(meta: Mapping[str, Any]) -> dict[str, Any]:
    schema = {"type": _json_schema_type(meta.get("type"))}
    if "default" in meta:
        schema["default"] = meta["default"]
    return schema


def _json_schema_type(type_name: Any) -> str:
    raw = str(type_name or "").lower()
    if raw in {"str", "string"}:
        return "string"
    if raw in {"int", "integer"}:
        return "integer"
    if raw in {"float", "number"}:
        return "number"
    if raw in {"bool", "boolean"}:
        return "boolean"
    if raw in {"dict", "mapping", "object"}:
        return "object"
    if raw in {"list", "sequence", "array"}:
        return "array"
    return "string"


def _mcp_success_result(result: Any) -> dict[str, Any]:
    projected = _safe_metadata_projection(result, depth=4)
    payload = {
        "content": [{"type": "text", "text": _mcp_text(projected)}],
        "isError": False,
    }
    if isinstance(projected, Mapping):
        payload["structuredContent"] = dict(projected)
    return payload


def _mcp_error_result(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _jsonrpc_error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _jsonrpc_line(message: Mapping[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n"


def _jsonrpc_http_body(message: Mapping[str, Any]) -> bytes:
    return json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _asgi_headers(scope: Mapping[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", []):
        name = bytes(raw_name).decode("latin-1").lower()
        value = bytes(raw_value).decode("latin-1")
        if name in headers:
            headers[name] = f"{headers[name]},{value}"
        else:
            headers[name] = value
    return headers


async def _read_asgi_body(receive: ASGIReceive, max_body_bytes: int) -> bytes:
    body = bytearray()
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            continue
        chunk = message.get("body", b"")
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        body.extend(chunk)
        if len(body) > max_body_bytes:
            raise ValueError("HTTP request body is too large")
        if not message.get("more_body", False):
            return bytes(body)


async def _send_http_response(
    send: ASGISend,
    status: int,
    body: bytes = b"",
    content_type: str | None = None,
    *,
    headers: Sequence[tuple[bytes, bytes]] | None = None,
) -> None:
    response_headers = list(headers or [])
    if content_type is not None:
        response_headers.append((b"content-type", content_type.encode("ascii")))
    response_headers.append((b"content-length", str(len(body)).encode("ascii")))
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": response_headers,
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })


async def _send_jsonrpc_http_response(
    send: ASGISend,
    message: Mapping[str, Any],
    *,
    headers: Sequence[tuple[bytes, bytes]] | None = None,
) -> None:
    await _send_http_response(
        send,
        200,
        _jsonrpc_http_body(message),
        "application/json; charset=utf-8",
        headers=headers,
    )


async def _send_jsonrpc_http_error(
    send: ASGISend,
    status: int,
    code: int,
    message: str,
    *,
    headers: Sequence[tuple[bytes, bytes]] | None = None,
) -> None:
    await _send_http_response(
        send,
        status,
        _jsonrpc_http_body(_jsonrpc_error_response(None, code, message)),
        "application/json; charset=utf-8",
        headers=headers,
    )


def _bearer_authorization(headers: Mapping[str, str]) -> tuple[str, str]:
    raw = headers.get("authorization", "")
    if "," in raw:
        return "", ""
    scheme, sep, token = raw.partition(" ")
    if not sep or scheme.lower() != "bearer" or not token.strip():
        return scheme, ""
    return scheme, token.strip()


def protected_resource_metadata_url(
    resource: str,
    *,
    well_known_path: str = "/.well-known/oauth-protected-resource",
) -> str:
    """Return the RFC 9728 well-known metadata URL for a resource identifier."""
    parsed = urlparse(resource)
    if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
        raise ValueError("resource must be an https URL without a fragment")
    resource_path = "" if parsed.path in {"", "/"} else parsed.path
    metadata_path = well_known_path.rstrip("/") + resource_path
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        metadata_path,
        "",
        parsed.query,
        "",
    ))


def _url_path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def _coerce_protected_resource_metadata(
    metadata: ClaudeMCPProtectedResourceMetadata | Mapping[str, Any] | None,
) -> ClaudeMCPProtectedResourceMetadata | None:
    if metadata is None:
        return None
    if isinstance(metadata, ClaudeMCPProtectedResourceMetadata):
        return metadata
    resource = metadata.get("resource")
    authorization_servers = metadata.get("authorization_servers")
    if not isinstance(resource, str):
        raise ValueError("protected_resource_metadata.resource must be a string")
    if not isinstance(authorization_servers, Sequence) or isinstance(authorization_servers, str):
        raise ValueError("protected_resource_metadata.authorization_servers must be a sequence")
    known = {
        "resource",
        "authorization_servers",
        "scopes_supported",
        "bearer_methods_supported",
        "resource_name",
        "resource_documentation",
        "resource_policy_uri",
        "resource_tos_uri",
    }
    extra = {str(key): value for key, value in metadata.items() if str(key) not in known}
    return ClaudeMCPProtectedResourceMetadata(
        resource=resource,
        authorization_servers=tuple(str(value) for value in authorization_servers),
        scopes_supported=_string_tuple(metadata.get("scopes_supported")),
        bearer_methods_supported=_string_tuple(
            metadata.get("bearer_methods_supported"),
            default=("header",),
        ),
        resource_name=str(metadata.get("resource_name", "Signal Gating Protocol MCP")),
        resource_documentation=str(metadata.get("resource_documentation", "")),
        resource_policy_uri=str(metadata.get("resource_policy_uri", "")),
        resource_tos_uri=str(metadata.get("resource_tos_uri", "")),
        extra=extra,
    )


def _string_tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return default


def _nonempty_string_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: Sequence[str] = (value,)
    else:
        raw_values = value
    return tuple(str(item) for item in raw_values if str(item))


def _claim_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _first_claim_value(claims: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        values = _claim_values(claims.get(name))
        if values:
            return values[0]
    return ""


def _claims_contain_value(
    claims: Mapping[str, Any],
    names: Sequence[str],
    expected_values: Sequence[str],
) -> bool:
    return bool(_matching_claim_value(claims, names, expected_values))


def _matching_claim_value(
    claims: Mapping[str, Any],
    names: Sequence[str],
    expected_values: Sequence[str],
) -> str:
    expected = set(expected_values)
    for name in names:
        for value in _claim_values(claims.get(name)):
            if value in expected:
                return value
    return ""


def _resource_claim_matches(
    claims: Mapping[str, Any],
    names: Sequence[str],
    expected_resource: str,
) -> bool:
    for name in names:
        values = _claim_values(claims.get(name))
        if values:
            return expected_resource in values
    return False


def _claims_scopes(claims: Mapping[str, Any], names: Sequence[str]) -> tuple[str, ...]:
    scopes: set[str] = set()
    for name in names:
        for raw_scope in _claim_values(claims.get(name)):
            scopes.update(_http_authorization_scopes(raw_scope))
    return tuple(sorted(scopes))


def _time_claim_valid(
    value: Any,
    now: float,
    leeway_seconds: float,
    *,
    upper_bound: bool,
) -> bool:
    if value is None:
        return True
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return False
    if upper_bound:
        return timestamp > now - leeway_seconds
    return timestamp <= now + leeway_seconds


def _jwk_from_set(jwks: Mapping[str, Any], kid: str) -> Mapping[str, Any]:
    keys = jwks.get("keys")
    if not isinstance(keys, Sequence) or isinstance(keys, str):
        raise ValueError("JWKS must contain a keys array")
    candidates = [key for key in keys if isinstance(key, Mapping)]
    if kid:
        for key in candidates:
            if str(key.get("kid", "")) == kid:
                return key
        raise ValueError("JWKS does not contain a key for the token kid")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("JWT kid is required when JWKS contains multiple keys")


def _query_contains_access_token(query_string: str) -> bool:
    return any(
        name == "access_token"
        for name, _ in parse_qsl(query_string, keep_blank_values=True)
    )


def _coerce_http_authorization_result(
    decision: ClaudeMCPHTTPAuthorizationDecision,
) -> ClaudeMCPHTTPAuthorizationResult:
    if isinstance(decision, ClaudeMCPHTTPAuthorizationResult):
        return decision
    if isinstance(decision, bool):
        return ClaudeMCPHTTPAuthorizationResult(allowed=decision)
    if isinstance(decision, str):
        if not decision:
            return ClaudeMCPHTTPAuthorizationResult(allowed=False)
        return ClaudeMCPHTTPAuthorizationResult(allowed=True, principal=decision)
    return ClaudeMCPHTTPAuthorizationResult(allowed=False)


def _authorization_reason(
    decision: ClaudeMCPHTTPAuthorizationResult,
    status_code: int,
) -> str:
    challenge = decision.www_authenticate.lower()
    message = decision.message.lower()
    if "insufficient_scope" in challenge or (
        "insufficient" in message and "scope" in message
    ):
        return "insufficient_scope"
    if "invalid_token" in challenge or ("invalid" in message and "token" in message):
        return "invalid_token"
    if status_code == 403:
        return "forbidden"
    return "unauthorized"


def _authorization_action(allowed: bool, reason: str) -> str:
    if allowed:
        return "claude_mcp_http_auth_allowed"
    if reason == "query_token_rejected":
        return "claude_mcp_http_auth_query_token_rejected"
    if reason == "authorization_required":
        return "claude_mcp_http_auth_challenged"
    if reason == "session_authorization_mismatch":
        return "claude_mcp_http_auth_session_mismatch"
    return "claude_mcp_http_auth_denied"


def _authorization_components(
    decision: ClaudeMCPHTTPAuthorizationResult | None,
    auth_identity: str,
) -> dict[str, Any]:
    if decision is not None:
        scopes = _http_authorization_scopes(decision.scopes)
        principal_present = bool(decision.principal)
        return {
            "principal_hash": _token_fingerprint(decision.principal) if decision.principal else "",
            "principal_present": principal_present,
            "audience_present": bool(decision.audience),
            "resource_present": bool(decision.resource),
            "scope_count": len(scopes),
            "identity_binding_kind": "claims" if principal_present else "token_fingerprint",
        }
    if not auth_identity.startswith("claims:"):
        identity_binding_kind = (
            "token_fingerprint" if auth_identity.startswith("sha256:") else "none"
        )
        return {
            "principal_hash": "",
            "principal_present": False,
            "audience_present": False,
            "resource_present": False,
            "scope_count": 0,
            "identity_binding_kind": identity_binding_kind,
        }
    try:
        claims = json.loads(auth_identity.removeprefix("claims:"))
    except json.JSONDecodeError:
        return {
            "principal_hash": "",
            "principal_present": False,
            "audience_present": False,
            "resource_present": False,
            "scope_count": 0,
            "identity_binding_kind": "none",
        }
    if not isinstance(claims, Mapping):
        return {
            "principal_hash": "",
            "principal_present": False,
            "audience_present": False,
            "resource_present": False,
            "scope_count": 0,
            "identity_binding_kind": "none",
        }
    principal = str(claims.get("principal", "") or "")
    scopes = _http_authorization_scopes(claims.get("scopes"))
    return {
        "principal_hash": _token_fingerprint(principal) if principal else "",
        "principal_present": bool(principal),
        "audience_present": bool(claims.get("audience")),
        "resource_present": bool(claims.get("resource")),
        "scope_count": len(scopes),
        "identity_binding_kind": "claims",
    }


def _http_authorization_identity(
    decision: ClaudeMCPHTTPAuthorizationResult,
    token: str,
) -> str:
    if not decision.principal:
        return _token_fingerprint(token)
    identity = {
        "audience": str(decision.audience),
        "principal": str(decision.principal),
        "resource": str(decision.resource),
        "scopes": list(_http_authorization_scopes(decision.scopes)),
        "version": 1,
    }
    return "claims:" + json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _http_authorization_scopes(scopes: Any) -> tuple[str, ...]:
    if scopes is None:
        return ()
    raw_scopes: Iterable[Any]
    if isinstance(scopes, str):
        raw_scopes = scopes.split()
    elif isinstance(scopes, Sequence):
        raw_scopes = scopes
    else:
        raw_scopes = (scopes,)
    normalized: set[str] = set()
    for raw_scope in raw_scopes:
        for scope in str(raw_scope).split():
            if scope:
                normalized.add(scope)
    return tuple(sorted(normalized))


async def _resolve_http_authorization_decision(
    callback: ClaudeMCPHTTPAuthorizeFn,
    context: ClaudeMCPHTTPAuthorizationContext,
) -> ClaudeMCPHTTPAuthorizationDecision:
    decision = callback(context)
    if inspect.isawaitable(decision):
        return await decision
    return decision


def _bearer_challenge(realm: str, resource_metadata_url: str | None) -> str:
    challenge = f'Bearer realm="{_quote_http_auth_param(realm)}"'
    if resource_metadata_url:
        challenge += f', resource_metadata="{_quote_http_auth_param(resource_metadata_url)}"'
    return challenge


def _bearer_error_challenge(error: str, *, scope: str = "") -> str:
    challenge = f'Bearer error="{_quote_http_auth_param(error)}"'
    if scope:
        challenge += f', scope="{_quote_http_auth_param(scope)}"'
    return challenge


def _challenge_with_resource_metadata(challenge: str, resource_metadata_url: str | None) -> str:
    if (
        not resource_metadata_url
        or not challenge.lower().startswith("bearer")
        or "resource_metadata=" in challenge.lower()
    ):
        return challenge
    return (
        challenge
        + f', resource_metadata="{_quote_http_auth_param(resource_metadata_url)}"'
    )


def _quote_http_auth_param(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _token_fingerprint(token: str) -> str:
    return "sha256:" + sha256(token.encode("utf-8")).hexdigest()


def _http_accepts(headers: Mapping[str, str], media_type: str) -> bool:
    raw = headers.get("accept", "")
    accepted = {
        part.split(";", 1)[0].strip().lower()
        for part in raw.split(",")
        if part.strip()
    }
    if not accepted:
        return False
    top_level = media_type.split("/", 1)[0]
    return media_type in accepted or "*/*" in accepted or f"{top_level}/*" in accepted


def _http_content_type(headers: Mapping[str, str]) -> str:
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _mcp_http_protocol_version_supported(headers: Mapping[str, str]) -> bool:
    protocol_version = headers.get("mcp-protocol-version")
    return protocol_version in {None, ClaudeMeshMCPHTTPApp.protocol_version}


def _is_jsonrpc_response_message(message: Mapping[str, Any]) -> bool:
    return (
        message.get("jsonrpc") == "2.0"
        and "method" not in message
        and "id" in message
        and ("result" in message or "error" in message)
    )


def _origin_host(origin: str) -> str:
    return str(urlparse(origin).hostname or "")


def _mcp_text(value: Any) -> str:
    if isinstance(value, Mapping | list):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, str):
        return value
    return str(value)


def _permission_decision_from_result(result: Any) -> ClaudePermissionDecision:
    raw = str(
        getattr(result, "decision", "")
        or getattr(result, "behavior", "")
        or getattr(result, "type", "")
        or type(result).__name__
    ).lower()
    if "deny" in raw or "denied" in raw:
        return "denied"
    if "allow" in raw or "allowed" in raw:
        return "allowed"
    if "prompt" in raw:
        return "prompted"
    return "unknown"


def _permission_reason_from_result(result: Any) -> str:
    raw = getattr(result, "message", "") or getattr(result, "reason", "")
    return str(raw) if raw else ""


def _safe_metadata_projection(value: Any, *, depth: int = 2) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if depth <= 0:
        return type(value).__name__
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_metadata_projection(item, depth=depth - 1)
            for key, item in value.items()
            if not _sensitive_metadata_key(str(key))
        }
    if isinstance(value, list | tuple | set):
        return [_safe_metadata_projection(item, depth=depth - 1) for item in list(value)[:20]]
    data = getattr(value, "__dict__", None)
    if isinstance(data, Mapping):
        return _safe_metadata_projection(data, depth=depth - 1)
    return type(value).__name__


def _sensitive_metadata_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in {"env", "environment"}
        or "token" in lowered
        or "secret" in lowered
        or "api_key" in lowered
        or "apikey" in lowered
    )


async def _record_claude_event(
    mesh: Any | None,
    action: str,
    signal: Signal,
    **metadata: Any,
) -> None:
    if mesh is None:
        return
    record_event = getattr(mesh, "_record_event")
    await record_event(
        action,
        signal,
        source="claude_agent_sdk",
        event_kind="claude_agent_sdk",
        **metadata,
    )


async def _record_claude_http_event(
    mesh: Any | None,
    action: str,
    signal: Signal,
    **metadata: Any,
) -> None:
    if mesh is None:
        return
    record_event = getattr(mesh, "_record_event")
    await record_event(
        action,
        signal,
        source="claude_mcp_http",
        event_kind="claude_mcp_http",
        **metadata,
    )


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


def _session_id_from_message(message: Any) -> str:
    raw = _message_value(message, "session_id")
    if isinstance(raw, str) and raw:
        return raw
    data = _message_value(message, "data")
    if isinstance(data, Mapping):
        nested = data.get("session_id")
        if isinstance(nested, str):
            return nested
    return ""


def _result_from_message(message: Any) -> dict[str, Any] | None:
    raw = _message_value(message, "result")
    if raw is None:
        return None
    cost = _message_value(message, "total_cost_usd")
    return {
        "text": raw if isinstance(raw, str) else str(raw),
        "subtype": str(_message_value(message, "subtype") or ""),
        "total_cost_usd": float(cost) if isinstance(cost, int | float) else None,
    }


def _tool_events_from_message(message: Any, session_id: str) -> list[ClaudeToolEventSignal]:
    content = _message_value(message, "content")
    if not isinstance(content, list):
        return []
    events: list[ClaudeToolEventSignal] = []
    for block in content:
        raw_kind = str(_message_value(block, "type") or "")
        if raw_kind == "tool_use":
            kind: ClaudeToolEventKind = "tool_use"
        elif raw_kind == "tool_result":
            kind = "tool_result"
        else:
            continue
        tool_name = str(_message_value(block, "name") or _message_value(block, "tool_name") or "")
        tool_call_id = str(
            _message_value(block, "id") or _message_value(block, "tool_call_id") or ""
        )
        parent_tool_use_id = str(_message_value(message, "parent_tool_use_id") or "")
        status = str(_message_value(block, "status") or "")
        raw_input = _message_value(block, "input")
        input_keys = sorted(raw_input) if isinstance(raw_input, Mapping) else []
        events.append(
            ClaudeToolEventSignal(
                event=kind,
                tool_name=tool_name,
                session_id=session_id,
                tool_call_id=tool_call_id,
                parent_tool_use_id=parent_tool_use_id,
                mcp_server=_mcp_server_from_tool_name(tool_name),
                status=status,
                tool_input_keys=list(input_keys),
            )
        )
    return events


def _mcp_server_from_tool_name(tool_name: str) -> str:
    if not tool_name.startswith("mcp__"):
        return ""
    parts = tool_name.split("__", 2)
    return parts[1] if len(parts) == 3 else ""


def _mcp_init_metadata(message: Any) -> dict[str, Any]:
    data = _message_value(message, "data")
    if not isinstance(data, Mapping):
        return {}
    raw_servers = data.get("mcp_servers")
    if not isinstance(raw_servers, list):
        return {}
    statuses: dict[str, str] = {}
    tool_names: list[str] = []
    failed: list[str] = []
    for server in raw_servers:
        if not isinstance(server, Mapping):
            continue
        name = server.get("name")
        if not isinstance(name, str):
            continue
        status = str(server.get("status") or "")
        statuses[name] = status
        if status == "failed":
            failed.append(name)
        tools = server.get("tools")
        if isinstance(tools, list):
            tool_names.extend(str(tool) for tool in tools)
    return {
        "mcp_server_statuses": statuses,
        "mcp_tool_names": sorted(tool_names),
        "failed_mcp_servers": sorted(failed),
    }


def _permission_denials_from_message(message: Any) -> list[dict[str, str]]:
    raw = _message_value(message, "permission_denials")
    if not isinstance(raw, list):
        return []
    denials: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, Mapping):
            tool_name = item.get("tool_name")
            reason = item.get("reason")
            denials.append({
                "tool_name": str(tool_name or ""),
                "reason": str(reason or ""),
            })
    return denials


def _tool_is_allowed(tool_name: str, allowed: Sequence[str], disallowed: Sequence[str]) -> bool:
    if tool_name in disallowed:
        return False
    if tool_name in allowed:
        return True
    for rule in allowed:
        if rule.endswith("*") and tool_name.startswith(rule[:-1]):
            return True
    return False


__all__ = [
    "ClaudeAgent",
    "ClaudeAgentSDKResult",
    "ClaudeAgentSDKClientSession",
    "ClaudeAgentSDKRunner",
    "ClaudeAgentSDKSession",
    "ClaudeClientFactory",
    "ClaudeMCPBearerTokenValidator",
    "ClaudeMCPHTTPAuthorizationContext",
    "ClaudeMCPHTTPAuthorizationDecision",
    "ClaudeMCPHTTPAuthorizationResult",
    "ClaudeMCPHTTPAuthorizationSignal",
    "ClaudeMCPHTTPAuthorizeFn",
    "ClaudeMCPJWKSCache",
    "ClaudeMCPJWKSLoader",
    "ClaudeMCPJWTBearerAuthorizer",
    "ClaudeMCPProtectedResourceMetadata",
    "ClaudeMCPTokenClaims",
    "ClaudeMCPTokenDecodeFn",
    "ClaudeMeshMCPAdapter",
    "ClaudeMeshMCPHTTPApp",
    "ClaudeMeshMCPStdioServer",
    "ClaudeAgentResultSignal",
    "ClaudeAgentRunSignal",
    "ClaudePermissionDecision",
    "ClaudePermissionDecisionSignal",
    "ClaudePermissionResultAllowFallback",
    "ClaudePermissionResultDenyFallback",
    "ClaudeQueryFn",
    "ClaudeToolEventKind",
    "ClaudeToolEventSignal",
    "ClaudeToolPolicy",
    "ClaudeToolRequestSignal",
    "claude_options",
    "mcp_tool_name",
    "protected_resource_metadata_url",
]
