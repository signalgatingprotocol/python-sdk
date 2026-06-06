"""Tests for production-style HTTP MCP bearer token authorization helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from signal_gating import (
    AgentError,
    ClaudeMCPBearerTokenValidator,
    ClaudeMCPHTTPAuthorizationContext,
    ClaudeMCPJWTBearerAuthorizer,
    ClaudeMeshMCPAdapter,
    ClaudeMeshMCPHTTPApp,
    Mesh,
)
from tests.test_claude_mcp_http import POST_HEADERS, _call_http_app, _json_body

NOW = 1_700_000_000.0
VALID_CLAIMS = {
    "iss": "https://auth.example.com",
    "aud": ["https://example.com/mcp", "sgp-api"],
    "resource": "https://example.com/mcp",
    "sub": "user-123",
    "scope": "tools.read tools.call",
    "exp": NOW + 300,
    "nbf": NOW - 30,
}


def _rsa_key_and_jwk(kid: str) -> tuple[Any, dict[str, Any]]:
    rsa: Any = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    jwt_algorithms: Any = pytest.importorskip("jwt.algorithms")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk = json.loads(jwt_algorithms.RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private_key, jwk


def _rs256_claims(claim_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    now = int(time.time())
    claims = {
        "iss": "https://auth.example.com",
        "aud": ["https://example.com/mcp", "sgp-api"],
        "resource": "https://example.com/mcp",
        "sub": "user-123",
        "client_id": "client-123",
        "iat": now,
        "jti": "token-id-123",
        "scope": "tools.read tools.call",
        "exp": now + 300,
        "nbf": now - 1,
    }
    claims.update(claim_overrides or {})
    return claims


def _encode_rs256_token(
    private_key: Any,
    *,
    kid: str | None = "test-key",
    claim_overrides: dict[str, Any] | None = None,
    header_overrides: dict[str, Any] | None = None,
) -> str:
    jwt: Any = pytest.importorskip("jwt")
    headers = {"typ": "at+jwt"}
    if kid is not None:
        headers["kid"] = kid
    headers.update(header_overrides or {})
    token = jwt.encode(
        _rs256_claims(claim_overrides),
        private_key,
        algorithm="RS256",
        headers=headers,
    )
    assert isinstance(token, str)
    return token


def _signed_rs256_token(
    *,
    claim_overrides: dict[str, Any] | None = None,
    header_overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    private_key, jwk = _rsa_key_and_jwk("test-key")
    token = _encode_rs256_token(
        private_key,
        claim_overrides=claim_overrides,
        header_overrides=header_overrides,
    )
    return token, {"keys": [jwk]}


def _validator(**overrides: Any) -> ClaudeMCPBearerTokenValidator:
    claims = dict(VALID_CLAIMS)
    claims.update(overrides.pop("claims", {}))

    def decode(token: str) -> dict[str, Any]:
        if token == "explode-secret-token":
            raise ValueError(f"bad token {token}")
        return claims

    return ClaudeMCPBearerTokenValidator(
        decode=decode,
        issuer="https://auth.example.com",
        audience="https://example.com/mcp",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
        now=lambda: NOW,
        **overrides,
    )


def test_claude_mcp_bearer_validator_accepts_valid_claims() -> None:
    decision = _validator().validate_claims(VALID_CLAIMS)

    assert decision.allowed is True
    assert decision.principal == "user-123"
    assert decision.audience == "https://example.com/mcp"
    assert decision.resource == "https://example.com/mcp"
    assert decision.scopes == ("tools.call", "tools.read")


@pytest.mark.parametrize(
    ("case", "claim_overrides"),
    [
        ("expired", {"exp": NOW - 1}),
        ("not_before", {"nbf": NOW + 61}),
        ("wrong_issuer", {"iss": "https://other.example.com"}),
        ("wrong_audience", {"aud": ["other-api"]}),
        ("wrong_resource", {"resource": "https://other.example.com/mcp"}),
        ("null_exp", {"exp": None}),
        ("missing_exp", {"exp": None}),
        ("missing_principal", {"sub": ""}),
    ],
)
def test_claude_mcp_bearer_validator_rejects_invalid_claims_with_401(
    case: str,
    claim_overrides: dict[str, Any],
) -> None:
    claims = dict(VALID_CLAIMS)
    claims.update(claim_overrides)
    if case == "missing_exp":
        claims.pop("exp")

    decision = _validator().validate_claims(claims)

    assert decision.allowed is False
    assert decision.status_code == 401
    assert decision.message == "invalid token"
    assert decision.www_authenticate == 'Bearer error="invalid_token"'


def test_claude_mcp_bearer_validator_rejects_missing_required_scope_with_403() -> None:
    claims = {**VALID_CLAIMS, "scope": "tools.read"}

    decision = _validator().validate_claims(claims)

    assert decision.allowed is False
    assert decision.status_code == 403
    assert decision.message == "insufficient scope"
    assert decision.www_authenticate == (
        'Bearer error="insufficient_scope", scope="tools.call"'
    )


async def test_claude_mcp_bearer_validator_awaits_async_claim_decoder() -> None:
    seen: list[str] = []

    async def decode(token: str) -> dict[str, Any]:
        seen.append(token)
        return dict(VALID_CLAIMS)

    validator = ClaudeMCPBearerTokenValidator(
        decode=decode,
        issuer="https://auth.example.com",
        audience="https://example.com/mcp",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
        now=lambda: NOW,
    )

    decision = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token="async-token",
        )
    )

    assert seen == ["async-token"]
    assert decision.allowed is True
    assert decision.principal == "user-123"


async def test_claude_mcp_bearer_validator_redacts_decode_failures_in_http_response() -> None:
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        authorize_http=_validator(),
    )

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer explode-secret-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 401
    assert response.json()["error"]["message"] == "invalid token"
    assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    assert "explode-secret-token" not in response.body.decode("utf-8")
    assert "explode-secret-token" not in response.headers["www-authenticate"]


async def test_claude_mcp_bearer_validator_insufficient_scope_uses_403_challenge() -> None:
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        authorize_http=_validator(claims={"scope": "tools.read"}),
        protected_resource_metadata_url=(
            "https://example.com/.well-known/oauth-protected-resource/mcp"
        ),
    )

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": "Bearer scoped-token"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 403
    assert response.json()["error"]["message"] == "insufficient scope"
    assert response.headers["www-authenticate"] == (
        'Bearer error="insufficient_scope", scope="tools.call", '
        'resource_metadata="https://example.com/.well-known/oauth-protected-resource/mcp"'
    )


def test_claude_mcp_jwt_bearer_authorizer_aliases_claim_validator() -> None:
    assert ClaudeMCPJWTBearerAuthorizer is ClaudeMCPBearerTokenValidator


def test_claude_mcp_pyjwt_authorizer_requires_auth_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal_gating.claude as claude_module

    def missing_jwt(name: str) -> Any:
        if name == "jwt":
            raise ImportError("jwt is not installed")
        raise AssertionError(name)

    monkeypatch.setattr(claude_module, "import_module", missing_jwt)

    with pytest.raises(AgentError, match=r"signal-gating\[auth\]"):
        ClaudeMCPBearerTokenValidator.pyjwt(signing_key="secret", algorithms=("HS256",))


async def test_claude_mcp_pyjwt_authorizer_accepts_signed_rs256_jwks_token() -> None:
    token, jwks = _signed_rs256_token()
    loads = 0

    async def load_jwks() -> dict[str, Any]:
        nonlocal loads
        loads += 1
        return jwks

    validator = ClaudeMCPBearerTokenValidator.pyjwt(
        jwks_loader=load_jwks,
        issuer="https://auth.example.com",
        audience=("primary-api", "sgp-api"),
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
    )

    first = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )
    second = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )

    assert first.allowed is True
    assert first.principal == "user-123"
    assert first.audience == "sgp-api"
    assert first.resource == "https://example.com/mcp"
    assert first.scopes == ("tools.call", "tools.read")
    assert second.allowed is True
    assert loads == 1


@pytest.mark.parametrize(
    ("case", "token_kwargs"),
    [
        ("wrong_typ", {"header_overrides": {"typ": "JWT"}}),
        ("wrong_issuer", {"claim_overrides": {"iss": "https://other.example.com"}}),
        ("wrong_audience", {"claim_overrides": {"aud": ["other-api"]}}),
        ("missing_audience", {"claim_overrides": {"aud": None}}),
        (
            "wrong_resource",
            {"claim_overrides": {"resource": "https://other.example.com/mcp"}},
        ),
    ],
)
async def test_claude_mcp_pyjwt_authorizer_rejects_invalid_signed_tokens_with_401(
    case: str,
    token_kwargs: dict[str, Any],
) -> None:
    token, jwks = _signed_rs256_token(**token_kwargs)
    validator = ClaudeMCPBearerTokenValidator.pyjwt(
        jwks_loader=lambda: jwks,
        issuer="https://auth.example.com",
        audience="https://example.com/mcp",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
    )

    decision = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )

    assert case
    assert decision.allowed is False
    assert decision.status_code == 401
    assert decision.message == "invalid token"
    assert decision.www_authenticate == 'Bearer error="invalid_token"'
    assert token not in decision.message
    assert token not in decision.www_authenticate


async def test_claude_mcp_pyjwt_authorizer_rejects_under_scoped_signed_token_with_403() -> None:
    token, jwks = _signed_rs256_token(claim_overrides={"scope": "tools.read"})
    validator = ClaudeMCPBearerTokenValidator.pyjwt(
        jwks_loader=lambda: jwks,
        issuer="https://auth.example.com",
        audience="https://example.com/mcp",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
    )

    decision = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )

    assert decision.allowed is False
    assert decision.status_code == 403
    assert decision.message == "insufficient scope"
    assert decision.www_authenticate == (
        'Bearer error="insufficient_scope", scope="tools.call"'
    )


async def test_claude_mcp_pyjwt_authorizer_selects_matching_jwks_kid() -> None:
    _private_key_a, jwk_a = _rsa_key_and_jwk("key-a")
    private_key_b, jwk_b = _rsa_key_and_jwk("key-b")
    token = _encode_rs256_token(private_key_b, kid="key-b")
    validator = ClaudeMCPBearerTokenValidator.pyjwt(
        jwks_loader=lambda: {"keys": [jwk_a, jwk_b]},
        issuer="https://auth.example.com",
        audience="sgp-api",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
    )

    decision = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )

    assert decision.allowed is True
    assert decision.principal == "user-123"
    assert decision.audience == "sgp-api"


@pytest.mark.parametrize("kid", ["missing", None])
async def test_claude_mcp_pyjwt_authorizer_rejects_unknown_or_missing_kid(
    kid: str | None,
) -> None:
    _private_key_a, jwk_a = _rsa_key_and_jwk("key-a")
    private_key_b, jwk_b = _rsa_key_and_jwk("key-b")
    token = _encode_rs256_token(private_key_b, kid=kid)
    validator = ClaudeMCPBearerTokenValidator.pyjwt(
        jwks_loader=lambda: {"keys": [jwk_a, jwk_b]},
        issuer="https://auth.example.com",
        audience="sgp-api",
        resource="https://example.com/mcp",
        required_scopes=("tools.call",),
    )

    decision = await validator(
        ClaudeMCPHTTPAuthorizationContext(
            method="POST",
            path="/mcp",
            authorization_scheme="Bearer",
            bearer_token=token,
        )
    )

    assert decision.allowed is False
    assert decision.status_code == 401
    assert decision.www_authenticate == 'Bearer error="invalid_token"'


async def test_claude_mcp_pyjwt_authorizer_redacts_bad_jwks_token_from_http_response() -> None:
    private_key, _jwk = _rsa_key_and_jwk("key-b")
    token = _encode_rs256_token(private_key, kid="key-b")
    app = ClaudeMeshMCPHTTPApp(
        ClaudeMeshMCPAdapter(Mesh()),
        authorize_http=ClaudeMCPBearerTokenValidator.pyjwt(
            jwks_loader=lambda: {"keys": []},
            issuer="https://auth.example.com",
            audience="sgp-api",
            resource="https://example.com/mcp",
            required_scopes=("tools.call",),
        ),
        protected_resource_metadata_url=(
            "https://example.com/.well-known/oauth-protected-resource/mcp"
        ),
    )

    response = await _call_http_app(
        app,
        headers={**POST_HEADERS, "authorization": f"Bearer {token}"},
        body=_json_body({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )

    assert response.status == 401
    assert response.json()["error"]["message"] == "invalid token"
    assert response.headers["www-authenticate"] == (
        'Bearer error="invalid_token", '
        'resource_metadata="https://example.com/.well-known/oauth-protected-resource/mcp"'
    )
    assert token not in response.body.decode("utf-8")
    assert token not in response.headers["www-authenticate"]
