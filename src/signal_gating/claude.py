"""Claude Agent SDK integration boundary.

This module keeps Claude Agent SDK sessions as an external runtime while making
their prompts, results, tool events, and permission decisions visible as typed
SGP signals. Core users do not pay the optional dependency cost unless they
instantiate ``ClaudeAgent`` without an injected query function.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from signal_gating.agent import Agent
from signal_gating.errors import AgentError
from signal_gating.signal import Signal

ClaudeToolEventKind = Literal["tool_use", "tool_result"]
ClaudePermissionDecision = Literal["allowed", "denied", "prompted", "unknown"]
Render = Callable[[Signal], str]


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


@dataclass(slots=True)
class ClaudeAgentSDKResult:
    """Summary returned by :class:`ClaudeAgentSDKRunner`."""

    session_id: str
    result: str
    subtype: str = ""
    total_cost_usd: float | None = None
    message_count: int = 0


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
        self._allowed_tools = list(allowed_tools or [])
        self._disallowed_tools = list(disallowed_tools or [])
        self._permission_mode = permission_mode
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
    kwargs: dict[str, Any] = {}
    if allowed_tools is not None:
        kwargs["allowed_tools"] = list(allowed_tools)
    if disallowed_tools is not None:
        kwargs["disallowed_tools"] = list(disallowed_tools)
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
        allowed = list(allowed_tools or [])
        disallowed = list(disallowed_tools or [])
        mcp_config = dict(mcp_servers or {})
        options_kwargs = _runner_options_kwargs(
            allowed_tools=allowed,
            disallowed_tools=disallowed,
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


class ClaudeAgentSDKSession:
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
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        permission_mode: str = "",
        mcp_servers: Mapping[str, Any] | None = None,
        cwd: str | Path | None = None,
        session_store: Any | None = None,
        strict_mcp_config: bool | None = None,
        max_turns: int | None = None,
        model: str = "",
        system_prompt: str = "",
    ) -> None:
        self._client_factory = client_factory
        self._options_factory = options_factory
        self._client: Any | None = None
        self._connected = False
        self._allowed_tools = list(allowed_tools or [])
        self._disallowed_tools = list(disallowed_tools or [])
        self._permission_mode = permission_mode
        self._mcp_servers = dict(mcp_servers or {})
        self._cwd = cwd
        self._options_kwargs = _runner_options_kwargs(
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            permission_mode=permission_mode,
            mcp_servers=self._mcp_servers,
            cwd=cwd,
            session_store=session_store,
            strict_mcp_config=strict_mcp_config,
            max_turns=max_turns,
            model=model,
            system_prompt=system_prompt,
        )

    async def __aenter__(self) -> ClaudeAgentSDKSession:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        await self.disconnect()

    async def connect(self, prompt: str | None = None) -> None:
        """Connect the underlying Claude SDK client once."""
        if self._connected:
            return
        if self._client is None:
            client_factory, options_factory = self._sdk_bindings()
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
                    "ClaudeAgentSDKSession",
                    "ClaudeSDKClient exposes neither connect() nor __aenter__()",
                )
            self._client = await _maybe_await(enter())
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect the underlying Claude SDK client if it is connected."""
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
        await self.connect()
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
            mcp_servers={name: {} for name in self._mcp_servers},
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
            mcp_server_names=sorted(self._mcp_servers),
            cwd=str(self._cwd or ""),
        )

        await _maybe_await(client.query(prompt, session_id=session_id))

        result = await _drain_client_response(
            client,
            mesh=mesh,
            run_signal=run_signal,
            allowed=self._allowed_tools,
            disallowed=self._disallowed_tools,
            mcp_servers=self._mcp_servers,
            resumed_from_session_id="",
            continued=True,
        )
        return result

    async def ask(self, prompt: str, **kwargs: Any) -> ClaudeAgentSDKResult:
        """Alias for ``query`` that reads naturally in chat-like code."""
        return await self.query(prompt, **kwargs)

    async def set_permission_mode(
        self,
        mode: str,
        *,
        mesh: Any | None = None,
    ) -> None:
        """Change Claude's permission mode for subsequent tool requests."""
        await self.connect()
        client = self._require_client()
        setter = getattr(client, "set_permission_mode", None)
        if setter is None:
            raise AgentError(
                "ClaudeAgentSDKSession",
                "ClaudeSDKClient does not expose set_permission_mode()",
            )
        await _maybe_await(setter(mode))
        signal = ClaudePermissionDecisionSignal(
            tool_name="*",
            decision="unknown",
            permission_mode=mode,
            reason="permission mode changed for continuous Claude session",
        )
        await _record_claude_event(
            mesh,
            "claude_permission_mode_set",
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
        await self.connect()
        client = self._require_client()
        setter = getattr(client, "set_model", None)
        if setter is None:
            raise AgentError(
                "ClaudeAgentSDKSession",
                "ClaudeSDKClient does not expose set_model()",
            )
        await _maybe_await(setter(model))
        await _record_claude_event(
            mesh,
            "claude_model_set",
            ClaudeAgentRunSignal(prompt="", cwd=str(self._cwd or "")),
            model=model,
        )

    def _require_client(self) -> Any:
        if self._client is None:
            raise AgentError("ClaudeAgentSDKSession", "ClaudeSDKClient is not connected")
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
            "ClaudeAgentSDKSession",
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
    "ClaudeAgentSDKRunner",
    "ClaudeAgentSDKSession",
    "ClaudeClientFactory",
    "ClaudeAgentResultSignal",
    "ClaudeAgentRunSignal",
    "ClaudePermissionDecision",
    "ClaudePermissionDecisionSignal",
    "ClaudeQueryFn",
    "ClaudeToolEventKind",
    "ClaudeToolEventSignal",
    "claude_options",
    "mcp_tool_name",
]
