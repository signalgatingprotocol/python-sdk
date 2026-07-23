"""LLM-backed agents: give an SGP Agent an OpenAI-compatible brain (e.g. Hermes)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol, cast

from signal_gating.agent import Agent, ToolSpec
from signal_gating.errors import AgentError
from signal_gating.mesh import Mesh
from signal_gating.signal import Signal


class Message(Signal):
    """A generic text-carrying signal, the zero-config input/output for LLMAgent."""

    text: str = ""


class _ToolFunction(Protocol):
    name: str
    arguments: str


class _ToolCall(Protocol):
    id: str
    function: _ToolFunction


class _ChatMessage(Protocol):
    content: str | None
    tool_calls: list[_ToolCall] | None


class _Choice(Protocol):
    message: _ChatMessage


class _Completion(Protocol):
    choices: Sequence[_Choice]


class _Completions(Protocol):
    async def create(
        self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> _Completion: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions: ...


class LLMClient(Protocol):
    """Minimal structural type for an OpenAI-compatible async client."""

    @property
    def chat(self) -> _Chat: ...


_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
}


def _param_schema(t: Any) -> dict[str, Any]:
    """JSON-Schema fragment for a tool parameter type.

    Containers must not collapse to ``"string"`` -- that tells the model to
    pass a string where the tool wants an array/object. Arrays also need an
    ``items`` schema for strict OpenAI-compatible servers.
    """
    json_type = _JSON_TYPES.get(str(t), "string")
    if json_type == "array":
        return {"type": "array", "items": {}}
    return {"type": json_type}


class ToolProvider(Protocol):
    """Supplies tools to an LLMAgent: OpenAI-format schemas plus an async invoker."""

    def tool_schemas(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class MeshToolProvider:
    """Expose explicitly authorized mesh tools to an LLMAgent.

    Model-selected tool calls cross a security boundary. Pass ``allow`` to
    expose only named tools, or opt a trusted internal mesh into unrestricted
    discovery with ``allow_all=True``. Omitting both policies fails closed.
    """

    def __init__(
        self,
        mesh: Mesh,
        *,
        allow: Mapping[str, Collection[str]] | None = None,
        allow_all: bool = False,
    ) -> None:
        self._mesh = mesh
        if allow is None and not allow_all:
            raise ValueError(
                "MeshToolProvider requires an explicit allow mapping; "
                "use allow={} to expose no tools or allow_all=True only for a trusted mesh."
            )
        if allow is not None and allow_all:
            raise ValueError(
                "MeshToolProvider allow cannot be combined with allow_all=True; "
                "choose one exposure policy."
            )

        self._allow = None if allow is None else self._normalize_allow(allow)
        self._authorized_bindings: dict[
            tuple[str, str], tuple[ToolSpec, object | None]
        ] = {}
        if self._allow is not None:
            self._authorized_bindings = self._validate_allowlist()
        # Fail early when the configured public surface is ambiguous.
        self._index()

    @staticmethod
    def _normalize_allow(
        allow: Mapping[str, Collection[str]],
    ) -> dict[str, frozenset[str]]:
        if not isinstance(allow, Mapping):
            raise TypeError("MeshToolProvider allow must be a mapping of agent to tool names")

        normalized: dict[str, frozenset[str]] = {}
        for owner, names in allow.items():
            if not isinstance(owner, str):
                raise TypeError("MeshToolProvider allow agent names must be strings")
            if isinstance(names, (str, bytes)):
                raise TypeError(
                    f"MeshToolProvider allow[{owner!r}] must be a collection of tool names, "
                    "not a string"
                )
            if not isinstance(names, Collection):
                raise TypeError(
                    f"MeshToolProvider allow[{owner!r}] must be a collection of tool names"
                )
            if any(not isinstance(name, str) for name in names):
                raise TypeError(
                    f"MeshToolProvider allow[{owner!r}] tool names must be strings"
                )
            normalized[owner] = frozenset(names)
        return normalized

    def _validate_allowlist(
        self,
    ) -> dict[tuple[str, str], tuple[ToolSpec, object | None]]:
        assert self._allow is not None
        agents = {agent.name: agent for agent in self._mesh.agents}
        authorized: dict[tuple[str, str], tuple[ToolSpec, object | None]] = {}
        for owner, allowed_names in self._allow.items():
            agent = agents.get(owner)
            if agent is None:
                raise ValueError(
                    f"MeshToolProvider allowlist references unknown agent {owner!r}; "
                    "add the agent to the mesh before constructing the provider."
                )
            available = {spec.name: spec for spec in agent.list_tools()}
            unknown_names = sorted(allowed_names - available.keys())
            if unknown_names:
                noun = "tool" if len(unknown_names) == 1 else "tools"
                joined = ", ".join(repr(name) for name in unknown_names)
                raise ValueError(
                    f"MeshToolProvider allowlist references unknown {noun} {joined} "
                    f"for agent {owner!r}; register it before constructing the provider."
                )
            for name in allowed_names:
                spec = available[name]
                authorized[(owner, name)] = (spec, spec.fn)
        return authorized

    def _is_allowed(self, owner: str, name: str) -> bool:
        return self._allow is None or name in self._allow.get(owner, ())

    def _index(self) -> dict[str, tuple[str, ToolSpec]]:
        index: dict[str, tuple[str, ToolSpec]] = {}
        for owner, specs in self._mesh.discover_tools().items():
            for spec in specs:
                if not self._is_allowed(owner, spec.name):
                    continue
                if self._allow is not None:
                    expected_spec, expected_fn = self._authorized_bindings[
                        (owner, spec.name)
                    ]
                    if spec is not expected_spec or spec.fn is not expected_fn:
                        raise AgentError(
                            "MeshToolProvider",
                            f"authorized tool binding {owner!r}.{spec.name!r} changed "
                            "after provider construction",
                        )
                if spec.name in index:
                    raise ValueError(
                        f"MeshToolProvider: duplicate exposed tool name {spec.name!r} "
                        f"(agents {index[spec.name][0]!r} and {owner!r}); "
                        "exposed tool names must be unique."
                    )
                index[spec.name] = (owner, spec)
        return index

    def tool_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for name, (_owner, spec) in self._index().items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                p: _param_schema(meta.get("type"))
                                for p, meta in spec.parameters.items()
                            },
                            "required": [
                                p for p, meta in spec.parameters.items() if meta.get("required")
                            ],
                        },
                    },
                }
            )
        return schemas

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        index = self._index()
        if name not in index:
            if self._allow is None:
                raise AgentError("MeshToolProvider", f"unknown tool {name!r}")
            if not any(name in allowed_names for allowed_names in self._allow.values()):
                raise AgentError(
                    "MeshToolProvider", f"tool {name!r} is not allowed by this provider"
                )
            raise AgentError(
                "MeshToolProvider", f"allowed tool {name!r} is no longer available"
            )
        owner, spec = index[name]
        return await self._mesh._call_tool(
            owner,
            name,
            arguments,
            expected_binding_id=spec.binding_id,
        )


Render = Callable[[Signal], str]
Build = Callable[[Signal, str], Signal]


def _default_render(signal: Signal) -> str:
    text = getattr(signal, "text", "")
    return str(text) if text else repr(signal)


class LLMAgent(Agent):
    """An Agent whose handler reasons via an OpenAI-compatible LLM (e.g. Hermes).

    On each `on`-typed signal it calls the model and emits an `emit`-typed signal
    built from the reply, preserving trace lineage.
    """

    def __init__(
        self,
        name: str,
        *,
        client: LLMClient,
        model: str,
        system: str = "",
        on: type[Signal] = Message,
        emit: type[Signal] = Message,
        temperature: float | None = None,
        render: Render | None = None,
        build: Build | None = None,
        tools: ToolProvider | None = None,
        max_tool_rounds: int = 4,
        timeout: float | None = None,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(name, **agent_kwargs)
        self._client = client
        self._model = model
        self._system = system
        self._emit_type: type[Any] = emit
        self._temperature = temperature
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._timeout = timeout
        self._render: Render = render or _default_render
        if build is None and "text" not in emit.model_fields:
            raise ValueError(
                f"LLMAgent({name!r}): emit type {emit.__name__} has no 'text' field; "
                "pass build=... to construct the output signal yourself."
            )
        self._build: Build = build or self._default_build
        self._tool_provider = tools
        self._max_tool_rounds = max_tool_rounds
        self.on(on)(self._handle)

    async def _handle(self, signal: Signal) -> None:
        messages: list[dict[str, Any]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": self._render(signal)})

        extra: dict[str, Any] = {}
        if self._temperature is not None:
            extra["temperature"] = self._temperature

        schemas = self._tool_provider.tool_schemas() if self._tool_provider is not None else None
        rounds = self._max_tool_rounds if self._tool_provider is not None else 1

        for _ in range(rounds):
            call_kwargs = dict(extra)
            if schemas is not None:
                call_kwargs["tools"] = schemas
            completion = self._client.chat.completions.create(
                model=self._model, messages=messages, **call_kwargs
            )
            if self._timeout is not None:
                try:
                    resp = await asyncio.wait_for(completion, timeout=self._timeout)
                except asyncio.TimeoutError:
                    raise AgentError(
                        self.name, f"LLM request timed out after {self._timeout}s"
                    ) from None
            else:
                resp = await completion
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls if schemas is not None else None

            if tool_calls:
                assert self._tool_provider is not None  # narrow for type-checkers
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        raise AgentError(
                            self.name,
                            f"invalid JSON arguments for tool {tc.function.name!r}: {e.msg}",
                        ) from e
                    if not isinstance(args, dict):
                        raise AgentError(
                            self.name,
                            f"tool {tc.function.name!r} arguments must decode to an object",
                        )
                    result = await self._tool_provider.call_tool(tc.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                continue

            content = msg.content
            if not content or not content.strip():
                raise AgentError(
                    self.name, f"LLM returned empty content for {type(signal).__name__}"
                )
            await self.emit(self._build(signal, content))
            return

        raise AgentError(
            self.name, f"exceeded max_tool_rounds ({self._max_tool_rounds})"
        )

    def _default_build(self, signal: Signal, text: str) -> Signal:
        return cast(
            Signal,
            self._emit_type(
                text=text,
                source=self.name,
                trace_id=signal.trace_id,
                parent_id=signal.id,
                priority=signal.priority,
            ),
        )

    @classmethod
    def from_openai(
        cls,
        name: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **kwargs: Any,
    ) -> LLMAgent:
        """Build an LLMAgent backed by an OpenAI-compatible endpoint (e.g. Hermes).

        Lazily imports `openai`; install it with `pip install signal-gating[llm]`.
        """
        try:
            import openai
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "LLMAgent.from_openai requires the 'openai' package. "
                "Install it with: pip install signal-gating[llm]"
            ) from e

        client = cast(LLMClient, openai.AsyncOpenAI(base_url=base_url, api_key=api_key))
        return cls(name, client=client, model=model, **kwargs)
