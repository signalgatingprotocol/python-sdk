"""LLM-backed agents: give an SGP Agent an OpenAI-compatible brain (e.g. Hermes)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
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


_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _json_type(t: Any) -> str:
    return _JSON_TYPES.get(str(t), "string")


class ToolProvider(Protocol):
    """Supplies tools to an LLMAgent: OpenAI-format schemas plus an async invoker."""

    def tool_schemas(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class MeshToolProvider:
    """Exposes a mesh's agent tools to an LLMAgent in OpenAI function-tool format."""

    def __init__(self, mesh: Mesh) -> None:
        self._mesh = mesh

    def _index(self) -> dict[str, tuple[str, ToolSpec]]:
        index: dict[str, tuple[str, ToolSpec]] = {}
        for owner, specs in self._mesh.discover_tools().items():
            for spec in specs:
                if spec.name in index:
                    raise ValueError(
                        f"MeshToolProvider: duplicate tool name {spec.name!r} "
                        f"(agents {index[spec.name][0]!r} and {owner!r}); "
                        "tool names must be unique across the mesh."
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
                                p: {"type": _json_type(meta.get("type"))}
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
            raise AgentError("MeshToolProvider", f"unknown tool {name!r}")
        owner, _spec = index[name]
        return await self._mesh.call_tool(owner, name, **arguments)


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
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(name, **agent_kwargs)
        self._client = client
        self._model = model
        self._system = system
        self._emit_type: type[Any] = emit
        self._temperature = temperature
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
            resp = await self._client.chat.completions.create(
                model=self._model, messages=messages, **call_kwargs
            )
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
