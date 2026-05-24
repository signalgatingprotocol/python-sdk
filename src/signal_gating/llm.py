"""LLM-backed agents — give an SGP Agent an OpenAI-compatible brain (e.g. Hermes)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from signal_gating.agent import Agent
from signal_gating.errors import AgentError
from signal_gating.signal import Signal


class Message(Signal):
    """A generic text-carrying signal — the zero-config input/output for LLMAgent."""

    text: str = ""


class _ChatMessage(Protocol):
    content: str | None


class _Choice(Protocol):
    message: _ChatMessage


class _Completion(Protocol):
    choices: Sequence[_Choice]


class _Completions(Protocol):
    async def create(
        self, *, model: str, messages: list[dict[str, str]], **kwargs: Any
    ) -> _Completion: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions: ...


class LLMClient(Protocol):
    """Minimal structural type for an OpenAI-compatible async client."""

    @property
    def chat(self) -> _Chat: ...


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
        self.on(on)(self._handle)

    async def _handle(self, signal: Signal) -> None:
        messages: list[dict[str, str]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": self._render(signal)})

        extra: dict[str, Any] = {}
        if self._temperature is not None:
            extra["temperature"] = self._temperature

        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, **extra
        )
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise AgentError(
                self.name, f"LLM returned empty content for {type(signal).__name__}"
            )
        await self.emit(self._build(signal, content))

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
