# LLMAgent: Autonomous LLM-backed Agents (Hermes)

- **Date:** 2026-05-24
- **Status:** Design (approved, pending spec review)
- **Repo:** `signalgatingprotocol/python-sdk`
- **Scope:** v1 of the "autonomous agent" direction. Adds a reusable `LLMAgent` that backs an SGP `Agent` with an OpenAI-compatible LLM brain (Hermes being the documented backend).

## Problem

The SDK's `Agent` is "LLM-ready" (`@agent.tool` / `tools_schema()` exist as the bridge for LLM-based agents) but ships **no LLM integration**. All examples are deterministic stubs. The docs now say "an `Agent` is where a runtime like Hermes plugs in," but nothing actually plugs one in. This closes that gap.

## Goal

A reusable, importable `LLMAgent` that, on receiving a signal, calls an OpenAI-compatible chat-completions endpoint and emits a result signal. Autonomous out of the box, composable into a `Mesh` like any other agent. Demonstrated with ≥2 Hermes-backed agents coordinating.

## Non-goals (v1)

- **LLM-driven tool-calling** (an agent's LLM invoking other agents' tools across the mesh). Deferred to v2; the bridge (`tools_schema()`, `mesh.call_tool`) already exists, so it is a clean extension.
- A hard `openai` dependency in the core. It stays an optional extra.
- Streaming, multi-turn memory, retries beyond what the existing Agent supervision/DLQ provides.
- Any change to `Signal`, `Gate`, `Mesh`, or the base `Agent`.

## Public API

```python
from signal_gating.llm import LLMAgent, Message

analyst = LLMAgent(
    "analyst",
    client=client,          # any OpenAI-compatible client (duck-typed; see LLMClient)
    model="hermes-agent",
    system="You are a research analyst.",
    on=Topic,               # input Signal subclass it reacts to   (default: Message)
    emit=Analysis,          # output Signal subclass it produces    (default: Message)
    temperature=0.7,        # optional, forwarded to the completion call
    **agent_kwargs,         # passes through to Agent (gates, priority_inbox, max_restarts, ...)
)
```

### Constructor

`LLMAgent(name, *, client, model, system="", on=Message, emit=Message, temperature=None, render=None, build=None, **agent_kwargs)`

- Subclasses `Agent`; forwards `name` and `**agent_kwargs` to `Agent.__init__`.
- In `__init__`, registers one handler via `self.on(on)` running the reasoning loop. (Dispatch is `isinstance`-based, so `on=` may be a base class to catch subclasses.)

### Convenience constructor

`LLMAgent.from_openai(name, *, base_url, api_key, model, system="", **kwargs) -> LLMAgent`

- Lazily imports `openai` (raises a clear `ImportError` with the `pip install signal-gating[llm]` hint if absent), builds `openai.AsyncOpenAI(base_url=base_url, api_key=api_key)`, and returns a configured `LLMAgent`.
- For Hermes: `from_openai(..., base_url="http://127.0.0.1:8642/v1", api_key="...", model="hermes-agent")`.

### Built-in signal

```python
class Message(Signal):
    text: str = ""
```

The zero-config input/output type, so two agents can be wired without defining domain signals.

### Customization hooks

- `render: Callable[[Signal], str]`: turns the incoming signal into the user prompt. Default: `signal.text` if present, else `str(signal)`.
- `build: Callable[[Signal, str], Signal]`: turns the LLM reply text into the output signal. Default: `input_signal.child()` cast to `emit` type with its text field set, preserving `trace_id`, `priority`, and lineage. (Default `build` requires `emit` to have a `text` field; documented. Supply a custom `build` otherwise.)

### `LLMClient` Protocol

A minimal structural type covering only what is used, so the core stays `openai`-free and mypy-strict:

```python
class LLMClient(Protocol):
    @property
    def chat(self) -> _Chat: ...
# _Chat.completions.create(model=..., messages=..., **kw) -> response
# response.choices[0].message.content -> str | None
```

`openai.AsyncOpenAI` satisfies this structurally.

## Data flow (the loop)

1. A signal of type `on` reaches the agent; the registered handler runs.
2. Build messages: `[{"role": "system", "content": system}, {"role": "user", "content": render(signal)}]`.
3. `resp = await client.chat.completions.create(model=model, messages=messages, **temperature?)`.
4. `text = resp.choices[0].message.content`.
5. `await self.emit(build(signal, text))`.

Output signals are children of the input (`signal.child(...)`), so trace lineage and priority propagate through the mesh.

## Error handling

- If the completion call raises, the exception propagates out of the handler into the **existing** Agent error path (DLQ + supervised restart). No new error machinery.
- If `resp.choices[0].message.content` is `None`/empty, raise a `SignalValidationError`-style error so the signal is dead-lettered with a clear reason rather than emitting an empty result.

## Testing

Unit tests (`tests/test_llm_agent.py`), **no live server, CI-safe**:

- **FakeClient** implementing `LLMClient`, returning a canned completion object (`choices[0].message.content`).
- `test_emits_response_text`: input signal produces output signal of type `emit` carrying the canned text.
- `test_lineage_preserved`: output `trace_id == input.trace_id`, `parent_id == input.id`.
- `test_render_and_build_customization`: custom `render`/`build` are honored.
- `test_input_type_filtering`: only `on`-type signals trigger the loop.
- `test_empty_response_dead_letters`: `None` content routes to the DLQ.
- `test_from_openai_missing_dependency`: monkeypatch import to assert the clear `ImportError` hint (and a happy-path build with a stubbed `openai` module).

`examples/hermes_mesh.py` is **not** run in CI (needs a live Hermes server); it is smoke-documented in the README.

## Files

| File | Change |
| --- | --- |
| `src/signal_gating/llm.py` | New: `Message`, `LLMClient` Protocol, `LLMAgent` (+ `from_openai`). No top-level `openai` import. |
| `src/signal_gating/__init__.py` | Export `LLMAgent`, `Message`; add to `__all__`. |
| `tests/test_llm_agent.py` | New: unit tests with `FakeClient`. |
| `examples/hermes_mesh.py` | New: planner (`Topic`→`Plan`) → writer (`Plan`→`Draft`) `LLMAgent`s in a `Mesh`, backed by Hermes via `from_openai`. |
| `pyproject.toml` | Add `[project.optional-dependencies] llm = ["openai>=1.0"]`. |
| `README.md` | New "LLM-backed agents (Hermes)" section with the quickstart and a run note. |

Docs-site guide page (in `signalgatingprotocol.github.io`) is a follow-on, separate from this SDK change.

## Example sketch (`examples/hermes_mesh.py`)

```python
import asyncio
from signal_gating import Agent, Mesh, Signal
from signal_gating.llm import LLMAgent

class Topic(Signal):
    text: str = ""
class Plan(Signal):
    text: str = ""
class Draft(Signal):
    text: str = ""

HERMES = dict(base_url="http://127.0.0.1:8642/v1", api_key="change-me-local-dev", model="hermes-agent")

planner = LLMAgent.from_openai("planner", system="Break the topic into a 3-bullet outline.",
                               on=Topic, emit=Plan, **HERMES)
writer  = LLMAgent.from_openai("writer", system="Write a tight paragraph from the outline.",
                               on=Plan, emit=Draft, **HERMES)

# Plain (non-LLM) sink so the result is visible.
reporter = Agent("reporter")

@reporter.on(Draft)
async def show(signal: Draft):
    print(signal.text)

async def main():
    mesh = Mesh([planner, writer, reporter])
    mesh.connect(planner, writer)
    mesh.connect(writer, reporter)
    async with mesh:
        await planner.emit(Topic(text="why signal gating matters for multi-agent systems"))
        await asyncio.sleep(2.0)

asyncio.run(main())
```

## Success criteria

1. `from signal_gating.llm import LLMAgent, Message` works; `import signal_gating` triggers no `openai` import.
2. `pytest` passes with all listed tests, no live server, with `openai` **not** installed.
3. `ruff check .` and `mypy src/` (strict) pass.
4. `examples/hermes_mesh.py` runs end-to-end against a local Hermes server (manual).
5. README documents the Hermes quickstart and the optional extra.
