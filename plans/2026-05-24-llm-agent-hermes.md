# LLMAgent (Hermes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `LLMAgent` that backs an SGP `Agent` with an OpenAI-compatible LLM brain (Hermes), turning the existing "LLM-ready" `Agent` into one that actually reasons.

**Architecture:** `LLMAgent(Agent)` registers one handler (for an `on=` signal type) that calls an injected, duck-typed `LLMClient`, then emits an `emit=` signal built from the reply. The core never imports `openai`; `from_openai` builds the client lazily, and `openai` is an optional `[llm]` extra. Outputs preserve trace lineage; LLM/empty-response errors ride the Agent's existing dead-letter path.

**Tech Stack:** Python 3.10+, pydantic v2 (`Signal`), pytest + pytest-asyncio (auto mode), ruff, mypy strict.

**Spec:** `specs/2026-05-24-llm-agent-hermes-design.md`

---

## Conventions

- **Commits:** Per the project's CLAUDE.md, commit **only when the user authorizes**. Commit steps below are ready-to-run; treat them as "stage + commit once authorized." Branch is `claude/llm-agent-hermes` (not default).
- **Working dir:** `/Users/p/code/github/signalgatingprotocol/python-sdk`.
- **Test runner:** `pytest` (asyncio auto mode — `async def test_*` needs no decorator).

## File structure

| File | Responsibility |
| --- | --- |
| `src/signal_gating/llm.py` | `Message`, `LLMClient` Protocol, `_default_render`, `LLMAgent` (+ `from_openai`). No top-level `openai` import. |
| `src/signal_gating/__init__.py` | Export `LLMAgent`, `Message`. |
| `tests/test_llm_agent.py` | Unit tests with a `FakeClient`. |
| `examples/hermes_mesh.py` | Planner→writer→reporter demo, Hermes-backed. |
| `pyproject.toml` | `[llm]` extra + mypy override for `openai`. |
| `README.md` | "LLM-backed agents (Hermes)" section. |

---

## Task 1: Module scaffold — `Message`, `LLMClient` Protocol, default render

**Files:**
- Create: `src/signal_gating/llm.py`
- Test: `tests/test_llm_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_agent.py`:
```python
from signal_gating.llm import Message, _default_render
from signal_gating import Signal


def test_message_carries_text():
    assert Message(text="hello").text == "hello"
    assert Message().text == ""


def test_default_render_prefers_text():
    assert _default_render(Message(text="hi")) == "hi"


def test_default_render_falls_back_to_repr():
    class Bare(Signal):
        pass
    out = _default_render(Bare())
    assert "Bare" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'signal_gating.llm'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/signal_gating/llm.py`:
```python
"""LLM-backed agents — give an SGP Agent an OpenAI-compatible brain (e.g. Hermes)."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_agent.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/llm.py tests/test_llm_agent.py
git commit -m "feat(llm): add Message signal and LLMClient protocol scaffold"
```

---

## Task 2: `LLMAgent` — constructor, reasoning loop, default build

**Files:**
- Modify: `src/signal_gating/llm.py`
- Test: `tests/test_llm_agent.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_llm_agent.py`:
```python
import pytest

from signal_gating import Signal
from signal_gating.errors import AgentError
from signal_gating.llm import LLMAgent, Message


class Topic(Signal):
    text: str = ""


class Plan(Signal):
    text: str = ""


class _FakeMsg:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.message = _FakeMsg(content)


class _FakeCompletion:
    def __init__(self, content): self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content, calls): self._content, self.calls = content, calls
    async def create(self, *, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, **kw})
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content, calls): self.completions = _FakeCompletions(content, calls)


class FakeClient:
    def __init__(self, content="ok"):
        self.calls = []
        self.chat = _FakeChat(content, self.calls)


def _capture(agent):
    emitted = []
    async def capture(sig): emitted.append(sig)
    agent.emit = capture  # type: ignore[method-assign]
    return emitted


async def test_emits_response_as_emit_type():
    agent = LLMAgent("a", client=FakeClient("the answer"), model="m", on=Topic, emit=Plan)
    emitted = _capture(agent)
    await agent._dispatch(Topic(text="q"))
    assert isinstance(emitted[0], Plan)
    assert emitted[0].text == "the answer"


async def test_lineage_preserved():
    agent = LLMAgent("a", client=FakeClient("r"), model="m", on=Topic, emit=Plan)
    emitted = _capture(agent)
    inp = Topic(text="q")
    await agent._dispatch(inp)
    assert emitted[0].trace_id == inp.trace_id
    assert emitted[0].parent_id == inp.id


async def test_system_and_prompt_sent():
    client = FakeClient("r")
    agent = LLMAgent("a", client=client, model="m", system="be terse", on=Topic, emit=Plan)
    _capture(agent)
    await agent._dispatch(Topic(text="hello"))
    msgs = client.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert client.calls[0]["model"] == "m"


async def test_no_system_message_when_empty():
    client = FakeClient("r")
    agent = LLMAgent("a", client=client, model="m", on=Topic, emit=Plan)
    _capture(agent)
    await agent._dispatch(Topic(text="hi"))
    assert all(m["role"] != "system" for m in client.calls[0]["messages"])


async def test_temperature_forwarded_only_when_set():
    client = FakeClient("r")
    agent = LLMAgent("a", client=client, model="m", on=Topic, emit=Plan, temperature=0.5)
    _capture(agent)
    await agent._dispatch(Topic(text="hi"))
    assert client.calls[0]["temperature"] == 0.5


async def test_render_and_build_customization():
    client = FakeClient("R")
    agent = LLMAgent(
        "a", client=client, model="m", on=Topic, emit=Plan,
        render=lambda s: f"PROMPT:{s.text}",
        build=lambda s, t: Plan(text=f"BUILT:{t}"),
    )
    emitted = _capture(agent)
    await agent._dispatch(Topic(text="x"))
    assert client.calls[0]["messages"][-1]["content"] == "PROMPT:x"
    assert emitted[0].text == "BUILT:R"


async def test_empty_content_raises_agent_error():
    agent = LLMAgent("a", client=FakeClient("   "), model="m", on=Topic, emit=Plan)
    _capture(agent)
    with pytest.raises(AgentError):
        await agent._dispatch(Topic(text="q"))


def test_emit_without_text_field_rejected_at_construction():
    class NoText(Signal):
        pass
    with pytest.raises(ValueError, match="no 'text' field"):
        LLMAgent("a", client=FakeClient(), model="m", on=Topic, emit=NoText)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_agent.py -q`
Expected: FAIL — `ImportError: cannot import name 'LLMAgent'`.

- [ ] **Step 3: Implement `LLMAgent`**

Append to `src/signal_gating/llm.py`:
```python
from signal_gating.agent import Agent
from signal_gating.errors import AgentError


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
        self._emit_type = emit
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
        return self._emit_type(
            text=text,
            source=self.name,
            trace_id=signal.trace_id,
            parent_id=signal.id,
            priority=signal.priority,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_agent.py -q`
Expected: PASS (all tests from Tasks 1–2 green).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/llm.py tests/test_llm_agent.py
git commit -m "feat(llm): add LLMAgent reasoning loop with lineage-preserving emit"
```

---

## Task 3: `LLMAgent.from_openai` — lazy client construction

**Files:**
- Modify: `src/signal_gating/llm.py`
- Test: `tests/test_llm_agent.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_llm_agent.py`:
```python
import builtins
import sys
import types


def test_from_openai_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"signal-gating\[llm\]"):
        LLMAgent.from_openai("a", base_url="x", api_key="y", model="m")


def test_from_openai_builds_agent(monkeypatch):
    mod = types.ModuleType("openai")

    class AsyncOpenAI:
        def __init__(self, **kw):
            self.kw = kw

    mod.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)

    agent = LLMAgent.from_openai(
        "a", base_url="b", api_key="k", model="m", on=Topic, emit=Plan
    )
    assert isinstance(agent, LLMAgent)
    assert agent._model == "m"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_agent.py -q -k from_openai`
Expected: FAIL — `AttributeError: type object 'LLMAgent' has no attribute 'from_openai'`.

- [ ] **Step 3: Implement `from_openai`**

Add this method to the `LLMAgent` class in `src/signal_gating/llm.py` (after `_default_build`):
```python
    @classmethod
    def from_openai(
        cls,
        name: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **kwargs: Any,
    ) -> "LLMAgent":
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
        from typing import cast

        client = cast(LLMClient, openai.AsyncOpenAI(base_url=base_url, api_key=api_key))
        return cls(name, client=client, model=model, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_agent.py -q -k from_openai`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/llm.py tests/test_llm_agent.py
git commit -m "feat(llm): add LLMAgent.from_openai lazy client constructor"
```

---

## Task 4: Export from the package root + import isolation

**Files:**
- Modify: `src/signal_gating/__init__.py`
- Test: `tests/test_llm_agent.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_llm_agent.py`:
```python
import subprocess


def test_exports_from_package_root():
    import signal_gating
    assert hasattr(signal_gating, "LLMAgent")
    assert hasattr(signal_gating, "Message")
    assert "LLMAgent" in signal_gating.__all__
    assert "Message" in signal_gating.__all__


def test_import_does_not_pull_openai():
    code = "import sys, signal_gating; assert 'openai' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_agent.py -q -k "exports or pull_openai"`
Expected: FAIL — `AssertionError` (no `LLMAgent` attribute / not in `__all__`).

- [ ] **Step 3: Add the exports**

In `src/signal_gating/__init__.py`, add an import line (after the existing `from signal_gating.signal import Signal` line):
```python
from signal_gating.llm import LLMAgent, Message
```
Then add `"LLMAgent"` and `"Message"` to the `__all__` list (keep it alphabetically ordered — insert `"LLMAgent"` after `"GateRejected"` and `"Message"` after `"MeshError"`):
```python
    "LLMAgent",
```
```python
    "Message",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_agent.py -q`
Expected: PASS (entire file green).

- [ ] **Step 5: Commit** (when authorized)

```bash
git add src/signal_gating/__init__.py tests/test_llm_agent.py
git commit -m "feat(llm): export LLMAgent and Message from package root"
```

---

## Task 5: `pyproject.toml` — `[llm]` extra + mypy override

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]` (which currently has only `dev`), add:
```toml
llm = [
    "openai>=1.0",
]
```

- [ ] **Step 2: Add a mypy override for the lazy openai import**

The lazy `import openai` would otherwise fail `mypy src/` strict ("cannot find stubs"). Append to the end of `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = "openai.*"
ignore_missing_imports = true
```

- [ ] **Step 3: Verify the file parses and the extra resolves**

Run: `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); assert 'openai>=1.0' in d['project']['optional-dependencies']['llm']; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit** (when authorized)

```bash
git add pyproject.toml
git commit -m "build(llm): add optional [llm] extra and openai mypy override"
```

---

## Task 6: Example — `examples/hermes_mesh.py`

**Files:**
- Create: `examples/hermes_mesh.py`

- [ ] **Step 1: Write the example**

Create `examples/hermes_mesh.py`:
```python
"""Two Hermes-backed agents coordinating through a mesh.

Requires a running Nous Hermes server exposing its OpenAI-compatible API
(default http://127.0.0.1:8642/v1). This example is NOT run in CI.

    python examples/hermes_mesh.py
"""

import asyncio

from signal_gating import Agent, Mesh, Signal
from signal_gating.llm import LLMAgent


class Topic(Signal):
    text: str = ""


class Plan(Signal):
    text: str = ""


class Draft(Signal):
    text: str = ""


HERMES = dict(
    base_url="http://127.0.0.1:8642/v1",
    api_key="change-me-local-dev",
    model="hermes-agent",
)


async def main() -> None:
    planner = LLMAgent.from_openai(
        "planner", system="Break the topic into a 3-bullet outline.",
        on=Topic, emit=Plan, **HERMES,
    )
    writer = LLMAgent.from_openai(
        "writer", system="Write one tight paragraph from the outline.",
        on=Plan, emit=Draft, **HERMES,
    )
    reporter = Agent("reporter")

    @reporter.on(Draft)
    async def show(signal: Draft) -> None:
        print("\n--- draft ---\n" + signal.text)

    mesh = Mesh([planner, writer, reporter])
    mesh.connect(planner, writer)
    mesh.connect(writer, reporter)

    async with mesh:
        await planner.emit(
            Topic(text="why signal gating matters for multi-agent systems")
        )
        await asyncio.sleep(5.0)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify it imports/compiles (does not call the server)**

Run: `python -c "import ast; ast.parse(open('examples/hermes_mesh.py').read()); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit** (when authorized)

```bash
git add examples/hermes_mesh.py
git commit -m "docs(llm): add Hermes multi-agent mesh example"
```

---

## Task 7: README — "LLM-backed agents (Hermes)" section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the section**

In `README.md`, add a new `### LLM-backed agents (Hermes)` subsection at the end of the "Core Primitives" area (immediately before the `## Architecture` heading). Insert:
````markdown
### LLM-backed agents (Hermes)

`LLMAgent` gives an agent an OpenAI-compatible brain — Nous Hermes, or any
OpenAI-compatible server. Install the extra:

```bash
pip install "signal-gating[llm]"
```

```python
from signal_gating import Mesh, Signal
from signal_gating.llm import LLMAgent

class Topic(Signal):
    text: str = ""
class Plan(Signal):
    text: str = ""

planner = LLMAgent.from_openai(
    "planner",
    base_url="http://127.0.0.1:8642/v1",  # Hermes' OpenAI-compatible server
    api_key="change-me-local-dev",
    model="hermes-agent",
    system="Break the topic into a 3-bullet outline.",
    on=Topic, emit=Plan,
)
```

`LLMAgent` is a normal `Agent`: gate it, connect it in a `Mesh`, and coordinate
several of them with `scatter` / `map_reduce` / `workflow`. See
`examples/hermes_mesh.py` for two Hermes agents coordinating end-to-end.
````

- [ ] **Step 2: Verify the section is present**

Run: `grep -n "LLM-backed agents (Hermes)" README.md`
Expected: one match.

- [ ] **Step 3: Commit** (when authorized)

```bash
git add README.md
git commit -m "docs(llm): document LLM-backed agents in README"
```

---

## Task 8: Full verification gate

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `ruff check .`
Expected: no errors. (If imports in the test file are flagged for ordering, run `ruff check --fix .` and re-commit the test file.)

- [ ] **Step 2: Type-check (strict)**

Run: `mypy src/`
Expected: `Success: no issues found`. (The `openai.*` override from Task 5 keeps the lazy import clean.)

- [ ] **Step 3: Tests pass without `openai` installed**

Run: `pip uninstall -y openai >/dev/null 2>&1; pytest -q`
Expected: full suite PASS (the existing suite plus `tests/test_llm_agent.py`); no test requires a live server or the `openai` package.

- [ ] **Step 4: Import isolation holds**

Run: `python -c "import sys, signal_gating; assert 'openai' not in sys.modules; print('clean')"`
Expected: prints `clean`.

- [ ] **Step 5: Commit any lint fixups** (when authorized)

```bash
git add -A
git commit -m "chore(llm): lint/type fixups"
```

---

## Done criteria (maps to spec success criteria)

1. `from signal_gating.llm import LLMAgent, Message` works; `import signal_gating` pulls no `openai` (Tasks 1, 4; Task 8 Step 4).
2. `pytest` green with `openai` absent, no live server (Task 8 Step 3).
3. `ruff check .` and `mypy src/` strict pass (Task 8 Steps 1–2).
4. `examples/hermes_mesh.py` parses and is runnable against a local Hermes server (Task 6).
5. README documents the Hermes quickstart and the `[llm]` extra (Task 7).
