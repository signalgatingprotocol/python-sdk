import builtins
import subprocess
import sys
import types

import pytest

from signal_gating import Signal
from signal_gating.errors import AgentError
from signal_gating.llm import LLMAgent, Message, _default_render


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
    def __init__(self, content: str | None = "ok"):
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
        build=lambda _, t: Plan(text=f"BUILT:{t}"),
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


async def test_input_type_filtering():
    client = FakeClient("r")
    agent = LLMAgent("a", client=client, model="m", on=Topic, emit=Plan)
    _capture(agent)

    class Other(Signal):
        text: str = ""

    await agent._dispatch(Other(text="ignored"))
    assert client.calls == []  # a non-Topic signal never reaches the LLM


async def test_none_content_raises_agent_error():
    agent = LLMAgent("a", client=FakeClient(None), model="m", on=Topic, emit=Plan)
    _capture(agent)
    with pytest.raises(AgentError):
        await agent._dispatch(Topic(text="q"))
