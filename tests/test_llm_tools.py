import asyncio

import pytest

from signal_gating import Agent, Mesh, Signal
from signal_gating.errors import AgentError
from signal_gating.llm import LLMAgent, MeshToolProvider

# --- Task 1: MeshToolProvider tests ---


async def test_mesh_tool_provider_schema_and_routing():
    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a topic")
    async def analyze(topic: str) -> dict:
        return {"points": topic.upper()}

    mesh = Mesh([analyst])
    provider = MeshToolProvider(mesh)

    schemas = provider.tool_schemas()
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert schemas[0]["type"] == "function"
    assert fn["name"] == "analyze"
    assert fn["parameters"]["properties"]["topic"]["type"] == "string"
    assert fn["parameters"]["required"] == ["topic"]

    async with mesh:
        result = await provider.call_tool("analyze", {"topic": "hi"})
    assert result == {"points": "HI"}


def test_mesh_tool_provider_duplicate_name_raises():
    a = Agent("a")
    b = Agent("b")

    @a.tool(name="dup")
    async def t1() -> int:
        return 1

    @b.tool(name="dup")
    async def t2() -> int:
        return 2

    mesh = Mesh([a, b])
    with pytest.raises(ValueError, match="duplicate tool name"):
        MeshToolProvider(mesh).tool_schemas()


async def test_mesh_tool_provider_unknown_tool_raises():
    mesh = Mesh([Agent("x")])
    with pytest.raises(AgentError, match="unknown tool"):
        async with mesh:
            await MeshToolProvider(mesh).call_tool("nope", {})


async def test_mesh_tool_provider_rejects_missing_required_argument():
    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a topic")
    async def analyze(topic: str) -> dict:
        return {"points": topic.upper()}

    mesh = Mesh([analyst])
    with pytest.raises(AgentError, match="missing required argument 'topic'"):
        async with mesh:
            await MeshToolProvider(mesh).call_tool("analyze", {})


async def test_mesh_tool_provider_rejects_unexpected_argument():
    analyst = Agent("analyst")

    @analyst.tool(description="No inputs")
    async def ping() -> str:
        return "pong"

    mesh = Mesh([analyst])
    with pytest.raises(AgentError, match="unexpected argument 'extra'"):
        async with mesh:
            await MeshToolProvider(mesh).call_tool("ping", {"extra": True})


async def test_mesh_tool_provider_rejects_wrong_argument_type():
    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a topic")
    async def analyze(topic: str) -> dict:
        return {"points": topic.upper()}

    mesh = Mesh([analyst])
    with pytest.raises(AgentError, match="argument 'topic' expected str, got int"):
        async with mesh:
            await MeshToolProvider(mesh).call_tool("analyze", {"topic": 123})


# --- Task 2: LLMAgent tool-calling loop ---


class Topic(Signal):
    text: str = ""


class Plan(Signal):
    text: str = ""


# --- scripted fake OpenAI-compatible client (queue of responses) ---
class _FF:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _FTC:
    def __init__(self, id, name, arguments):
        self.id, self.function = id, _FF(name, arguments)


class _FMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _FChoice:
    def __init__(self, msg):
        self.message = msg


class _FComp:
    def __init__(self, msg):
        self.choices = [_FChoice(msg)]


class _ScriptedCompletions:
    def __init__(self, msgs):
        self._msgs = list(msgs)
        self.calls = []

    async def create(self, *, model, messages, **kw):
        self.calls.append({"messages": list(messages), **kw})
        # repeat the last scripted message once the queue is exhausted
        msg = self._msgs.pop(0) if len(self._msgs) > 1 else self._msgs[0]
        return _FComp(msg)


class _ScriptedChat:
    def __init__(self, msgs):
        self.completions = _ScriptedCompletions(msgs)


class ScriptedClient:
    def __init__(self, *msgs):
        self.chat = _ScriptedChat(msgs)


class FakeProvider:
    def __init__(self, result=None):
        self.called = []
        self._result = result if result is not None else {"ok": True}

    def tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                    },
                },
            }
        ]

    async def call_tool(self, name, arguments):
        self.called.append((name, arguments))
        return self._result


def _capture(agent):
    emitted = []

    async def capture(sig):
        emitted.append(sig)

    agent.emit = capture  # type: ignore[method-assign]
    return emitted


async def test_tool_loop_executes_then_emits():
    client = ScriptedClient(
        _FMsg(tool_calls=[_FTC("c1", "analyze", '{"topic": "x"}')]),
        _FMsg(content="FINAL"),
    )
    provider = FakeProvider()
    agent = LLMAgent("a", client=client, model="m", on=Topic, emit=Plan, tools=provider)
    emitted = _capture(agent)

    await agent._dispatch(Topic(text="q"))

    assert provider.called == [("analyze", {"topic": "x"})]
    assert emitted[0].text == "FINAL"
    # the tool result was fed back on the second call
    second = client.chat.completions.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second)
    # tools schema was passed on calls
    assert "tools" in client.chat.completions.calls[0]


async def test_exceeds_max_tool_rounds_raises():
    # always returns a tool call -> never finishes
    client = ScriptedClient(_FMsg(tool_calls=[_FTC("c", "analyze", "{}")]))
    agent = LLMAgent(
        "a", client=client, model="m", on=Topic, emit=Plan,
        tools=FakeProvider(), max_tool_rounds=2,
    )
    _capture(agent)
    with pytest.raises(AgentError, match="max_tool_rounds"):
        await agent._dispatch(Topic(text="q"))


async def test_invalid_tool_call_json_raises_agent_error():
    client = ScriptedClient(_FMsg(tool_calls=[_FTC("c", "analyze", "{")]))
    agent = LLMAgent(
        "a", client=client, model="m", on=Topic, emit=Plan, tools=FakeProvider()
    )
    _capture(agent)
    with pytest.raises(AgentError, match="invalid JSON arguments.*analyze"):
        await agent._dispatch(Topic(text="q"))


async def test_non_object_tool_call_arguments_raise_agent_error():
    client = ScriptedClient(_FMsg(tool_calls=[_FTC("c", "analyze", "[]")]))
    agent = LLMAgent(
        "a", client=client, model="m", on=Topic, emit=Plan, tools=FakeProvider()
    )
    _capture(agent)
    with pytest.raises(AgentError, match="arguments must decode to an object"):
        await agent._dispatch(Topic(text="q"))


async def test_no_provider_is_v1_path():
    client = ScriptedClient(_FMsg(content="ONLY"))
    agent = LLMAgent("a", client=client, model="m", on=Topic, emit=Plan)
    emitted = _capture(agent)
    await agent._dispatch(Topic(text="q"))
    assert emitted[0].text == "ONLY"
    assert "tools" not in client.chat.completions.calls[0]


# --- Task 3: Exports + end-to-end integration ---


def test_exports():
    import signal_gating
    assert hasattr(signal_gating, "MeshToolProvider")
    assert "MeshToolProvider" in signal_gating.__all__
    assert "ToolProvider" in signal_gating.__all__


async def test_llm_agent_calls_mesh_tool_end_to_end():
    calls = []
    mesh = Mesh()

    analyst = Agent("analyst")

    @analyst.tool(description="Analyze a topic")
    async def analyze(topic: str) -> dict:
        calls.append(topic)
        return {"summary": f"analyzed {topic}"}

    client = ScriptedClient(
        _FMsg(tool_calls=[_FTC("c1", "analyze", '{"topic": "signals"}')]),
        _FMsg(content="DONE"),
    )
    planner = LLMAgent(
        "planner", client=client, model="m", on=Topic, emit=Plan,
        tools=MeshToolProvider(mesh),
    )

    received = []
    done = asyncio.Event()
    reporter = Agent("reporter")

    @reporter.on(Plan)
    async def collect(signal: Plan) -> None:
        received.append(signal)
        done.set()

    mesh.add(analyst)
    mesh.add(planner)
    mesh.add(reporter)
    mesh.connect(planner, reporter)

    async with mesh:
        await mesh.inject(planner, Topic(text="seed"))
        await asyncio.wait_for(done.wait(), timeout=3.0)

    assert calls == ["signals"]            # the tool actually executed
    assert received[0].text == "DONE"      # the final answer flowed to the sink
