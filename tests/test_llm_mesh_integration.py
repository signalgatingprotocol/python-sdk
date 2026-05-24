"""End-to-end: LLMAgents coordinating through a real Mesh with a fake brain.

The unit tests in test_llm_agent.py exercise the reasoning loop in isolation
(by dispatching directly). This test drives the real orchestration path:
a signal injected into the first agent's inbox, processed by its LLM handler,
emitted across mesh edges to the next LLMAgent, and finally to a plain sink.
No live model: a fake OpenAI-compatible client returns canned replies.
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


# --- minimal fake OpenAI-compatible client (one fixed reply) ---
class _Msg:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def create(self, *, model, messages, **kw):
        return _Completion(self._reply)


class _Chat:
    def __init__(self, reply: str) -> None:
        self.completions = _Completions(reply)


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.chat = _Chat(reply)


async def test_llm_agents_coordinate_through_mesh():
    planner = LLMAgent(
        "planner", client=FakeClient("OUTLINE"), model="m", on=Topic, emit=Plan
    )
    writer = LLMAgent(
        "writer", client=FakeClient("DRAFT"), model="m", on=Plan, emit=Draft
    )
    reporter = Agent("reporter")

    received: list[Draft] = []
    done = asyncio.Event()

    @reporter.on(Draft)
    async def collect(signal: Draft) -> None:
        received.append(signal)
        done.set()

    mesh = Mesh([planner, writer, reporter])
    mesh.connect(planner, writer)
    mesh.connect(writer, reporter)

    topic = Topic(text="seed")
    async with mesh:
        # inject() feeds the planner's inbox so it processes the topic;
        # planner.emit() would send it past the planner to the writer.
        await mesh.inject(planner, topic)
        await asyncio.wait_for(done.wait(), timeout=3.0)

    assert len(received) == 1
    draft = received[0]
    assert isinstance(draft, Draft)
    assert draft.text == "DRAFT"
    # trace lineage survives both LLM hops through the mesh
    assert draft.trace_id == topic.trace_id
