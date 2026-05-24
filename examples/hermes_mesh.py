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
