"""Secure agent: fail-closed tool authorization end to end.

An agent with three tools at three risk levels, running behind a
policy engine wired in as an SGP gate. READ auto-approves, DRAFT and
above require approval, unknown tools deny.

Run: PYTHONPATH=src python examples/secure_agent.py
"""

from __future__ import annotations

import asyncio

from signal_gating.agent import Agent, ToolCallSignal
from signal_gating.mesh import Mesh
from signal_gating.security import (
    Decision,
    PolicyEngine,
    RiskLevel,
    ToolRisk,
    classify_tool,
)


def approver(request) -> bool:
    print(f"  [approval] {request.tool_name} ({request.risk.level.name}): {request.reason}")
    # Demo policy: approve DRAFT, reject everything else.
    return request.risk.level is RiskLevel.DRAFT


async def main() -> None:
    engine = PolicyEngine(max_auto_approve=RiskLevel.READ, approver=approver)
    worker = Agent("worker", gates=[engine.as_gate("worker")])

    @worker.tool(risk=RiskLevel.READ)
    async def read_config(key: str) -> str:
        """Read a configuration value."""
        return f"value-of-{key}"

    @worker.tool(risk=RiskLevel.DRAFT)
    async def write_draft(name: str, content: str) -> str:
        """Write a draft document."""
        return f"draft {name} staged"

    @worker.tool(risk=RiskLevel.SEND)
    async def send_email(to: str, body: str) -> str:
        """Send an email."""
        return f"email sent to {to}"

    # Register classifications with the engine (production: load from policy YAML)
    for spec in worker.list_tools():
        engine.register(spec.name, ToolRisk(spec.risk, ("declared",), 1.0))

    async with Mesh([worker]):
        # READ: auto-approved, no approver call
        d = engine.evaluate("read_config", agent_name="worker")
        print(f"read_config -> {d.value}")

        # DRAFT: approver allows
        d = engine.evaluate("write_draft", agent_name="worker")
        print(f"write_draft -> {d.value}")

        # SEND: approver rejects
        d = engine.evaluate("send_email", agent_name="worker")
        print(f"send_email -> {d.value}")

        # Unknown: default deny, no approver consulted
        d = engine.evaluate("rm_rf_everything", agent_name="worker")
        print(f"rm_rf_everything -> {d.value}")

        # Heuristic classification for undeclared tools
        guess = classify_tool("fetch_url", "Fetch a URL over HTTP")
        print(f"classify fetch_url -> {guess.level.name} ({', '.join(guess.reasons)})")

    print(f"\ndenial rate: {engine.denial_rate():.0%}")
    print(f"audit records: {len(engine.records)}")


if __name__ == "__main__":
    asyncio.run(main())
