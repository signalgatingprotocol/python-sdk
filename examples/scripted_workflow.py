"""Scripted workflow example: a checkpointed fan-out sweep with resume.

The plan lives in the `audit` coroutine, not in any agent: it fans scan
requests across two scanners, dedupes the findings, then verifies each one.
Every completed step is checkpointed to sweep-checkpoints.jsonl — Ctrl-C
mid-run and rerun to watch resume skip the finished steps (cached steps
print nothing; only unfinished work touches the mesh). Delete the JSONL to
start fresh.
"""

import asyncio

from signal_gating import (
    Agent,
    AgentContext,
    CheckpointStore,
    Mesh,
    Script,
    ScriptContext,
    Signal,
)

PATHS = [f"src/module_{n}.py" for n in range(10)]


class ScanReq(Signal):
    __signal_type__ = "examples.sweep.scan_req"
    path: str = ""


class Finding(Signal):
    __signal_type__ = "examples.sweep.finding"
    issue: str = ""


class Verdict(Signal):
    __signal_type__ = "examples.sweep.verdict"
    issue: str = ""
    confirmed: bool = False


def make_scanner(name: str) -> Agent:
    agent = Agent(name)

    @agent.on(ScanReq)
    async def scan(signal: ScanReq, ctx: AgentContext):
        print(f"  [{name}] scanning {signal.path}")
        await asyncio.sleep(0.1)  # simulate work so Ctrl-C mid-run is easy
        issue = f"unchecked input in {signal.path}"
        await ctx.reply(Finding(issue=issue))

    return agent


def make_verifier() -> Agent:
    agent = Agent("verifier")

    @agent.on(Finding)
    async def verify(signal: Finding, ctx: AgentContext):
        print(f"  [verifier] checking: {signal.issue}")
        await ctx.reply(Verdict(issue=signal.issue, confirmed=True))

    return agent


async def audit(ctx: ScriptContext) -> dict:
    async with ctx.phase("scan"):
        findings = await ctx.fan_out(
            ["scan-a", "scan-b"], [ScanReq(path=p) for p in ctx.args["paths"]]
        )
    issues = sorted({f.issue for f in findings})

    async with ctx.phase("verify"):
        verdicts = [await ctx.run("verifier", Finding(issue=i)) for i in issues]

    confirmed = [v.issue for v in verdicts if v.confirmed]
    return {"scanned": len(findings), "confirmed": len(confirmed)}


async def main():
    mesh = Mesh([make_scanner("scan-a"), make_scanner("scan-b"), make_verifier()])
    script = Script(
        "endpoint-sweep",
        mesh,
        audit,
        max_concurrency=4,
        store=CheckpointStore("sweep-checkpoints.jsonl"),
    )
    async with mesh:
        report = await script.run(args={"paths": PATHS})
    print(f"\nReport: {report}")
    print("Rerun this script: completed steps replay from sweep-checkpoints.jsonl.")


asyncio.run(main())
