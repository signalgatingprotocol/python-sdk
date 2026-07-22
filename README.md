# SGP Python SDK

Agent-native executive control for autonomous AI systems.

Multi-agent systems degrade when every event reaches every agent: context fills
with noise, stale information displaces relevant state, and one bad output can
cascade through the system. SGP puts a controlled, observable routing layer
between producers and consumers. Typed signals carry intent, composable gates
decide what passes, agents process admitted work, and meshes define the network.

SGP complements tool and agent-communication protocols such as MCP and A2A; it
controls which signals flow through an agent system and records what happened.

## Install

> Pre-release: the PyPI package has not been published yet. Install the current
> alpha from source:

```bash
pip install git+https://github.com/signalgatingprotocol/python-sdk
```

After the first release, the stable install command will be
`pip install signal-gating`.

For LLM-backed agents (the optional `openai` client):

```bash
pip install "signal-gating[llm] @ git+https://github.com/signalgatingprotocol/python-sdk"
```

For OpenTelemetry export:

```bash
pip install "signal-gating[otel] @ git+https://github.com/signalgatingprotocol/python-sdk"
```

> Alpha: the API surface is still moving.

## Quick Start

```python
import asyncio
from signal_gating import Signal, Gate, Agent, Mesh

class TaskSignal(Signal):
    task: str

# Create agents
planner = Agent("planner")
worker = Agent("worker", gates=[Gate.by_priority(3)])

@worker.on(TaskSignal)
async def handle(signal: TaskSignal):
    print(f"Working on: {signal.task}")

# Connect in a mesh
mesh = Mesh([planner, worker])
mesh.connect(planner, worker)

async def main():
    async with mesh:
        await planner.emit(TaskSignal(task="build feature", priority=5))

asyncio.run(main())
```

A clean `async with mesh` exit waits for queued and in-flight work, including
signals emitted to downstream agents. To wait while keeping the mesh open, use
`await mesh.wait_idle(timeout=10)`; a timeout identifies the agents that are
still busy instead of silently dropping work.

## Core Primitives

### Signal

Typed, immutable events that flow through the system. Subclass to define your domain:

```python
class AlertSignal(Signal):
    message: str
    severity: int = 0
```

Signals are immutable. Use `evolve()` to create modified copies:

```python
alert = AlertSignal(message="CPU spike", severity=8)
enriched = alert.with_metadata(region="us-east")
escalated = alert.evolve(severity=10)
```

### Gate

Composable predicates that control signal flow. Combine with operators:

```python
# Filter, transform, compose
high_priority = Gate.by_priority(5)
enrich = Gate.transform(lambda s: s.with_metadata(reviewed=True))
dedup = Gate.deduplicate(window=60)

# Compose with operators
pipeline = high_priority >> dedup >> enrich  # chain
either = Gate.by_type(AlertSignal) | Gate.by_priority(10)  # or
both = high_priority & Gate.filter(lambda s: s.source == "sensor")  # and
not_low = ~Gate.by_priority(1)  # invert
```

Starter gates: `by_type`, `by_priority`, `filter`, `transform`, `deduplicate`, `passthrough`.

Advanced gates: `rate_limit`, `throttle`, `retry`, `circuit_breaker`, `timeout`, `ttl`, `debounce`, `sample`, `when`, `block`, `tap`, `batch`, `parallel`, `fallback`, `window`, `map`.

**Real-time signal control:**

```python
# Throttle: drop excess signals instead of queuing (unlike rate_limit which sleeps)
fast_gate = Gate.throttle(100)  # Max 100/sec, drop the rest

# TTL: drop stale signals (freshness matters in real-time systems)
fresh_only = Gate.ttl(30)  # Drop signals older than 30 seconds

# Debounce: wait for silence before passing, to tame noisy signal sources
stable = Gate.debounce(0.5)  # Pass only after 500ms of quiet

# Conditional branching: the agent-native if/else for signal flow
gate = Gate.when(
    lambda s: s.priority >= 8,
    then=Gate.transform(enrich_urgent),
    otherwise=Gate.transform(enrich_normal),
)
```

### Agent

Autonomous signal processors with lifecycle management, request/response, and priority inboxes:

```python
worker = Agent("worker", gates=[Gate.by_priority(3)], priority_inbox=True)

@worker.on_start
async def setup():
    worker.state["db"] = await connect_db()

@worker.on_stop
async def cleanup():
    await worker.state["db"].close()

@worker.on(TaskSignal)
async def handle(signal: TaskSignal):
    result = await process(signal.task)
    await worker.reply(signal, ResultSignal(result=result))
```

**AgentContext**: handlers can receive a context object, eliminating closure boilerplate:

```python
from signal_gating import AgentContext

@worker.on(TaskSignal)
async def handle(signal: TaskSignal, ctx: AgentContext):
    ctx.state["count"] = ctx.state.get("count", 0) + 1
    await ctx.emit(ResultSignal(result="done"))
    await ctx.reply(ResultSignal(result="response"))  # auto-correlates
```

**once()**: handlers that fire exactly once, then auto-remove:

```python
@worker.once(StartupSignal)
async def first_only(signal: StartupSignal):
    print("Initialization complete. Won't fire again.")
```

Request/response: agents can ask questions and wait for answers:

```python
response = await planner.request(TaskSignal(task="analyze data"), timeout=5.0)
```

**Restartable agents**: agents can be stopped and restarted with fresh inboxes:

```python
await worker.stop()
# ... fix the issue, update config ...
await worker.start()  # Fresh inbox, preserved state and handlers
```

**Supervision**: agents auto-restart on failure with exponential backoff:

```python
worker = Agent("worker", max_restarts=5, restart_delay=1.0)
# Delays: 1s, 2s, 4s, 8s, 16s between restarts
```

### Mesh

Agent network topology with gated connections and content-based routing:

```python
mesh = Mesh([coordinator, analyst, reporter])
mesh.connect(coordinator, analyst, gate=Gate.by_priority(5))
mesh.connect(analyst, reporter)

# Fan-out and fan-in
mesh.broadcast_connect(source, [a, b, c])
mesh.converge_connect([a, b, c], target)

# Content-based routing: signals go where they need to based on content
mesh.route(coordinator, [
    (lambda s: s.priority >= 8, critical_handler),
    (lambda s: isinstance(s, AnalysisTask), analyst),
], default=general_worker)

# Dynamic topology: rewire at runtime
mesh.disconnect(coordinator, analyst)  # Fully stops signal flow
await mesh.remove(analyst)  # Remove agent, cleanup all connections

# Lifecycle with graceful drain
async with mesh:
    await coordinator.emit(TaskSignal(task="analyze"))
# Clean context exit waits for all pending and in-flight signals to complete.
```

**Interceptors**: mesh-level cross-cutting concerns (auth, logging, metrics):

```python
def audit_log(signal, source, target):
    print(f"[AUDIT] {source} -> {target}: {type(signal).__name__}")
    return signal  # Return None to block

mesh.intercept(audit_log)
```

Interceptors run on every delivery mediated by `Mesh`: connected, content, and
load-balanced routes; direct orchestration such as injection, pub/sub,
workflows, and tool calls; request, scatter, and race sends and their correlated
responses; and trajectory replay. Returning `None` blocks that delivery.
Awaited point-to-point operations fail immediately with an actionable
`MeshError`. Connected and other fire-and-forget routes drop observably.
`publish()` continues to the remaining subscribers and returns the number that
accepted the signal. `race()` continues while an allowed candidate remains.
Each allowed interceptor output passes to later interceptors and, for routed
delivery, any route gate. The final allowed post-gate signal is the exact signal
delivered, delivery-traced, and action-recorded by mesh event sinks.

Direct writes to `agent.inbox` deliberately bypass mesh delivery interceptors,
mesh delivery traces, and receipts. If the receiving agent processes that
signal, it can still emit its own gate and dispatch processing spans through its
attached tracer. Treat inboxes as internal channels whenever the mesh is your
authorization or audit boundary.

**Capability Discovery**: find agents by what they can do, not just by name:

```python
mesh.declare_capabilities(analyst, "analysis", "summarization")
mesh.declare_capabilities(coder, "code_generation", "debugging")

# Find all agents capable of analysis
agents = mesh.find_capable("analysis")
```

**Scatter/Gather**: the fundamental multi-agent coordination pattern:

```python
# Send work to N agents in parallel, collect all responses
responses = await mesh.scatter(
    TaskSignal(task="analyze market"),
    [analyst1, analyst2, analyst3],
    timeout=10.0,
)
```

Scatter is all-or-nothing: if any target misses the deadline, it raises
`asyncio.TimeoutError` naming every missing agent instead of returning ambiguous
partial data. For custom partial-result policies, compose individual
`mesh.request()` tasks.

Each scatter target must be unique, including when mixing agent objects and
names. Passing the same agent more than once raises `MeshError` before any work
is sent; use separate requests when intentionally repeating work.

**Map/Reduce**: parallel analysis, then synthesis:

```python
# Distribute to N analysts, then combine through a synthesizer
result = await mesh.map_reduce(
    AnalyzeSignal(data="quarterly revenue report"),
    mappers=[trend_analyst, risk_analyst, sentiment_analyst],
    reducer=synthesizer,
    timeout=30.0,
)
# synthesizer receives all three analyses in metadata["responses"]
```

**Branching Workflows**: conditional agent chains:

```python
# Route through different agent chains based on signal content
result = await mesh.branch_workflow(
    TaskSignal(task="analyze", priority=9),
    router=lambda s: "critical" if s.priority >= 8 else "normal",
    branches={
        "critical": [validator, deep_analyzer, reviewer],
        "normal": [quick_analyzer],
    },
)
```

**Sequential Workflows**: ordered multi-step processing:

```python
# Chain agents: each response becomes the next agent's input
result = await mesh.workflow(
    TaskSignal(task="analyze quarterly revenue"),
    steps=[data_fetcher, analyzer, summarizer, formatter],
    timeout=60.0,
)
```

**Race**: first response wins:

```python
# Try multiple strategies in parallel, take the fastest
result = await mesh.race(
    AnalyzeSignal(data=data),
    [cache_lookup, fast_analyzer, deep_analyzer],
    timeout=5.0,
)
```

### Tool Calling: Agent-Native RPC

Agents can expose tools that other agents (or LLMs) discover and invoke. This is the bridge between signal-based communication and structured function calling:

```python
from signal_gating import Agent, Mesh, ToolSpec

analyst = Agent("analyst")

@analyst.tool(description="Analyze data and return insights")
async def analyze(data: str, depth: int = 1) -> dict:
    return {"insights": await run_analysis(data, depth)}

@analyst.tool(description="Summarize text")
async def summarize(text: str) -> str:
    return text[:100] + "..."

# Discover tools across the mesh
mesh = Mesh([analyst, coordinator])
all_tools = mesh.discover_tools()
# {"analyst": [ToolSpec(name="analyze", ...), ToolSpec(name="summarize", ...)]}

# Call tools directly through the mesh
result = await mesh.call_tool(analyst, "analyze", data="revenue Q4", depth=2)

# Export SGP tool schemas for discovery
schema = analyst.tools_schema()
# [{"name": "analyze", "description": "...", "parameters": {...}}]
```

`MeshToolProvider` converts these tools into OpenAI-compatible function-tool
schemas. It is not an MCP adapter; direct `mesh.call_tool()` calls route through
the mesh request path and are captured by `mesh.record(...)`.

`Agent.tool()` accepts both async and synchronous functions. Async tools run on
the event loop; synchronous tools run in a worker thread so blocking I/O does
not stall the mesh. Use an async tool for event-loop-bound resources or when
cancellation must stop the underlying operation.

### Pipeline

Ordered gate chains:

```python
pipeline = Pipeline([
    Gate.by_priority(3),
    Gate.deduplicate(window=30),
    Gate.transform(enrich),
])
result = await pipeline.process(signal)
```

### Channel

Async typed conduits with backpressure and priority ordering:

```python
# Standard FIFO channel with backpressure
channel = Channel(Signal, buffer_size=100)
await channel.send(signal)        # Raises ChannelFull if full
await channel.send_wait(signal)   # Blocks until space is available

# Priority channel: highest priority dequeued first
from signal_gating import PriorityChannel
channel = PriorityChannel(Signal, buffer_size=1000)
```

### Tracing

Signal flow observability, in memory by default and exportable when you need to
plug SGP into production telemetry:

```python
from signal_gating import OpenTelemetrySpanExporter, Tracer

tracer = Tracer(sinks=[OpenTelemetrySpanExporter()])
tracer.record(trace_id, signal_id, "agent-a", "priority_gate", "passed")
trace = tracer.get_trace(trace_id)
print(tracer.summary())
```

Sinks are best-effort observers, so a collector outage does not stop the agent
mesh. You can also export retained spans after a run:

```python
exported = tracer.export(OpenTelemetrySpanExporter())
```

Tracing is lightweight observability. Trajectories are the durable audit/replay
log for signal-carrying mesh events.

### LLM-backed agents (Hermes)

`LLMAgent` gives an agent an OpenAI-compatible brain. Nous Hermes, or any
OpenAI-compatible server. Install the extra:

```bash
pip install "signal-gating[llm] @ git+https://github.com/signalgatingprotocol/python-sdk"
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

**Autonomous tool-calling.** Give the agent the mesh's tools and it will reason,
call them, and act before answering:

```python
from signal_gating import Agent, Mesh
from signal_gating.llm import LLMAgent, MeshToolProvider

mesh = Mesh()
analyst = Agent("analyst")

@analyst.tool(description="Analyze a topic and return key points")
async def analyze(topic: str) -> dict:
    return {"points": [...]}

planner = LLMAgent.from_openai(
    "planner",
    base_url="http://127.0.0.1:8642/v1", api_key="...", model="hermes-agent",
    tools=MeshToolProvider(mesh),
)
mesh.add(analyst)
mesh.add(planner)
# planner can now call analyst.analyze before emitting its result.
```

The loop is bounded by `max_tool_rounds` (default 4).

### Focused improvement loops

Trajectories preserve experience. `ImprovementLoop` turns evaluation evidence
into controlled improvement across runs. It is model- and harness-agnostic: the
candidate can be a prompt, model selection, tool policy, memory strategy,
fine-tune ID, mesh topology, or a compound configuration owned by your code.

Define the capability target explicitly, then supply three callables:

- a harness that runs one candidate on one case;
- an evaluator that returns every objective's score and concrete evidence;
- an improver that receives the weakest dimension, its weakest cases, evaluator
  evidence, and prior experiment history.

```python
from dataclasses import replace
from signal_gating import (
    Assessment, EvaluationCase, EvaluationSuite, ImprovementLoop, Objective,
)

cases = [
    EvaluationCase("routine", "summarize the filing"),
    EvaluationCase("adversarial", "find the unsupported claim", weight=2),
]
objectives = [
    Objective("correctness", target=0.90, weight=2),
    Objective("reliability", target=0.95, regression_tolerance=0.01),
]

async def harness(config, case):
    return await run_your_model(config, case.input)

async def evaluate(case, output):
    judged = await your_judge(case, output)
    return Assessment(scores=judged.scores, evidence=judged.evidence)

async def improve(context):
    # Change one thing aimed at context.focus; rejected attempts remain in
    # context.history and never replace context.incumbent.
    return replace(context.incumbent, system=revise_prompt(context))

suite = EvaluationSuite(cases, objectives, harness, evaluate, samples=3)
loop = ImprovementLoop(suite, improve, identify=lambda config: config.version)
result = await loop.run(initial_config, max_iterations=8)
```

`max_concurrency` is a hard execution and scheduler bound: the suite creates a
fixed worker pool, not one task per case or sample. Raw model outputs are
discarded immediately after evaluation by default; scores and evidence remain
in the report. Set `retain_outputs=True` only when the improver genuinely needs
the full responses and the memory cost is understood.

Each candidate runs on the same weighted cases. Scores reduce by median across
samples. The default acceptance policy requires both total target progress and
the selected weak dimension to improve. Any aggregate or per-case regression
beyond an objective's tolerance rejects the candidate, even if its average score
rose. `ImprovementHistory("experiments.jsonl")` adds hash-chained persistence so
later runs can learn from accepted and rejected scores, regressions, and
per-case evaluator evidence without serializing opaque candidate objects, raw
harness outputs, or model clients.

Durable history is bounded by default: 30 days, 500 experiment records, and
8 MiB, whichever limit is reached first. Cleanup runs only when a bound is
crossed, keeps the newest evidence, atomically rebuilds the hash chain, and
writes the compacted file with `0600` permissions. Tune the limits for a smaller
machine or disable an individual bound with `None`:

```python
from signal_gating import ImprovementHistory, RetentionPolicy

history = ImprovementHistory(
    "experiments.jsonl",
    retention=RetentionPolicy(
        max_age_seconds=7 * 24 * 60 * 60,
        max_records=100,
        max_bytes=2 * 1024 * 1024,
    ),
)
```

Pass `retention=None` only when an external lifecycle system already owns
cleanup. Compaction affects durable history, never the active run's in-memory
feedback context or returned records.

Lifecycle events (`baseline_evaluated`, `focus_selected`, candidate decision,
and stop state) are ordinary `ImprovementEvent` signals exposed through an
optional observer. Bridge that observer to a mesh when improvement events should
drive agents or durable trajectory recording. See `examples/focused_improvement.py`
for a deterministic end-to-end loop that first rejects a regression, then reaches
all targets through focused changes.

### Trajectories

Capture a verifiable, structured record of signal-carrying mesh events,
exportable as JSONL for audit, learning, or training. The production hook records
connected routes plus direct orchestration paths such as `inject`, `request`,
`workflow`, `scatter`, `race`, `publish`, and `call_tool`:

```python
from signal_gating import TrajectoryRecorder

recorder = TrajectoryRecorder()
mesh.record(recorder)

async with mesh:
    await mesh.inject(planner, Topic(text="..."))

recorder.trajectories()              # {trace_id: [Receipt, ...]}, grouped per run
recorder.export_jsonl("runs.jsonl")  # one Receipt per line
```

Each `Receipt` carries `event_kind`, `action`, the signal's lineage
(`trace_id` / `parent_id`), routing (`source` -> `target`), typed domain
`payload`, event `metadata`, and a `digest` (sha256) — an integrity checksum
that `receipt.verify()` uses to catch accidental corruption (truncated writes,
bit-rot) before replay. The digest is keyless, so it is not a cryptographic
signature; to make a persisted trajectory tamper-evident against a motivated
actor, sign or HMAC the file with a key the verifier holds out of band.

A trajectory is more than readable: its signal-carrying receipts are
**reconstructable**. Each receipt stores the full signal wire envelope, so a run
persisted to disk reloads as verifiable receipts and exact typed signals.

```python
reloaded = TrajectoryRecorder()
reloaded.load_jsonl("runs.jsonl")    # verifiable Receipts, after a restart
signals = reloaded.replay()          # -> [TaskSignal, ...], original types and ids
```

`TrajectoryRecorder.replay()` reconstructs signals for inspection, audit, or
training. To deliver recorded entry events through a mesh again, use
`TrajectoryReplayRunner.replay_into(mesh)`.

`TrajectoryReplayRunner` re-delivers entry events such as `inject`,
`request_sent`, `scatter_sent`, `race_sent`, and `published` into the current
mesh. The mesh must already contain matching target agents and handlers. It
skips audit/control events and does not recreate pending request futures,
workflow loops, LLM memory, filesystem state, or Claude Agent SDK sessions:

```python
from signal_gating import TrajectoryReplayRunner

runner = TrajectoryReplayRunner.from_jsonl("runs.jsonl")
result = await runner.replay_into(mesh)
print(result.delivered, result.skipped)
```

For legacy interceptor-based capture, `mesh.intercept(recorder)` still works.
Because interceptors cover the complete mesh trust boundary, it observes every
mesh-mediated delivery, including direct orchestration, correlated responses,
and replay, rather than only connected edge hops. It records generic `hop`
receipts at the interceptor observation point. Prefer `mesh.record(recorder)`
for stable, action-specific receipts emitted after successful delivery, plus
structured `intercepted` and `edge_rejected` receipts for blocked attempts.

### Wire format & durability

A protocol that only lives inside one process is a library. Signals serialize to
a self-describing JSON envelope and reconstruct as their **original subclass** —
the foundation for persistence, durable replay, and crossing process or network
boundaries. Subclasses register themselves automatically; no boilerplate:

```python
class TaskSignal(Signal):
    task: str

sig = TaskSignal(task="build", priority=5)
raw = sig.to_json()                     # {"sgp": 1, "type": "TaskSignal", "data": {...}}

restored = Signal.from_json(raw)        # -> TaskSignal, not a dict
assert type(restored) is TaskSignal
assert restored == sig                  # faithful: id, trace_id, timestamp, fields
```

`model_dump()` is lossy in the way that matters — it gives you a `dict` with no
way back to `TaskSignal`. The registry closes that loop. Pin a stable wire name
across refactors with `__signal_type__ = "task.v2"`, or register an alias with
`register_signal`. Unknown types raise `UnknownSignalType`; pass `strict=False`
to get a best-effort base `Signal` with the payload preserved in `metadata`.

This makes recovery durable. Persist the dead-letter queue on shutdown, then
reload and replay after a crash or redeploy — signals come back as their real
types and dispatch to the same handlers:

```python
agent.dead_letters.to_jsonl("dlq.jsonl")   # persist failed signals + context

# ... process restarts ...

agent.dead_letters.load_jsonl("dlq.jsonl") # reconstruct as original types
await agent.dead_letters.replay(agent.inbox)
```

### Teams

A `Team` coordinates long-lived peer agents over a shared `TaskBoard` — a
durable task ledger folded from signal events, with hash-chained JSONL
persistence and transition policy expressed as ordinary `Gate`s. Members carry
exactly one obligation: handle `TaskAssigned` and reply a `TaskResult`. The
protocol (claiming, completion, release-on-failure, idle notification,
shutdown) lives in team-owned steward coroutines, never inside your agents:

```python
from signal_gating import Agent, AgentContext, Mesh, TaskAssigned, TaskResult, Team

writer = Agent("writer")

@writer.on(TaskAssigned)
async def work(signal: TaskAssigned, ctx: AgentContext):
    await ctx.reply(TaskResult(task_id=signal.task_id, result={"done": True}))

mesh = Mesh([writer])
team = Team("docs", mesh)
team.enroll(writer)

async with mesh:
    async with team:
        tid = await team.open("rewrite channel docs", payload={"path": "channel.py"})
        # stewards self-claim; or direct it: await team.assign(tid, "writer")
```

A crashed handler dead-letters as usual and its task is released back to
pending for a peer to pick up. Tasks can depend on other tasks
(`depends_on=...`); completing the last dependency unblocks dependents
automatically. See `examples/agent_team.py`.

### Scripted workflows

A `Script` moves the plan into code: a coroutine you write owns the loop and
the branching, and the runtime contributes bounded concurrency, an agent
budget, and resume — every completed step is checkpointed under a
content-addressed key, so an interrupted run reruns only unfinished work:

```python
from signal_gating import CheckpointStore, Script

async def audit(ctx):
    async with ctx.phase("scan"):
        findings = await ctx.fan_out(["scan-a", "scan-b"], requests)
    async with ctx.phase("verify"):
        return [await ctx.run("verifier", f) for f in dedupe(findings)]

script = Script("sweep", mesh, audit, max_concurrency=16,
                store=CheckpointStore("sweep.jsonl"))
report = await script.run(args={...})   # rerun after interruption -> resumes
```

`ctx.spawn(factory, signal)` runs ephemeral agents — added, started, asked one
checkpointed request, then removed — so a script can use dozens of agents
without preregistering them. Unrelated to `mesh.workflow()`, which is a
one-shot step chain. See `examples/scripted_workflow.py`.

## Architecture

```
Signal -> [Gate >> Gate >> Gate] -> Agent -> [Gate] -> Agent -> ...
              Pipeline                  Edge (Mesh)

Mesh: Agent --(gate)--> Agent --(gate)--> Agent
        |                                    ^
        +----------(gate)-------------------+
                   fan-out / fan-in
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/
```

## License

Apache 2.0
