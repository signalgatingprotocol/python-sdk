# SGP Python SDK

Agent-native signal orchestration for autonomous AI systems.

The Signal Gating Protocol provides composable primitives for building multi-agent systems with controlled, observable signal flow. Signals are typed, immutable events. Gates are composable predicates that control which signals pass. Agents process signals autonomously. Meshes connect agents into networks.

## Install

```bash
pip install signal-gating
```

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
        await asyncio.sleep(0.05)

asyncio.run(main())
```

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

Built-in gates: `filter`, `transform`, `by_type`, `by_priority`, `rate_limit`, `throttle`, `deduplicate`, `retry`, `circuit_breaker`, `timeout`, `ttl`, `debounce`, `sample`, `when`, `passthrough`, `block`, `tap`, `batch`, `parallel`, `fallback`, `window`, `map`.

**Real-time signal control:**

```python
# Throttle: drop excess signals instead of queuing (unlike rate_limit which sleeps)
fast_gate = Gate.throttle(100)  # Max 100/sec, drop the rest

# TTL: drop stale signals — freshness matters in real-time systems
fresh_only = Gate.ttl(30)  # Drop signals older than 30 seconds

# Debounce: wait for silence before passing — tame noisy signal sources
stable = Gate.debounce(0.5)  # Pass only after 500ms of quiet

# Conditional branching — the agent-native if/else for signal flow
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

**AgentContext** — handlers can receive a context object, eliminating closure boilerplate:

```python
from signal_gating import AgentContext

@worker.on(TaskSignal)
async def handle(signal: TaskSignal, ctx: AgentContext):
    ctx.state["count"] = ctx.state.get("count", 0) + 1
    await ctx.emit(ResultSignal(result="done"))
    await ctx.reply(ResultSignal(result="response"))  # auto-correlates
```

**once()** — handlers that fire exactly once, then auto-remove:

```python
@worker.once(StartupSignal)
async def first_only(signal: StartupSignal):
    print("Initialization complete — won't fire again")
```

Request/response — agents can ask questions and wait for answers:

```python
response = await planner.request(TaskSignal(task="analyze data"), timeout=5.0)
```

**Restartable agents** — agents can be stopped and restarted with fresh inboxes:

```python
await worker.stop()
# ... fix the issue, update config ...
await worker.start()  # Fresh inbox, preserved state and handlers
```

**Supervision** — agents auto-restart on failure with exponential backoff:

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

# Content-based routing — signals go where they need to based on content
mesh.route(coordinator, [
    (lambda s: s.priority >= 8, critical_handler),
    (lambda s: isinstance(s, AnalysisTask), analyst),
], default=general_worker)

# Dynamic topology — rewire at runtime
mesh.disconnect(coordinator, analyst)  # Fully stops signal flow
await mesh.remove(analyst)  # Remove agent, cleanup all connections

# Lifecycle with graceful drain
async with mesh:
    await coordinator.emit(TaskSignal(task="analyze"))
await mesh.stop(drain=True)  # Wait for all pending signals to complete
```

**Interceptors** — mesh-level cross-cutting concerns (auth, logging, metrics):

```python
def audit_log(signal, source, target):
    print(f"[AUDIT] {source} -> {target}: {type(signal).__name__}")
    return signal  # Return None to block

mesh.intercept(audit_log)
```

**Capability Discovery** — find agents by what they can do, not just by name:

```python
mesh.declare_capabilities(analyst, "analysis", "summarization")
mesh.declare_capabilities(coder, "code_generation", "debugging")

# Find all agents capable of analysis
agents = mesh.find_capable("analysis")
```

**Scatter/Gather** — the fundamental multi-agent coordination pattern:

```python
# Send work to N agents in parallel, collect all responses
responses = await mesh.scatter(
    TaskSignal(task="analyze market"),
    [analyst1, analyst2, analyst3],
    timeout=10.0,
)
```

**Map/Reduce** — parallel analysis, then synthesis:

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

**Branching Workflows** — conditional agent chains:

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

**Sequential Workflows** — ordered multi-step processing:

```python
# Chain agents: each response becomes the next agent's input
result = await mesh.workflow(
    TaskSignal(task="analyze quarterly revenue"),
    steps=[data_fetcher, analyzer, summarizer, formatter],
    timeout=60.0,
)
```

**Race** — first response wins:

```python
# Try multiple strategies in parallel, take the fastest
result = await mesh.race(
    AnalyzeSignal(data=data),
    [cache_lookup, fast_analyzer, deep_analyzer],
    timeout=5.0,
)
```

### Tool Calling — Agent-Native RPC

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

# Export tool schemas for LLM integration
schema = analyst.tools_schema()
# [{"name": "analyze", "description": "...", "parameters": {...}}]
```

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

# Priority channel — highest priority dequeued first
from signal_gating import PriorityChannel
channel = PriorityChannel(Signal, buffer_size=1000)
```

### Tracing

Signal flow observability:

```python
tracer = Tracer()
tracer.record(trace_id, signal_id, "agent-a", "priority_gate", "passed")
trace = tracer.get_trace(trace_id)
print(tracer.summary())
```

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

MIT
