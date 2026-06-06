# SGP Python SDK

Agent-native signal orchestration for autonomous AI systems.

The Signal Gating Protocol provides composable primitives for building multi-agent systems with controlled, observable signal flow. Signals are typed, immutable events. Gates are composable predicates that control which signals pass. Agents process signals autonomously. Meshes connect agents into networks.

## Install

```bash
pip install git+https://github.com/signalgatingprotocol/python-sdk
```

For LLM-backed agents (the optional `openai` client):

```bash
pip install "signal-gating[llm] @ git+https://github.com/signalgatingprotocol/python-sdk"
```

For OpenTelemetry export:

```bash
pip install "signal-gating[otel] @ git+https://github.com/signalgatingprotocol/python-sdk"
```

For Claude Agent SDK integrations:

```bash
pip install "signal-gating[claude] @ git+https://github.com/signalgatingprotocol/python-sdk"
```

For local development from a checkout:

```bash
pip install -e ".[dev]"
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
await mesh.stop(drain=True)  # Wait for all pending signals to complete
```

**Interceptors**: mesh-level cross-cutting concerns (auth, logging, metrics):

```python
def audit_log(signal, source, target):
    print(f"[AUDIT] {source} -> {target}: {type(signal).__name__}")
    return signal  # Return None to block

mesh.intercept(audit_log)
```

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
schemas. `ClaudeMeshMCPAdapter` exposes the same registry through MCP-shaped
`tools/list` and `tools/call` payloads for Claude integration. Direct
`mesh.call_tool()` calls route through the mesh request path and are captured by
`mesh.record(...)`.

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

### Financial physics gates

Finance-like agent meshes need controls that generic task examples do not
stress: event-time freshness, sequence monotonicity, quote sanity, edge after
slippage, per-order notional limits, liquidity participation, and cumulative
exposure budgets. The SDK keeps those as domain helpers in `signal_gating.finance`
so they compose with ordinary gates without expanding the generic protocol
surface:

```python
from signal_gating import MarketGate, MarketTick

tick_gate = (
    MarketGate.freshness(max_age_ms=250)
    >> MarketGate.monotonic_sequence()
    >> MarketGate.quote_sanity(max_spread_bps=12)
)

tick = MarketTick(
    symbol="AAPL",
    venue="XNAS",
    bid=100.00,
    ask=100.04,
    event_ts=event_time,
    sequence=101,
    priority=8,
)
```

`MarketDecision` and `MarketGate.decision_edge()` add a second control layer for
execution-facing signals: only decisions with enough net edge after expected
slippage and enough confidence pass. `MarketGate.notional_limit()` bounds a
single order, `MarketGate.participation_limit()` bounds the order relative to
visible or estimated liquidity, and `MarketGate.exposure_limit()` bounds
cumulative gross or net exposure for a symbol/venue key, optionally over a
rolling window. See
`examples/financial_physics_mesh.py` for an offline, deterministic market-signal
mesh with trajectory receipts.

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

Receipt metrics can use the same optional `signal-gating[otel]` extra without
turning raw receipts into telemetry events:

```python
from signal_gating import OpenTelemetryReceiptMetricsExporter
from signal_gating.receipts_cli import build_receipt_metrics

metrics = build_receipt_metrics("runs.jsonl", event_kinds=["claude_mcp_http"])
OpenTelemetryReceiptMetricsExporter()(metrics)
```

The same aggregate batch is available from the receipt CLI:

```bash
signal-gating-receipts auth runs.jsonl --otel
```

`--otel` is additive: on success it exports the verified aggregate metrics
through OpenTelemetry and still prints the same aggregate JSON summary to
stdout.

Path labels are explicit opt-in. By default, `--otel` omits raw receipt
`payload.path` values from OpenTelemetry dimensions to avoid high-cardinality
labels and accidental disclosure of route, tenant, file, or object identifiers.
Use `--otel-include-paths` only for approved internal telemetry pipelines:

```bash
signal-gating-receipts auth runs.jsonl --otel --otel-include-paths --otel-max-paths 25
```

When path labels are enabled, only the top path values are exported; remaining
path counts are folded into `sgp.value="__other__"`. The default cap is 100.

The receipt metrics exporter consumes aggregate counters only. It does not read
or export receipt `payload`, `wire`, or `metadata`, and it skips raw path values
by default to avoid high-cardinality labels.

Tracing is lightweight observability. Trajectories are the durable audit/replay
log for signal-carrying mesh events.

Mesh execution events are bridged into tracing automatically. Direct
orchestration paths such as `inject`, `request_sent`, `scatter_sent`,
`response_received`, `tool_call_start`, and `tool_call_complete` become
`mesh_event` spans even when no `TrajectoryRecorder` is attached. The bridge
exports safe routing and correlation metadata, not raw signal payloads, tool
arguments, or model results.

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

### Claude Agent SDK integration

`ClaudeAgent` delegates a mesh agent's work to the Claude Agent SDK while keeping
the SGP boundary typed and replay-aware:

```python
from signal_gating import ClaudeAgent, ClaudeAgentRunSignal, ClaudeToolPolicy, Mesh

claude = ClaudeAgent(
    "claude_worker",
    tool_policy=ClaudeToolPolicy.read_only(),
    permission_mode="acceptEdits",
)

mesh = Mesh([claude, sink])
mesh.connect(claude, sink)

async with mesh:
    await mesh.inject(claude, ClaudeAgentRunSignal(prompt="Review this module"))
```

Claude handles its own agent loop features such as built-in tools, MCP servers,
permissions, hooks, subagents, and session transcripts. SGP handles typed signal
routing, gates, mesh coordination, tracing, and trajectory receipts around those
runs. `ClaudeAgentSDKRunner` records audit-only receipts for a direct one-shot
SDK `query()`. `ClaudeAgentSDKClientSession` wraps `ClaudeSDKClient` for
multi-turn sessions that need explicit connect/disconnect, permission-mode
changes, model changes, MCP status/reconnect/toggle, interrupts, task stops, and
file rewind receipts.

`ClaudeToolPolicy` compiles SGP controls into Claude options: `allowed_tools`,
`disallowed_tools`, and a `can_use_tool` callback backed by an ordinary `Gate`.
The callback receives a `ClaudeToolRequestSignal` containing only the tool name,
MCP server name, input key names, and safe SDK context labels/reasons, so
permission receipts can prove policy decisions without storing raw tool inputs.

`ClaudeMeshMCPAdapter` maps `mesh.discover_tools()` and `mesh.call_tool()` to
MCP-style `initialize`, `tools/list`, and `tools/call` JSON-RPC handlers.
`ClaudeMeshMCPStdioServer` serves that adapter over newline-delimited stdio
JSON-RPC, writing only MCP messages to stdout.
`ClaudeMeshMCPHTTPApp` exposes the same adapter as a dependency-free ASGI app
for MCP Streamable HTTP POST requests that return `application/json`.

For local MCP clients, install the package and point the stdio command at a
factory in `module:attribute` form:

```bash
signal-gating-mcp my_app.mesh:build_mesh --server-name sgp
```

If the factory returns a `Mesh`, the command starts it before serving and stops
it on EOF. If the factory returns a `ClaudeMeshMCPAdapter`, the adapter is served
as supplied; use that form when the factory owns any custom lifecycle itself.
Factory, mesh lifecycle, and tool stdout are redirected to stderr so stdout stays
valid MCP JSON-RPC.

For HTTP-capable MCP clients, mount the ASGI app at a single endpoint such as
`/mcp`:

```python
from signal_gating import ClaudeMeshMCPAdapter, ClaudeMeshMCPHTTPApp, Mesh

mesh = Mesh([...])
mcp_app = ClaudeMeshMCPHTTPApp(ClaudeMeshMCPAdapter(mesh), path="/mcp")
```

`ClaudeMeshMCPHTTPApp` creates a secure `Mcp-Session-Id` on `initialize` and
requires that header on later POST, GET, and DELETE requests. It validates
present `Origin` headers, requires POST clients to accept both `application/json`
and `text/event-stream`, accepts notifications with `202 Accepted`, returns
`405` for unsupported GET SSE streams, and supports `DELETE` session
termination with `204 No Content`. It does not add a web server dependency; mount
it in the ASGI stack you already use. Claude Agent SDK should configure it as an
HTTP MCP server and allow its tools explicitly:

```python
mcp_servers = {
    "sgp": {"type": "http", "url": "https://example.com/mcp"},
}
allowed_tools = ["mcp__sgp__*"]
```

For protected HTTP MCP servers, pass `authorize_http` and
`protected_resource_metadata`. The metadata helper emits the RFC 9728 JSON
document at the well-known path derived from the MCP resource URL, for example
`https://example.com/.well-known/oauth-protected-resource/mcp` for
`https://example.com/mcp`, and the app advertises that URL through
`WWW-Authenticate` on `401` responses and on explicit bearer-error challenges
returned by the authorization hook.

```python
from signal_gating import ClaudeMCPProtectedResourceMetadata

mcp_app = ClaudeMeshMCPHTTPApp(
    ClaudeMeshMCPAdapter(mesh),
    authorize_http=validate_access_token,
    protected_resource_metadata=ClaudeMCPProtectedResourceMetadata(
        resource="https://example.com/mcp",
        authorization_servers=("https://auth.example.com",),
        scopes_supported=("tools.call",),
    ),
)
```

The hook receives a `ClaudeMCPHTTPAuthorizationContext` containing the bearer
token, origin, session id, method, path, and safe header names. It may return
`True`, a principal string, or `ClaudeMCPHTTPAuthorizationResult`. When the hook
is set, the app requires `Authorization: Bearer ...` on every HTTP request,
returns `401` / `403` authorization failures, and binds each `Mcp-Session-Id` to
the returned principal plus canonical audience, resource, and normalized scope
set. If no principal is returned, it falls back to a SHA-256 token fingerprint
without storing the raw token. Tokens sent through URI query strings are
rejected. The hook is the resource-server validation point: deployments should
validate issuer, expiry, audience/resource, and scopes.

When the backing mesh has `mesh.record(...)`, HTTP auth decisions are emitted as
`ClaudeMCPHTTPAuthorizationSignal` receipts with `event_kind="claude_mcp_http"`.
Actions distinguish `claude_mcp_http_auth_allowed`,
`claude_mcp_http_auth_challenged`, `claude_mcp_http_auth_denied`,
`claude_mcp_http_auth_session_mismatch`, and
`claude_mcp_http_auth_query_token_rejected`. The receipt payload includes
outcome, status, method, path, reason, bearer-token presence, principal/session
SHA-256 hashes, audience/resource presence, scope count, identity binding kind,
and whether protected-resource metadata was advertised. It does not include raw
bearer tokens, `Authorization` header values, raw principals, full
`Mcp-Session-Id` values, request bodies, or tool arguments.

To turn those receipts into auth metrics, filter by the Claude MCP auth event
kind and aggregate only sanitized fields:

```python
from collections import Counter

auth_receipts = recorder.filter_receipts(
    event_kinds="claude_mcp_http",
    signal_types="sgp.integrations.claude.mcp_http_authorization.v1",
    verify=True,
)
recorder.export_jsonl(
    "auth-receipts.jsonl",
    event_kinds="claude_mcp_http",
    signal_types="sgp.integrations.claude.mcp_http_authorization.v1",
    verify=True,
)

auth_metrics = {
    "actions": Counter(receipt.action for receipt in auth_receipts),
    "outcomes": Counter(receipt.payload["outcome"] for receipt in auth_receipts),
    "status_codes": Counter(receipt.payload["status_code"] for receipt in auth_receipts),
    "reasons": Counter(receipt.payload["reason"] for receipt in auth_receipts),
    "identity_bindings": Counter(
        receipt.payload["identity_binding_kind"] for receipt in auth_receipts
    ),
    "session_bound_requests": sum(
        1 for receipt in auth_receipts if receipt.payload["mcp_session_present"]
    ),
}
```

These counters are safe to export because they use statuses, reasons, counts,
booleans, and low-cardinality categories. Use principal/session hashes for
internal cardinality or mismatch analysis, not as public dashboard labels.
Filtering is namespace selection, not payload redaction: export only receipt
namespaces whose payloads are already safe for the target sink.

The same aggregate view is available from the CLI:

```bash
signal-gating-receipts auth runs.jsonl --pretty
signal-gating-receipts auth runs.jsonl --action claude_mcp_http_auth_denied --pretty
signal-gating-receipts auth runs.jsonl --otel --pretty
```

`auth` verifies receipt digests by default and pre-filters to
`event_kind="claude_mcp_http"` plus
`sgp.integrations.claude.mcp_http_authorization.v1`. It emits aggregate JSON
only: counts, presence totals, loaded/matched totals, and active filters. It
does not print raw receipts, raw payloads, bearer tokens, principals, session
IDs, request bodies, or tool arguments. Use `--no-verify` only for
known-tampered forensic inspection.

With `--otel`, `auth` sends only this aggregate metrics batch to
`OpenTelemetryReceiptMetricsExporter`. It does not convert raw receipts, receipt
`payload`, `wire`, or `metadata` into telemetry. Raw path values are omitted as
metric dimensions unless `--otel-include-paths` is explicitly supplied; when
enabled, `--otel-max-paths` caps the number of path labels and sends overflow to
`__other__`.

`ClaudeMCPBearerTokenValidator` is a ready-to-mount helper for that validation
boundary. It accepts a sync or async decoder for already-verified token claims,
checks issuer, expiry, not-before, audience, MCP resource, principal, and
required scopes, and returns MCP-compatible `401 invalid_token` or
`403 insufficient_scope` decisions without echoing token material. The
`ClaudeMCPJWTBearerAuthorizer` alias exposes the same class, and
`ClaudeMCPBearerTokenValidator.pyjwt(...)` lazy-loads `PyJWT` / JWKS support from
the optional `signal-gating[auth]` extra, enforcing JWT access-token `typ`
headers of `at+jwt` or `application/at+jwt` by default. For MCP deployments,
set `audience` and `resource` to the canonical MCP resource identifier; the
PyJWT helper uses `resource` as the default audience when `audience` is omitted.

```python
from signal_gating import ClaudeMCPBearerTokenValidator

authorize_http = ClaudeMCPBearerTokenValidator.pyjwt(
    jwks_url="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="https://example.com/mcp",
    resource="https://example.com/mcp",
    required_scopes=("tools.call",),
)
```

See `examples/claude_agent_sdk_mesh.py` for an offline, deterministic Claude
mesh example with fake SDK bindings and verifiable receipts.

This is not full Claude session replay. SGP receipts can correlate a run to the
Claude `session_id`, but resuming the Claude transcript belongs to the Claude
Agent SDK `resume` / `continue_conversation` / `session_store` mechanisms.

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
`payload`, event `metadata`, and a `digest` (sha256) so the record is
tamper-evident: `receipt.verify()`.

A trajectory is more than readable: its signal-carrying receipts are
**reconstructable**. Each receipt stores the full signal wire envelope, so a run
persisted to disk reloads as verifiable receipts and exact typed signals.

```python
reloaded = TrajectoryRecorder()
reloaded.load_jsonl("runs.jsonl")    # verifiable Receipts, after a restart
signals = reloaded.replay()          # -> [TaskSignal, ...], original types and ids
```

Use `filter_receipts(...)` or filtered `replay(...)` when a mixed trajectory file
contains audit namespaces you want to inspect separately. `signal_types` accepts
stable wire-type strings or `Signal` subclasses:

```python
auth_receipts = reloaded.filter_receipts(event_kinds="claude_mcp_http", verify=True)
reloaded.export_jsonl("auth-receipts.jsonl", event_kinds="claude_mcp_http", verify=True)
auth_signals = reloaded.replay(event_kinds="claude_mcp_http")
denials = reloaded.filter_receipts(actions="claude_mcp_http_auth_denied")
```

With `verify=True`, filtered export checks the selected receipt digests before
opening the output file.

For command-line inspection of mixed JSONL exports, use `summary` with explicit
filters:

```bash
signal-gating-receipts summary runs.jsonl \
  --event-kind claude_mcp_http \
  --signal-type sgp.integrations.claude.mcp_http_authorization.v1 \
  --action claude_mcp_http_auth_session_mismatch \
  --pretty
```

The CLI mirrors `filter_receipts(...)`: filters select receipt namespaces and
actions; output stays aggregate-only.

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

For legacy edge-hop-only capture, `mesh.intercept(recorder)` still works, but it
does not see direct orchestration paths.

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
