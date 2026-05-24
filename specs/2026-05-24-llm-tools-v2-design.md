# LLMAgent v2 — Autonomous Tool-Calling Across the Mesh

- **Date:** 2026-05-24
- **Status:** Design — approved, proceeding to plan
- **Repo:** `signalgatingprotocol/python-sdk` (branch `claude/llm-tools-v2`, off `main` @ the merged v1 `LLMAgent`)
- **Scope:** v2 of the autonomous-agent direction. Gives `LLMAgent` a tool-calling loop so it can reason, call tools exposed by other agents in its mesh, feed results back, and then emit — the multi-agent autonomy that single-agent runtimes (Hermes) and topology-only frameworks (LangGraph/CrewAI) don't combine with composable context gating.

## Problem

v1 `LLMAgent` does one LLM call and emits the reply. It cannot *act* — it can't invoke the tools other agents expose via `@agent.tool`. The SDK already has the bridge (`mesh.discover_tools()`, `mesh.call_tool()`, `agent.tools_schema()`), but nothing drives an LLM through it. v2 closes the perceive→**reason→act**→emit loop.

## Goal

When given a tool provider, an `LLMAgent` runs: call the model with available tool schemas → if it returns `tool_calls`, execute them and feed results back → repeat until the model returns a final answer → emit it. Tools are sourced **mesh-wide** through a small `ToolProvider` seam that the `Mesh` satisfies, keeping the agent decoupled and unit-testable with a fake provider (no live server).

## Non-goals (v2)

- Changing the base `Agent`, `Mesh`, `Signal`, or `Gate`. `MeshToolProvider` *wraps* the mesh's existing public API.
- A hard `openai` dependency in the core (`tool_calls` are parsed with stdlib `json`).
- Tool-name namespacing/collision resolution (v2 assumes unique tool names across the mesh; collisions raise a clear error).
- Streaming, parallel tool execution within a round (tools in a round run sequentially), or multi-turn memory beyond the single handler invocation.
- Gating *which* tools are exposed (signal gating on inputs is unchanged; tool-exposure gating is future work).

## The seam

```python
class ToolProvider(Protocol):
    def tool_schemas(self) -> list[dict[str, Any]]: ...        # OpenAI function-tool format
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...
```

### `MeshToolProvider`

Wraps a mesh (duck-typed via the imported `Mesh`), translating between the SDK's tool registry and the OpenAI tool contract:

- `tool_schemas()` — calls `mesh.discover_tools()`, converts each `ToolSpec` to an OpenAI function tool:
  ```python
  {"type": "function", "function": {
      "name": spec.name,
      "description": spec.description,
      "parameters": {
          "type": "object",
          "properties": {pname: {"type": _json_type(p.get("type"))} for pname, p in spec.parameters.items()},
          "required": [pname for pname, p in spec.parameters.items() if p.get("required")],
      },
  }}
  ```
  Builds a `name -> owner_agent_name` map while iterating. **Raises `ValueError` on a duplicate tool name** across agents (collision; namespacing deferred).
- `call_tool(name, arguments)` — looks up the owner, `await mesh.call_tool(owner, name, **arguments)`. (Mesh tool-calls route directly to the target's inbox and await the reply; no connect-edge required.)
- `_json_type` maps SDK param types to JSON Schema: `str→string`, `int→integer`, `float→number`, `bool→boolean`, anything else → `string`.

`MeshToolProvider` lives in `llm.py`; `mesh.py`/`agent.py` are untouched.

## LLMAgent changes

Constructor gains:
- `tools: ToolProvider | None = None`
- `max_tool_rounds: int = 4`

When `tools is None`, behavior is exactly v1 (backward compatible). When present, `_handle` runs the loop:

```python
async def _handle(self, signal):
    messages = [...]  # system?, user(render(signal))   (typed list[dict[str, Any]])
    if self._tools is None:
        # v1 path (single call, emit) — unchanged
        ...
        return
    schemas = self._tools.tool_schemas()
    for _ in range(self._max_tool_rounds):
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, tools=schemas, **extra
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = await self._tools.call_tool(tc.function.name, args)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            continue
        content = msg.content
        if not content or not content.strip():
            raise AgentError(self.name, f"LLM returned empty content for {type(signal).__name__}")
        await self.emit(self._build(signal, content))
        return
    raise AgentError(self.name, f"exceeded max_tool_rounds ({self._max_tool_rounds})")
```

### Protocol extension

The `LLMClient` Protocol stack widens to carry tool calls (still structural; `openai.AsyncOpenAI` satisfies it):
- `_ToolFunction` Protocol: `name: str`, `arguments: str`.
- `_ToolCall` Protocol: `id: str`, `function: _ToolFunction`.
- `_ChatMessage` gains `tool_calls: list[_ToolCall] | None` (alongside `content`).
- `_Completions.create` messages param widens from `list[dict[str, str]]` to `list[dict[str, Any]]`.

## Usage

```python
mesh = Mesh()
analyst = Agent("analyst")

@analyst.tool(description="Analyze a topic and return key points")
async def analyze(topic: str) -> dict:
    return {"points": [...]}

planner = LLMAgent(
    "planner", client=client, model="hermes-agent",
    tools=MeshToolProvider(mesh), on=Topic, emit=Plan,
)
mesh.add(analyst)
mesh.add(planner)
# planner now reasons and can call analyst.analyze before emitting its Plan.
```

Schemas are read lazily per handler call, so agents added after the `LLMAgent` is constructed are still discoverable.

## Error handling

- Exceeding `max_tool_rounds` without a final answer → `AgentError` → dead-lettered via the Agent's existing path.
- A tool that errors: `mesh.call_tool` raises `AgentError`, which propagates out of `_handle` → dead-lettered. (No swallowing.)
- Empty final content → `AgentError` (same as v1).

## Testing (CI-safe, no live server)

`tests/test_llm_tools.py`:
- **Scripted FakeClient** returning a `tool_calls` response on round 1 and final `content` on round 2; a **FakeProvider** exposing one tool. Assert: provider.call_tool invoked with the parsed args; a `role:"tool"` message was appended; the final `emit`-typed signal carries the content.
- **max_tool_rounds cap:** a FakeClient that always returns tool_calls → `pytest.raises(AgentError, match="max_tool_rounds")`.
- **No provider → v1 path:** with `tools=None`, one call, emits (no `tools=` passed to create).
- **MeshToolProvider unit:** a real `Mesh` with a tool-exposing `Agent`; `tool_schemas()` returns OpenAI-shaped entries with converted JSON types; `call_tool` routes to the owner and returns its result; duplicate tool name across two agents → `ValueError`.
- **Integration:** real `Mesh` + tool-exposing `Agent` + `LLMAgent(tools=MeshToolProvider(mesh))` with a scripted FakeClient (round 1 calls the tool, round 2 answers) → the tool actually executes and the final signal reaches a sink agent.
- **Regression:** the existing v1 `test_llm_agent.py` / `test_llm_mesh_integration.py` still pass; `import signal_gating` still pulls no `openai`.

## Files

| File | Change |
| --- | --- |
| `src/signal_gating/llm.py` | Extend: Protocol stack (`_ToolFunction`/`_ToolCall`/`tool_calls`), `ToolProvider`, `MeshToolProvider`, `_json_type`, LLMAgent `tools`/`max_tool_rounds` + loop. |
| `src/signal_gating/__init__.py` | Export `MeshToolProvider` (and `ToolProvider`); add to `__all__`. |
| `tests/test_llm_tools.py` | New — unit + integration tests above. |
| `README.md` | Extend the "LLM-backed agents (Hermes)" section with the tool-calling example. |

## Success criteria

1. An `LLMAgent` with a `MeshToolProvider` calls another agent's tool and emits a final, lineage-preserving signal — proven by the integration test, no live server.
2. `max_tool_rounds` bounds the loop; exceeding it dead-letters with a clear error.
3. v1 behavior unchanged when no provider is given; full suite + `ruff` + `mypy --strict` green; `import signal_gating` pulls no `openai`.
