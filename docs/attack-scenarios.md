# Synthetic tool-policy checks

Run the five deterministic policy fixtures:

```bash
PYTHONPATH=src python benchmarks/bench_attack_block.py
```

The fixture's **5/5** result means its five scripted expectations passed.
It is not an attack-success rate measured against adversarial agents.

The runner calls the policy engine directly or scans hardcoded source
strings. It does not invoke an LLM, connect to an MCP server, execute injected
tool output, or measure downstream external effects.

## What each fixture checks

| Fixture | Executed check | What remains untested |
| --- | --- | --- |
| Prompt injection via tool output | A registered SEND tool exceeds a READ approval ceiling. | Whether an agent follows injected text, uses an allowed capability to leak data, or completes legitimate work. |
| False MCP annotations | A source string containing a SQL mutation conflicts with a declared read-only hint. | Actual remote behavior, dynamically loaded code, obfuscated effects, or completeness of the heuristic scanner. |
| Data exfiltration | A static source string contains a network operation inconsistent with the fixture's declared behavior. | Whether a real tool transmits secrets. Legitimate remote reads also use networks. |
| Tool squatting | An unknown tool name is denied by the policy registry. | Replacement of an existing registered binding. Binding-identity checks elsewhere in the SDK need their own tests. |
| Unclassified tools | Forty unregistered names are denied. | Arbitrary malicious tools, compromised execution, or access to resources outside the mediated path. |

For example, this checks a risk policy, without executing an email tool:

```python
engine.register("send_email", ToolRisk(RiskLevel.SEND, ("outbound",), 0.9))
decision = engine.evaluate("send_email", arguments={"to": "attacker@example.com"})
assert decision in (Decision.DENY, Decision.REQUIRE_APPROVAL)
```

## Enforcement boundary

The default registry denies unregistered tools. Registered tools at or below
the configured ceiling can be allowed without an approval callback; higher
risk calls require one. Risk categories are not a complete principal,
resource, argument, or tenant authorization model.

When using `engine.as_gate()`, an approval callback receives a deep copy of the actual
tool arguments, and it can change the copy. Mutable built-in values are isolated
from the executing call; custom objects follow Python's `deepcopy` semantics.
An argument-sensitive policy must inspect the values it is approving; merely
registering a tool name is insufficient.

SGP's checks operate within the host application and its mediated delivery
paths. They do not sandbox Python code or replace scoped credentials and
authorization at the resource boundary. See [SECURITY.md](../SECURITY.md).

## Evaluate security and usefulness together

Existing systems also expose authorization and guardrail mechanisms:

- [MCP per-operation scope checks](https://ts.sdk.modelcontextprotocol.io/v2/serving/authorization#enforce-per-operation-scopes)
- [OpenAI Agents SDK tool guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- [A2A authentication and authorization](https://a2a-protocol.org/latest/specification/#7-authentication-and-authorization)

A security comparison must test matched threat models, actual forbidden
effects, legitimate task completion, and enforcement bypasses. Blocking every
outbound operation can satisfy a denial fixture while preventing useful work.
These local checks are regression fixtures, not evidence of general protection
against prompt injection, malicious MCP servers, or data exfiltration.
