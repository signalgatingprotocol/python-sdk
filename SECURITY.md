# Security Policy

## Supported versions

The SDK is pre-1.0. Security fixes are applied to the latest `0.1.x` release
and to `main`. Older commits and unreleased branches are not supported.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1.0 | No |

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's private vulnerability
reporting for this repository:

https://github.com/signalgatingprotocol/python-sdk/security/advisories/new

Do not open a public issue for a suspected vulnerability, and do not include
proof-of-concept exploits against third-party systems in a report.

A useful report includes:

- The affected version or commit.
- A minimal reproduction: the smallest signal, gate, policy, or tool
  registration that triggers the problem.
- The impact as you see it: which guarantee fails (for example, a tool call
  that should have been denied was allowed, an audit record was lost, or a
  policy rule was bypassed).

You will get an acknowledgement, and a fix or a documented rationale if the
report is not accepted as a vulnerability. Please give the maintainers a
reasonable window to ship a fix before any public disclosure.

## Scope

In scope, in priority order:

- The fail-closed authorization layer (`signal_gating.security`): any path
  where an unclassified, denied, or approval-gated tool call executes
  without the required decision.
- The policy loader (`signal_gating.policy`): YAML policies that evaluate
  differently from their written rules.
- Signal deserialization: reconstructing typed signals from untrusted JSON
  must never execute code or instantiate unintended classes.
- Audit trail integrity: decisions that are enforced but not recorded.
- The experimental executive kernel (`signal_gating.executive`): mediated
  dispatch without current authority, resource scope, freshness or budget
  admission; unauthenticated owner control; cross-goal artifact access;
  forged completion evidence; loss of recorded uncertainty or unsafe blind
  retries after an unknown consequential effect.

Out of scope:

- Vulnerabilities in applications built on the SDK, including tools an
  application registers and the side effects inside those tools. The SDK
  gates calls; each tool remains responsible for validating its own inputs
  and protecting its own secrets.
- Denial of service through unbounded local inputs supplied by the host
  application itself.
- Findings that require a malicious host process; the SDK is a library and
  shares its host's trust boundary.

## Executive mediation boundary

The kernel is the sole writer of its authoritative SQLite journal and derived
state. Workers produce typed proposals, not authority. Model confidence and
artifact text cannot grant permissions. Current bindings, grants, constraints,
read revisions and reservations are checked before registered calls. External
calls occur outside database transactions. A successful external effect can
precede its durable receipt; unknown effects need reconciliation or explicit
owner resolution. This is not an exactly-once guarantee.

The foreground CLI trusts local filesystem access as owner access. Protect the
configuration, store, private context/artifact bytes and local credentials with
host permissions. `inspect` is local privileged metadata access; it is broader
than the redacted observation-only export. Neither is a remote authentication
or delegation protocol. Export checks the exact current persisted owner and
current authorized artifact validity/integrity; its redacted grant/binding
metadata cannot confer execution authority or restore a store.

Configured local check programs and explicitly supplied legacy callables are
trusted host code. Exact argv, path checks, minimal subprocess environments,
timeouts and output bounds do not sandbox what those programs can do. A worker
or application that independently performs calls outside the registered
adapters is outside mediation. Provider billing and independent host activity
are not controlled by local budget accounting. Unknown usage remains held;
deterministic fixtures and constructor-only SDK checks do not verify paid/live
provider behavior or task quality.

Outbound email in Preston's environment is human-only. Never configure a check
or adapter to send, schedule or queue it on the owner's behalf. See the
[operator guide](docs/executive-kernel.md) for supported local workflows,
recovery semantics and explicitly deferred adaptive/federation behavior.
