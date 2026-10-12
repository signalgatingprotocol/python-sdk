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

Out of scope:

- Vulnerabilities in applications built on the SDK, including tools an
  application registers and the side effects inside those tools. The SDK
  gates calls; each tool remains responsible for validating its own inputs
  and protecting its own secrets.
- Denial of service through unbounded local inputs supplied by the host
  application itself.
- Findings that require a malicious host process; the SDK is a library and
  shares its host's trust boundary.
