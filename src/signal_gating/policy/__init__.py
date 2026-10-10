"""Policy as code: declarative, versioned, auditable authorization policy.

A policy file declares what agents may do. It is checked into version
control, reviewed like code, and loaded at startup. Example:

.. code-block:: yaml

    version: 1
    name: production-agents
    default: deny
    max_auto_approve: READ
    rules:
      - match:
          tools: ["read_*", "search_*"]
        allow: true
      - match:
          risk: ["SEND", "DESTRUCTIVE"]
        require_approval: true
        approver: human-in-the-loop
      - match:
          agents: ["researcher"]
          tools: ["write_draft"]
        allow: true

Semantics: rules are evaluated in order; the first matching rule wins.
If no rule matches, ``default`` applies (deny unless explicitly set
to allow, which the loader warns about loudly).
"""

from signal_gating.policy.loader import (
    Policy,
    PolicyRule,
    load_policy,
    load_policy_from_dict,
)

__all__ = ["Policy", "PolicyRule", "load_policy", "load_policy_from_dict"]
