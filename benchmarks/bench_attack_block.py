"""Benchmark: attack-scenario block rate.

Runs every documented attack scenario against the default fail-closed
policy engine and reports the block rate. Expected: 5/5.

Run: PYTHONPATH=src python benchmarks/bench_attack_block.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_gating.security.enforcement import PolicyEngine
from signal_gating.security.redteam import all_scenarios, run_all
from signal_gating.security.risk import RiskLevel


def main() -> None:
    engine = PolicyEngine(max_auto_approve=RiskLevel.READ)
    scenarios = all_scenarios()
    results = run_all(engine)

    print(f"attack scenarios: {len(scenarios)}")
    blocked = 0
    for scenario, result in zip(scenarios, results):
        mark = "BLOCKED" if result.blocked else "NOT BLOCKED"
        print(f"  [{mark}] {scenario.id}")
        print(f"           {result.detail}")
        blocked += result.blocked

    print()
    print(f"block rate: {blocked}/{len(scenarios)}")
    if blocked == len(scenarios):
        print("Verdict: every documented attack class is stopped by default policy.")
    else:
        print("Verdict: REGRESSION. The control plane has a hole.")
        sys.exit(1)


if __name__ == "__main__":
    main()
