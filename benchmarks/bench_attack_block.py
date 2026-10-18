"""Run five synthetic policy and source-audit fixtures.

Expected: five scripted checks pass. This does not execute adversarial
agents or establish protection against an attack class.

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

    print(f"synthetic policy fixtures: {len(scenarios)}")
    blocked = 0
    for scenario, result in zip(scenarios, results):
        mark = "PASS" if result.blocked else "FAIL"
        print(f"  [{mark}] {scenario.id}")
        print(f"           {result.detail}")
        blocked += result.blocked

    print()
    print(f"fixture checks passed: {blocked}/{len(scenarios)}")
    if blocked == len(scenarios):
        print("All scripted expectations passed; adversarial effectiveness is unmeasured.")
    else:
        print("REGRESSION: at least one scripted policy expectation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
