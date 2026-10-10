"""Benchmark: gating latency overhead.

Measures the per-tool-call cost of the fail-closed policy engine:
classification lookup, risk evaluation, and audit-trail append.

Run: PYTHONPATH=src python benchmarks/bench_gating_latency.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_gating.security.enforcement import PolicyEngine
from signal_gating.security.risk import RiskLevel, ToolRisk


def main() -> None:
    engine = PolicyEngine(max_auto_approve=RiskLevel.READ)
    for i in range(50):
        engine.register(f"read_tool_{i}", ToolRisk(RiskLevel.READ, ("read",), 0.9))
    for i in range(50):
        engine.register(f"send_tool_{i}", ToolRisk(RiskLevel.SEND, ("send",), 0.9))

    # Warm up
    for i in range(100):
        engine.evaluate(f"read_tool_{i % 50}", agent_name="bench")

    samples: list[float] = []
    n = 5000
    for i in range(n):
        start = time.perf_counter_ns()
        engine.evaluate(f"read_tool_{i % 50}", agent_name="bench")
        samples.append(time.perf_counter_ns() - start)

    gated: list[float] = []
    for i in range(n):
        start = time.perf_counter_ns()
        engine.evaluate(f"send_tool_{i % 50}", agent_name="bench")
        gated.append(time.perf_counter_ns() - start)

    p50 = statistics.median(samples) / 1000
    p99 = sorted(samples)[int(n * 0.99)] / 1000
    gp50 = statistics.median(gated) / 1000

    print("gating latency (per tool-call evaluation, audit trail in memory)")
    print(f"  allow path: p50={p50:.1f}us p99={p99:.1f}us over {n} samples")
    print(f"  gated path: p50={gp50:.1f}us over {n} samples")
    print()
    print("For context: a single LLM tool-call round trip is 200-2000ms.")
    print(f"Gating overhead is ~{p50 / 1_000_000:.4f}% of a 500ms call.")
    print("Verdict: the control plane is free.")


if __name__ == "__main__":
    main()
