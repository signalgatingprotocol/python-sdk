"""Measure how gating cuts alert noise without hiding critical incidents.

Run with:

    python examples/incident_triage.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from signal_gating.core import Agent, Gate, Mesh, Signal


class IncidentAlert(Signal):
    service: str
    fingerprint: str
    summary: str


@dataclass(frozen=True, slots=True)
class TriageResult:
    total_alerts: int
    admitted_alerts: int
    priority_drops: int
    duplicate_drops: int
    unique_critical_incidents: int
    critical_incidents_preserved: int

    @property
    def context_reduction_percent(self) -> float:
        return round((1 - self.admitted_alerts / self.total_alerts) * 100, 1)

    @property
    def preserved_all_critical_incidents(self) -> bool:
        return self.critical_incidents_preserved == self.unique_critical_incidents


def incident_stream() -> tuple[IncidentAlert, ...]:
    """Return a deterministic mix of low-priority noise and repeated incidents."""
    low_priority = tuple(
        IncidentAlert(
            service="checkout",
            fingerprint=f"informational-{index}",
            summary=f"Informational event {index}",
            priority=2,
        )
        for index in range(6)
    )
    critical = tuple(
        IncidentAlert(
            service=service,
            fingerprint=fingerprint,
            summary=summary,
            priority=9,
        )
        for service, fingerprint, summary in (
            ("checkout", "payments-down", "Payment authorization is failing"),
            ("identity", "login-errors", "Login error rate is elevated"),
            ("orders", "queue-stalled", "Order queue is not draining"),
        )
    )
    repeated_critical = tuple(alert.evolve() for alert in critical)
    return low_priority + critical + repeated_critical


async def run_scenario() -> TriageResult:
    """Run the incident stream through real SDK gates and return measured outcomes."""
    alerts = incident_stream()
    admitted: list[IncidentAlert] = []
    worker = Agent(
        "incident-triage",
        gates=[Gate.by_priority(7), Gate.deduplicate(window=60)],
    )

    @worker.on(IncidentAlert)
    async def record_admitted(alert: IncidentAlert) -> None:
        admitted.append(alert)

    mesh = Mesh([worker])
    async with mesh:
        for alert in alerts:
            await mesh.inject(worker, alert)

    spans = mesh.tracer.get_agent_spans(worker.name)
    priority_drops = sum(
        span.gate == "priority_filter" and span.action == "rejected" for span in spans
    )
    duplicate_drops = sum(
        span.gate == "dedup" and span.action == "rejected" for span in spans
    )
    critical_fingerprints = {alert.fingerprint for alert in alerts if alert.priority >= 7}
    admitted_fingerprints = {alert.fingerprint for alert in admitted}

    return TriageResult(
        total_alerts=len(alerts),
        admitted_alerts=len(admitted),
        priority_drops=priority_drops,
        duplicate_drops=duplicate_drops,
        unique_critical_incidents=len(critical_fingerprints),
        critical_incidents_preserved=len(critical_fingerprints & admitted_fingerprints),
    )


async def main() -> None:
    result = await run_scenario()
    print(f"signals_in={result.total_alerts}")
    print(f"signals_admitted={result.admitted_alerts}")
    print(f"context_reduction={result.context_reduction_percent:.1f}%")
    print(f"priority_drops={result.priority_drops}")
    print(f"duplicate_drops={result.duplicate_drops}")
    print(
        "critical_incidents_preserved="
        f"{result.critical_incidents_preserved}/{result.unique_critical_incidents}"
    )


if __name__ == "__main__":
    asyncio.run(main())
