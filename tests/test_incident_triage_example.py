from signal_gating.demo import run_scenario


async def test_incident_triage_quantifies_handler_load_reduction() -> None:
    result = await run_scenario()

    assert result.total_alerts == 12
    assert result.admitted_alerts == 3
    assert result.priority_drops == 6
    assert result.duplicate_drops == 3
    assert result.handler_load_reduction_percent == 75.0


async def test_incident_triage_preserves_every_unique_critical_incident() -> None:
    result = await run_scenario()

    assert result.critical_incidents_preserved == 3
    assert result.unique_critical_incidents == 3
    assert result.preserved_all_critical_incidents
