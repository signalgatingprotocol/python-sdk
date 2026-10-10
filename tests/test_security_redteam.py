"""Tests: every documented attack scenario must be blocked."""

from signal_gating.security.enforcement import PolicyEngine
from signal_gating.security.redteam import (
    all_scenarios,
    run_all,
    run_scenario,
)
from signal_gating.security.risk import RiskLevel


def _engine():
    return PolicyEngine(max_auto_approve=RiskLevel.READ)


def test_all_scenarios_blocked():
    engine = _engine()
    results = run_all(engine)
    assert len(results) == len(all_scenarios()) == 5
    for r in results:
        assert r.blocked, f"REGRESSION: scenario {r.scenario_id} not blocked: {r.detail}"


def test_individual_scenarios():
    engine = _engine()
    for scenario in all_scenarios():
        result = run_scenario(scenario.id, _engine())
        assert result.blocked, scenario.id


def test_unknown_scenario_raises():
    try:
        run_scenario("not-a-scenario", _engine())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
