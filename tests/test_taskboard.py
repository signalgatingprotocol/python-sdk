from signal_gating import Signal
from signal_gating.errors import BudgetExceeded, TaskRejected, TeamError
from signal_gating.trajectory import domain_payload


def test_error_hierarchy():
    err = TaskRejected("t1", "no_empty_results")
    assert err.task_id == "t1" and err.gate_name == "no_empty_results"
    assert BudgetExceeded(1000, "k").budget == 1000
    assert issubclass(TeamError, Exception)


def test_domain_payload_excludes_envelope():
    class Probe(Signal):
        text: str = ""

    assert domain_payload(Probe(text="x", priority=9)) == {"text": "x"}
