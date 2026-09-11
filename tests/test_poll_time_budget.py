"""Poll waits share declared deadlines without treating expiry as process failure."""
import pytest

from test_process_manager import FakeClock, make_manager
from scripts.llm_solver.harness import time_budget


@pytest.mark.parametrize("scope", ["run", "parent_command", "captured_run"])
def test_poll_cannot_outwait_declared_budget(tmp_path, monkeypatch, scope):
    clock = FakeClock()
    monkeypatch.setattr(time_budget.time, "monotonic", clock.monotonic)
    budget = time_budget.command_time_budget if scope == "parent_command" else time_budget.run_time_budget
    with budget(.25):
        manager, factory, _, events = make_manager(tmp_path, clock=clock, poll_timeout_s=5)
        proc_id = manager.start("quiet").proc_id
        if scope != "captured_run":
            result = manager.poll(proc_id, timeout_s=4)
    try:
        if scope == "captured_run":
            result = manager.poll(proc_id, timeout_s=4)
        assert clock.now == pytest.approx(.25)
        assert result.running and result.timed_out and result.exit_code is None
        assert not result.result.timed_out  # wait expiry is not an execution timeout
        assert not factory.processes[0].terminated
        event = [e for e in events if e["event"] == "proc_poll"][-1]
        assert event["wait_budget"]["effective_seconds"] == pytest.approx(.25)
        assert event["wait_budget"]["elapsed_seconds"] == pytest.approx(.25)
    finally:
        manager.close()


def test_zero_additional_cap_honors_an_explicit_wait(tmp_path):
    manager, _, clock, _ = make_manager(tmp_path, poll_timeout_s=0)
    try:
        proc_id = manager.start("quiet").proc_id
        assert manager.poll(proc_id, timeout_s=.125).running
        assert clock.now == pytest.approx(.125)
        manager.poll(proc_id)  # no requested wait and no independent cap
        assert clock.now == pytest.approx(.125)
    finally:
        manager.close()


@pytest.mark.parametrize("state", ["buffered", "exited", "quiet"])
def test_exhausted_budget_reads_available_state_without_sleep(tmp_path, monkeypatch, state):
    clock = FakeClock()
    monkeypatch.setattr(time_budget.time, "monotonic", clock.monotonic)
    with time_budget.run_time_budget(1):
        manager, factory, _, _ = make_manager(tmp_path, clock=clock)
        try:
            proc_id = manager.start("quiet").proc_id
            if state == "buffered":
                factory.processes[0].write(b"available")
            elif state == "exited":
                factory.processes[0].finish(3)
            clock.now = 1
            result = manager.poll(proc_id, timeout_s=5)
            assert clock.now == 1
            assert result.running is (state != "exited")
            assert ("available" in result.result) is (state == "buffered")
        finally:
            manager.close()
