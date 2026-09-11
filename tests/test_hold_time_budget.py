"""The hold gate consumes the shared allowance, including setup time."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from llm_solver.harness import time_budget
from llm_solver.harness._loop import hold_gate
from llm_solver.harness.loop import solve_task


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=100.0, sleeps=[])
    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds
    source = SimpleNamespace(monotonic=lambda: state.now, sleep=sleep)
    monkeypatch.setattr(time_budget, "time", source)
    monkeypatch.setattr(hold_gate, "time", source)
    monkeypatch.delenv("YUJ_HOLD_TIMEOUT_S", raising=False)
    monkeypatch.setenv("YUJ_HOLD_UNTIL", "/fixture/signal")
    return state


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("timeout,expected,source", [
    (None, 1.0, "run_remaining"),
    ("10", 1.0, "run_remaining"),
    ("0.07", 0.07, "hold_limit"),
])
def test_wait_is_capped_by_remaining_run_and_explicit_hold(
    tmp_path, monkeypatch, clock, timeout, expected, source,
):
    if timeout is not None:
        monkeypatch.setenv("YUJ_HOLD_TIMEOUT_S", timeout)
    trace = tmp_path / "trace.jsonl"
    with time_budget.run_time_budget(5, source="owner"):
        clock.now += 4  # Setup has already used four seconds.
        with time_budget.run_time_budget(50, source="nested"):
            with patch.object(hold_gate.os.path, "exists", return_value=False):
                with pytest.raises(RuntimeError):
                    hold_gate.wait_for_hold(trace, MagicMock())
    assert sum(clock.sleeps) == pytest.approx(expected)
    start, end = events(trace)
    assert start["allocation"]["run_budget_source"] == "owner"
    assert start["allocation"]["effective_seconds"] == pytest.approx(expected)
    assert start["allocation"]["limiting_source"] == source
    assert end["elapsed_seconds"] == pytest.approx(expected)
    assert end["status"] == ("timed_out" if source == "hold_limit" else "run_exhausted")


def test_release_preserves_remaining_allowance_for_later_commands(tmp_path, clock):
    with time_budget.run_time_budget(5):
        clock.now += 4
        with patch.object(hold_gate.os.path, "exists", side_effect=lambda _: clock.now >= 104.1):
            hold_gate.wait_for_hold(tmp_path / "trace.jsonl", MagicMock())
        with time_budget.command_time_budget() as command:
            assert command.remaining() == pytest.approx(0.9)
    record = events(tmp_path / "trace.jsonl")[-1]
    assert record["status"] == "released"
    assert record["elapsed_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize("present", [False, True])
def test_exhausted_run_refuses_even_an_existing_signal(tmp_path, monkeypatch, clock, present):
    monkeypatch.setenv("YUJ_HOLD_TIMEOUT_S", "0")
    with time_budget.run_time_budget(1):
        clock.now += 1
        with patch.object(hold_gate.os.path, "exists", return_value=present):
            with pytest.raises(RuntimeError, match="run time budget exhausted"):
                hold_gate.wait_for_hold(tmp_path / "trace.jsonl", MagicMock())
    assert clock.sleeps == []
    assert events(tmp_path / "trace.jsonl")[-1]["status"] == "run_exhausted"


def test_no_declared_deadline_does_not_invent_a_hold_cap(tmp_path, clock):
    def delayed_signal(_):
        if not clock.sleeps:
            return False
        clock.now += 2000  # Simulated scheduling delay, not a real wait.
        return True
    with patch.object(hold_gate.os.path, "exists", side_effect=delayed_signal):
        hold_gate.wait_for_hold(tmp_path / "trace.jsonl", MagicMock())
    start, end = events(tmp_path / "trace.jsonl")
    assert start["allocation"]["effective_seconds"] is None
    assert start["allocation"]["limiting_source"] == "no_declared_limit"
    assert end["status"] == "released"
    assert end["elapsed_seconds"] > 1800


def test_signal_probe_cannot_release_after_using_remaining_run_time(tmp_path, clock):
    def slow_probe(_):
        clock.now += 2
        return True
    with time_budget.run_time_budget(1):
        with patch.object(hold_gate.os.path, "exists", side_effect=slow_probe):
            with pytest.raises(RuntimeError, match="run time budget exhausted"):
                hold_gate.wait_for_hold(tmp_path / "trace.jsonl", MagicMock())
    assert not clock.sleeps
    assert events(tmp_path / "trace.jsonl")[-1]["status"] == "run_exhausted"


def test_expired_parent_command_cannot_be_bypassed_by_zero_hold(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("YUJ_HOLD_TIMEOUT_S", "0")
    with time_budget.command_time_budget(1):
        clock.now += 1
        with patch.object(hold_gate.os.path, "exists", return_value=True):
            with pytest.raises(RuntimeError):
                hold_gate.wait_for_hold(tmp_path / "trace.jsonl", MagicMock())
    assert not clock.sleeps


def test_driver_setup_and_hold_share_one_deadline(tmp_path, clock):
    from llm_solver.harness._loop import driver
    from llm_solver._shared.telemetry_paths import trace_path
    (tmp_path / "prompt.txt").write_text("Task")
    client = MagicMock()
    original = driver.load_transforms_and_estimator
    def setup(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += 3
        return result
    with patch.object(driver, "load_transforms_and_estimator", side_effect=setup):
        with pytest.raises(RuntimeError, match="run time budget exhausted"):
            solve_task(tmp_path, make_config(task_wall_clock_limit_s=5), client)
    assert sum(clock.sleeps) == pytest.approx(2)
    client.chat.assert_not_called()
    record = events(trace_path(tmp_path))[-1]
    assert record["allocation"]["effective_seconds"] == 2
    assert record["run_remaining_seconds"] == 0
