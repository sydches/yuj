"""Deterministic tests for the session-scoped background process kernel."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

from scripts.llm_solver.harness.process_manager import (
    ProcessManager,
    ProcessManagerError,
    ReplayProcessManager,
    build_background_sandbox_argv,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            callback, self.on_sleep = self.on_sleep, None
            callback()


class FakeProcess:
    pid = 0  # makes ProcessManager use terminate()/kill(), not os.killpg()

    def __init__(self, output_fd: int) -> None:
        self._output = os.fdopen(os.dup(output_fd), "wb", buffering=0)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def write(self, data: bytes) -> None:
        self._output.write(data)

    def finish(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._output.close()

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -signal.SIGTERM
        self._output.close()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -signal.SIGKILL
        if not self._output.closed:
            self._output.close()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class FakePopenFactory:
    def __init__(self) -> None:
        self.calls = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        process = FakeProcess(kwargs["stdout"].fileno())
        self.processes.append(process)
        return process


def make_manager(tmp_path: Path, **overrides):
    events = overrides.pop("events", [])
    factory = overrides.pop("factory", FakePopenFactory())
    clock = overrides.pop("clock", FakeClock())
    manager = ProcessManager(
        run_dir=tmp_path / "run",
        cwd=tmp_path,
        argv_builder=lambda command: ["sandbox", "bash", "-c", command],
        max_procs=overrides.pop("max_procs", 2),
        poll_timeout_s=overrides.pop("poll_timeout_s", 5),
        admit_output=overrides.pop("admit_output", None),
        event_sink=events.append,
        popen_factory=factory,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_s=overrides.pop("poll_interval_s", 1),
        **overrides,
    )
    return manager, factory, clock, events


@pytest.mark.parametrize('mismatch', [False, True])
def test_guarded_background_process_preserves_task_output_and_refuses_changed_credentials(
    tmp_path, native_process_identity, mismatch,
):
    from dataclasses import replace
    from scripts.llm_solver.harness.process_identity import guarded_process_argv
    identity = native_process_identity
    if mismatch:
        identity = replace(identity, uids=(identity.uids[0] + 1,) * 4)
    events = []
    prefix = ['bash', '-c', 'printf transport-warning >&2; exec "$@"', 'fixture']
    manager = ProcessManager(
        run_dir=tmp_path / 'run', cwd=tmp_path, max_procs=1, poll_timeout_s=1,
        argv_builder=lambda command: guarded_process_argv(
            prefix, ['bash', '-c', command], identity, combine_output=True),
        event_sink=events.append,
    )
    def check_metadata(proc_id, exit_code):
        assert not mismatch, 'unverified startup must not supply completed-check evidence'
        return {'verification_status': 'fixture-check'}
    manager.poll_metadata = check_metadata
    try:
        started = manager.start('touch ran; printf out; printf err >&2; exit 77')
        manager._records[started.proc_id].process.wait(timeout=3)
        poll = manager.poll(started.proc_id)
        assert poll.exit_code == 77 and not poll.running
        assert 'yuj-process-identity-v1' not in poll.result
        assert (tmp_path / 'ran').exists() is not mismatch
        if mismatch:
            assert 'did not start with verified process credentials' in poll.result
            assert poll.result.verification_status == 'process_identity_unverified'
            assert 'transport-warning' not in poll.result
        else:
            assert 'outerr' in poll.result and 'transport-warning' in poll.result
            assert events[-1]['execution_metadata']['process_identity_verified'] is True
            again = manager.poll(started.proc_id)
            assert 'outerr' not in again.result and 'transport-warning' not in again.result
    finally:
        manager.close()


def test_background_partial_frame_and_admission_retry_do_not_lose_output(tmp_path):
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv, VERIFIED
    manager, factory, _clock, _events = make_manager(tmp_path)
    manager.argv_builder = lambda command: GuardedProcessArgv(['fixture', command])
    proc_id = manager.start('fixture').proc_id
    process = factory.processes[0]
    try:
        process.write(VERIFIED[:4])
        partial = manager.poll(proc_id, timeout_s=0)
        assert partial.cursor_end == 0 and partial.timed_out
        assert 'yuj-' not in partial.result
        process.write(VERIFIED[4:] + VERIFIED + b'payload')
        process.finish()
        def reject(_text):
            raise ValueError('admission failed')
        manager.admit_output = reject
        with pytest.raises(ValueError, match='admission failed'):
            manager.poll(proc_id)
        manager.admit_output = lambda text: text
        accepted = manager.poll(proc_id)
        assert accepted.result.count('yuj-process-identity-v1') == 1
        assert 'payload' in accepted.result
        assert accepted.cursor_end == len(VERIFIED) * 2 + len(b'payload')
    finally:
        manager.close()


def test_guarded_start_preserves_an_existing_transport_log(tmp_path):
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv
    manager, _factory, _clock, _events = make_manager(tmp_path)
    manager.argv_builder = lambda command: GuardedProcessArgv(['fixture', command])
    existing = manager.procs_dir / 'p0001.stderr'
    existing.write_bytes(b'previous transport evidence')
    try:
        with pytest.raises(ProcessManagerError, match='could not start'):
            manager.start('fixture')
        assert existing.read_bytes() == b'previous transport evidence'
    finally:
        manager.close()


def test_final_frame_write_is_read_after_exit_is_observed(tmp_path):
    from scripts.llm_solver.harness.process_identity import GuardedProcessArgv, VERIFIED
    manager, factory, _clock, _events = make_manager(tmp_path)
    manager.argv_builder = lambda command: GuardedProcessArgv(['fixture', command])
    proc_id = manager.start('fixture').proc_id
    process = factory.processes[0]
    def finish_on_poll():
        if process.returncode is None:
            process.write(VERIFIED + b'final output')
            process.finish()
        return process.returncode
    process.poll = finish_on_poll
    try:
        result = manager.poll(proc_id, timeout_s=0)
        assert 'final output' in result.result and not result.running
        assert 'verified process credentials' not in result.result
    finally:
        manager.close()


@pytest.mark.parametrize("name", ["poll_timeout_s", "poll_interval_s", "terminate_grace_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_manager_rejects_nonfinite_durations_before_creating_artifacts(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        make_manager(tmp_path, **{name: value})

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("output_ready", [False, True])
def test_poll_rejects_nonfinite_wait_without_consuming_output_or_stopping_process(tmp_path, value, output_ready):
    manager, factory, clock, events = make_manager(tmp_path)
    proc_id = manager.start("fake background command").proc_id
    process = factory.processes[0]
    if output_ready:
        process.write(b"retained output")

    def unexpected_sleep():
        pytest.fail("invalid poll reached the wait loop")

    clock.on_sleep = unexpected_sleep
    try:
        with pytest.raises(ProcessManagerError, match="poll timeout"):
            manager.poll(proc_id, timeout_s=value)

        assert clock.now == 0
        assert [event["event"] for event in events] == ["proc_start"]
        assert process.poll() is None
        assert not process.terminated and not process.killed
        if not output_ready:
            process.write(b"retained output")
        result = manager.poll(proc_id, timeout_s=0)
        assert "retained output" in result.result
        assert result.cursor_start == 0
        assert result.running is True
    finally:
        manager.close()


def test_fractional_poll_wait_expires_without_ending_process(tmp_path):
    manager, factory, clock, _events = make_manager(tmp_path, poll_timeout_s=0.25)
    proc_id = manager.start("fake background command").proc_id
    try:
        result = manager.poll(proc_id, timeout_s=0.125)
        assert clock.now == 0.125
        assert result.timed_out is True
        assert result.running is True
        assert result.result.timed_out is False
        assert not factory.processes[0].terminated
    finally:
        manager.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_session_dispatch_reports_invalid_poll_wait_without_waiting(tmp_path, value):
    from types import SimpleNamespace

    from _config_helpers import make_config
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.harness.tools import dispatch

    cfg = make_config(tools_background_enabled=True, sandbox_bash=False)
    session = Session(cfg, SimpleNamespace(), "system", "task", str(tmp_path))
    session._process_manager.close()
    manager, factory, clock, _events = make_manager(tmp_path)
    session._process_manager = manager
    proc_id = manager.start("fake background command").proc_id
    try:
        result = dispatch(
            "bash_poll", {"proc_id": proc_id, "timeout_s": value},
            cwd=str(tmp_path), cfg=cfg, tool_registry=session._tool_registry,
        )
        assert "ERROR: poll timeout must be finite and >= 0" in result
        assert clock.now == 0
        assert factory.processes[0].poll() is None
    finally:
        manager.close()


def test_start_is_nonblocking_sandboxed_and_logs_under_run_dir(tmp_path):
    manager, factory, _clock, events = make_manager(tmp_path)

    started = manager.start("slow-build --serve")

    assert started.proc_id == "p0001"
    assert started.log_path == ".procs/p0001.log"
    assert (tmp_path / "run" / started.log_path).is_file()
    argv, kwargs = factory.calls[0]
    assert argv == ["sandbox", "bash", "-c", "slow-build --serve"]
    assert kwargs["cwd"] == str(tmp_path.resolve())
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert events[0]["event"] == "proc_start"
    assert events[0]["proc_id"] == "p0001"
    assert "/home/" not in str(events[0])


def test_poll_returns_only_new_output_and_traces_admitted_bytes(tmp_path):
    admitted = []

    def admit(text: str) -> str:
        admitted.append(text)
        return f"<admitted>{text}</admitted>"

    manager, factory, _clock, events = make_manager(
        tmp_path, admit_output=admit
    )
    proc_id = manager.start("server").proc_id
    process = factory.processes[0]
    process.write(b"ready\n")

    first = manager.poll(proc_id, timeout_s=0)
    process.write(b"request complete\n")
    process.finish(0)
    second = manager.poll(proc_id, timeout_s=0)

    assert "ready\n" in first.result
    assert "request complete" not in first.result
    assert first.running is True
    assert first.cursor_start == 0
    assert first.cursor_end == 6
    assert "request complete\n" in second.result
    assert "ready" not in second.result
    assert second.running is False
    assert second.exit_code == 0
    poll_events = [event for event in events if event["event"] == "proc_poll"]
    assert [event["result"] for event in poll_events] == [first.result, second.result]
    assert admitted == [
        "ready\n[background process p0001: running]",
        "request complete\n[background process p0001: exited (0)]",
    ]


def test_empty_poll_wait_is_capped_and_does_not_advance_cursor(tmp_path):
    manager, _factory, clock, events = make_manager(
        tmp_path, poll_timeout_s=2
    )
    proc_id = manager.start("server").proc_id

    result = manager.poll(proc_id, timeout_s=99)

    assert clock.now == 2
    assert result.timed_out is True
    assert result.running is True
    assert result.cursor_start == result.cursor_end == 0
    assert events[-1]["timed_out"] is True


def test_poll_wakes_when_fake_process_produces_output(tmp_path):
    manager, factory, clock, _events = make_manager(tmp_path)
    proc_id = manager.start("server").proc_id
    clock.on_sleep = lambda: factory.processes[0].write(b"booted")

    result = manager.poll(proc_id)

    assert clock.now == 1
    assert "booted" in result.result
    assert result.timed_out is False


def test_max_proc_limit_counts_only_live_processes(tmp_path):
    manager, factory, _clock, _events = make_manager(tmp_path, max_procs=1)
    manager.start("one")
    with pytest.raises(ProcessManagerError, match="limit reached"):
        manager.start("two")

    factory.processes[0].finish(0)
    assert manager.start("two").proc_id == "p0002"


def test_explicit_kill_and_close_kill_every_live_process(tmp_path):
    manager, factory, _clock, events = make_manager(tmp_path)
    first = manager.start("one").proc_id
    second = manager.start("two").proc_id

    killed = manager.kill(first)
    manager.close()
    manager.close()

    assert killed.was_running is True
    assert killed.result == f"Killed background process {first}"
    assert killed.exit_code == -signal.SIGTERM
    assert factory.processes[0].terminated is True
    assert factory.processes[1].terminated is True
    kill_events = [event for event in events if event["event"] == "proc_kill"]
    assert [(event["proc_id"], event["reason"]) for event in kill_events] == [
        (first, "explicit"),
        (second, "session_end"),
    ]
    with pytest.raises(ProcessManagerError, match="closed"):
        manager.start("three")


@pytest.mark.parametrize("failure", ["wait_timeout", "signal_error"])
def test_kill_does_not_claim_exit_when_cleanup_is_unconfirmed(tmp_path, monkeypatch, failure):
    manager, factory, _clock, events = make_manager(tmp_path)
    proc_id = manager.start("server").proc_id
    process = factory.processes[0]
    signals = []

    def signal_process(_process, sig):
        signals.append(sig)
        if failure == "signal_error":
            raise PermissionError("cannot signal process")

    monkeypatch.setattr(manager, "_signal", signal_process)
    try:
        result = manager.kill(proc_id)
        assert result.was_running and result.exit_code is None
        assert result.result == (
            f"Requested termination of background process {proc_id}; exit not confirmed"
        )
        assert process.poll() is None
        assert events[-1]["result"] == result.result
        assert events[-1]["exit_code"] is None
        assert signals == ([signal.SIGTERM, signal.SIGKILL]
                           if failure == "wait_timeout" else [signal.SIGTERM])
    finally:
        process.finish()
        manager.close()


def test_context_manager_kills_process_on_exception(tmp_path):
    manager, factory, _clock, events = make_manager(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        with manager:
            manager.start("server")
            raise RuntimeError("boom")

    assert factory.processes[0].terminated is True
    assert events[-1]["event"] == "proc_kill"
    assert events[-1]["reason"] == "session_end"


def test_replay_returns_exact_poll_result_without_launching_process(tmp_path):
    manager, factory, _clock, events = make_manager(
        tmp_path, admit_output=lambda text: f"ADMITTED:{text}"
    )
    command = "server --slow"
    proc_id = manager.start(command).proc_id
    factory.processes[0].write(b"exact bytes\n")
    live_poll = manager.poll(proc_id, timeout_s=0)
    manager.close()

    replay = ReplayProcessManager(events)
    replay_start = replay.start(command)
    replay_poll = replay.poll(replay_start.proc_id)
    replay.close()

    assert replay_poll == live_poll
    assert replay.consumed_all is True


def test_replay_rejects_changed_command_or_corrupt_poll(tmp_path):
    manager, _factory, _clock, events = make_manager(tmp_path)
    proc_id = manager.start("original").proc_id
    manager.poll(proc_id, timeout_s=0)

    with pytest.raises(ProcessManagerError, match="command digest mismatch"):
        ReplayProcessManager(events).start("different")

    corrupt = [dict(event) for event in events]
    corrupt[-1]["result"] = "changed"
    replay = ReplayProcessManager(corrupt)
    started = replay.start("original")
    with pytest.raises(ProcessManagerError, match="output digest mismatch"):
        replay.poll(started.proc_id)


def test_background_sandbox_argv_uses_normal_bwrap_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)

    argv = build_background_sandbox_argv(
        "server",
        cwd=str(tmp_path),
        bwrap_bin="/usr/bin/bwrap",
        sandbox_required=True,
    )

    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert "--die-with-parent" in argv
    assert "--clearenv" in argv
    assert argv[-7:] == [
        "bash", "--noprofile", "--norc", "-o", "pipefail", "-c", "server",
    ]
