"""Bounded PTY process, session, config, and replay coverage for issue #63."""
from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness.approvals import approval_decision
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.terminal_process import (
    ReplayTerminalProcessManager,
    TerminalProcessError,
    TerminalProcessManager,
)
from scripts.llm_solver.harness.tools import dispatch

from _config_helpers import make_config


def _command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(source)}"


def _manager(tmp_path: Path, **overrides):
    events: list[dict[str, object]] = []
    manager = TerminalProcessManager(
        run_dir=tmp_path / "run",
        cwd=tmp_path,
        argv_builder=lambda command: [
            "/bin/bash", "--noprofile", "--norc", "-c", command,
        ],
        read_timeout_s=overrides.pop("read_timeout_s", 1.0),
        max_lifetime_s=overrides.pop("max_lifetime_s", 5.0),
        max_output_bytes=overrides.pop("max_output_bytes", 4096),
        max_input_chars=overrides.pop("max_input_chars", 64),
        event_sink=events.append,
        **overrides,
    )
    return manager, events


def _read_until(
    manager: TerminalProcessManager,
    terminal_id: str,
    needle: str,
    *,
    attempts: int = 8,
) -> tuple[str, object]:
    output = ""
    latest = None
    for _ in range(attempts):
        latest = manager.read(terminal_id, timeout_s=0.5)
        output += latest.result
        if needle in output and not latest.running:
            return output, latest
    raise AssertionError(f"terminal output never reached {needle!r}: {output!r}")


def _names(cfg, client=None) -> tuple[str, ...]:
    schemas = apply_profile_to_schemas(
        get_tool_schemas(), cfg, client or SimpleNamespace()
    )
    return tuple(item["function"]["name"] for item in schemas)


def test_terminal_config_is_opt_in_bounded_and_validated(tmp_path):
    defaults = load_config()
    assert defaults.tools_terminal_enabled is False
    assert defaults.tools_terminal_read_timeout == 5
    assert defaults.tools_terminal_max_lifetime == 900
    assert defaults.tools_terminal_max_output_bytes == 1_000_000
    assert defaults.tools_terminal_max_input_chars == 16_384

    overlay = tmp_path / "terminal.toml"
    overlay.write_text(
        "[tools]\n"
        "terminal_enabled = true\n"
        "terminal_read_timeout = 1.5\n"
        "terminal_max_lifetime = 30\n"
        "terminal_max_output_bytes = 2048\n"
        "terminal_max_input_chars = 128\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_terminal_enabled is True
    assert configured.tools_terminal_read_timeout == 1.5
    assert configured.tools_terminal_max_lifetime == 30
    assert configured.tools_terminal_max_output_bytes == 2048
    assert configured.tools_terminal_max_input_chars == 128

    for body, match in (
        ('terminal_enabled = "yes"', "terminal_enabled"),
        ('terminal_read_timeout = "slow"', "terminal_read_timeout"),
        ("terminal_read_timeout = -1", "terminal_read_timeout"),
        ("terminal_max_lifetime = 0", "terminal_max_lifetime"),
        ("terminal_max_output_bytes = 0", "terminal_max_output_bytes"),
        ("terminal_max_input_chars = 1.5", "terminal_max_input_chars"),
    ):
        overlay.write_text(f"[tools]\n{body}\n")
        with pytest.raises(ValueError, match=match):
            load_config(user_config=overlay)


def test_terminal_surface_is_assistant_only_and_preserves_core_under_cap():
    disabled = make_config(runtime_mode="assistant", tools_terminal_enabled=False)
    measurement = make_config(
        runtime_mode="measurement", tools_terminal_enabled=True
    )
    assistant = make_config(runtime_mode="assistant", tools_terminal_enabled=True)
    assert "terminal_start" not in _names(disabled)
    assert "terminal_start" not in _names(measurement)
    assert {"terminal_start", "terminal_io"} <= set(_names(assistant))

    profile = SimpleNamespace(
        max_tools=8, simplify_schemas=False, edit_format="", name="bounded"
    )
    capped = set(_names(assistant, SimpleNamespace(profile=profile)))
    assert {"terminal_start", "terminal_io"} <= capped
    assert {"bash", "read", "write", "edit", "done"} <= capped
    assert len(capped) == 8


def test_pty_input_output_exit_and_evidence_are_explicit(tmp_path):
    manager, events = _manager(tmp_path)
    source = (
        "import os,sys; "
        "print(f'tty={os.isatty(0)},{os.isatty(1)}', flush=True); "
        "print('name?', flush=True); "
        "name=input(); print('hello '+name, flush=True); sys.exit(7)"
    )
    started = manager.start(_command(source))

    assert [event["event"] for event in events] == ["terminal_start"]
    first = manager.read(started.terminal_id, timeout_s=1)
    assert "tty=True,True" in first.result
    assert "name?" in first.result
    manager.write(started.terminal_id, "Ada")
    output, final = _read_until(manager, started.terminal_id, "hello Ada")
    manager.close()

    assert "hello Ada" in output
    assert final.exit_code == 7
    assert final.termination_reason == "process_exit"
    assert "exited (7; reason=process_exit)" in final.result
    input_event = next(e for e in events if e["event"] == "terminal_input")
    assert input_event["input_chars"] == 3
    assert input_event["input_sha256"] == hashlib.sha256(b"Ada\n").hexdigest()
    assert "input" not in input_event
    end = next(e for e in events if e["event"] == "terminal_end")
    assert end["reason"] == "process_exit"
    assert end["exit_code"] == 7
    assert (tmp_path / "run" / started.log_path).is_file()


def test_one_live_process_input_cap_and_session_cleanup(tmp_path):
    manager, events = _manager(tmp_path, max_input_chars=4)
    terminal_id = manager.start(
        _command("import time; print('ready', flush=True); time.sleep(30)")
    ).terminal_id

    with pytest.raises(TerminalProcessError, match="already running"):
        manager.start("sleep 30")
    with pytest.raises(TerminalProcessError, match="character limit"):
        manager.write(terminal_id, "12345")
    with pytest.raises(TerminalProcessError, match="finite number"):
        manager.read(terminal_id, timeout_s=float("nan"))

    rejected = next(e for e in events if e["event"] == "terminal_input")
    assert rejected["complete"] is False
    assert rejected["bytes_written"] == 0
    assert rejected["rejection"] == "input_limit"
    assert "12345" not in json.dumps(rejected)

    manager.close()
    end = next(e for e in events if e["event"] == "terminal_end")
    assert end["reason"] == "session_end"
    assert end["exit_code"] is not None
    with pytest.raises(TerminalProcessError, match="closed"):
        manager.start("true")


def test_output_and_lifetime_bounds_terminate_processes_truthfully(tmp_path):
    output_manager, output_events = _manager(
        tmp_path / "output", max_output_bytes=64
    )
    output_id = output_manager.start(
        _command("import sys; sys.stdout.write('x'*4096); sys.stdout.flush()")
    ).terminal_id
    output, output_final = _read_until(
        output_manager, output_id, "output_limit_reached"
    )
    output_manager.close()
    output_end = next(e for e in output_events if e["event"] == "terminal_end")
    assert output_final.output_limited is True
    assert output_end["reason"] == "output_limit"
    assert "output_limit_reached" in output
    assert (tmp_path / "output" / "run" / ".terminals" / "t0001.log").stat().st_size == 64

    lifetime_manager, lifetime_events = _manager(
        tmp_path / "lifetime", max_lifetime_s=0.1
    )
    lifetime_id = lifetime_manager.start("sleep 30").terminal_id
    _output, lifetime_final = _read_until(
        lifetime_manager, lifetime_id, "lifetime_timeout"
    )
    lifetime_manager.close()
    lifetime_end = next(
        e for e in lifetime_events if e["event"] == "terminal_end"
    )
    assert lifetime_final.running is False
    assert lifetime_final.termination_reason == "lifetime_timeout"
    assert lifetime_end["reason"] == "lifetime_timeout"


def test_sandboxed_manager_reuses_the_normal_shell_boundary(tmp_path, monkeypatch):
    captured = {}

    def fake_builder(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return ["/bin/true"]

    monkeypatch.setattr(
        "scripts.llm_solver.harness.terminal_process."
        "build_background_sandbox_argv",
        fake_builder,
    )
    manager = TerminalProcessManager.sandboxed(
        run_dir=tmp_path / "run",
        cwd=tmp_path,
        bwrap_bin="/usr/bin/bwrap",
        unreadable_paths=(".git",),
        readable_paths=("docs",),
        sandbox_required=True,
        sandbox=True,
        read_timeout_s=1,
        max_lifetime_s=5,
        max_output_bytes=1024,
        max_input_chars=32,
    )

    assert manager.argv_builder("python -i") == ["/bin/true"]
    assert captured["command"] == "python -i"
    assert captured["cwd"] == str(tmp_path.resolve())
    assert captured["unreadable_paths"] == (".git",)
    assert captured["readable_paths"] == ("docs",)
    assert captured["sandbox_required"] is True
    assert captured["sandbox"] is True
    assert captured["interactive"] is True
    manager.close()


def test_session_routes_terminal_calls_and_closes_on_interruption(
    tmp_path, monkeypatch
):
    trace_path = tmp_path / ".trace.jsonl"
    cfg = make_config(
        runtime_mode="assistant",
        tools_terminal_enabled=True,
        tools_terminal_read_timeout=1,
        tools_terminal_max_lifetime=30,
        tools_terminal_max_output_bytes=4096,
        tools_terminal_max_input_chars=64,
        tools_unified_envelope_enabled=False,
        sandbox_bash=False,
    )
    with trace_path.open("a+", encoding="utf-8") as trace:
        session = Session(
            cfg,
            SimpleNamespace(),
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            session_number=4,
            artifact_dir=tmp_path / "artifacts",
        )
        session._current_turn = 2
        common = {
            "cwd": str(tmp_path), "cfg": cfg,
            "tool_registry": session._tool_registry,
        }
        started = dispatch(
            "terminal_start",
            {"cmd": _command("import time; print('ready', flush=True); time.sleep(30)")},
            **common,
        )
        assert "terminal t0001" in started
        status = dispatch(
            "terminal_io", {"terminal_id": "t0001", "timeout_s": 1}, **common
        )
        assert "ready" in status

        import scripts.llm_solver.harness._loop.run_step as run_step

        def interrupted(_session):
            raise KeyboardInterrupt

        monkeypatch.setattr(run_step, "run_session_loop", interrupted)
        with pytest.raises(KeyboardInterrupt):
            session.run()

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    start = next(e for e in events if e["event"] == "terminal_start")
    read = next(e for e in events if e["event"] == "terminal_read")
    end = next(e for e in events if e["event"] == "terminal_end")
    assert start["session_number"] == read["session_number"] == 4
    assert start["turn_number"] == read["turn_number"] == 2
    assert read["result"] == status
    assert end["reason"] == "session_end"


def test_terminal_io_can_explicitly_terminate_and_return_final_status(tmp_path):
    cfg = make_config(
        runtime_mode="assistant",
        tools_terminal_enabled=True,
        tools_terminal_read_timeout=1,
        tools_terminal_max_lifetime=30,
        tools_terminal_max_output_bytes=4096,
        tools_terminal_max_input_chars=64,
        tools_unified_envelope_enabled=False,
        sandbox_bash=False,
    )
    session = Session(
        cfg, SimpleNamespace(), "system", "task", str(tmp_path),
        artifact_dir=tmp_path / "artifacts",
    )
    common = {
        "cwd": str(tmp_path), "cfg": cfg,
        "tool_registry": session._tool_registry,
    }
    dispatch(
        "terminal_start",
        {"cmd": _command("import time; print('ready', flush=True); time.sleep(30)")},
        **common,
    )

    rejected = dispatch(
        "terminal_io",
        {"terminal_id": "t0001", "input": "x", "terminate": True},
        **common,
    )
    assert "cannot send input and terminate" in rejected
    final = dispatch(
        "terminal_io", {"terminal_id": "t0001", "terminate": True}, **common
    )
    assert "reason=explicit" in final
    assert "exited" in final
    session._terminal_manager.close()


def test_terminal_replay_is_exact_and_never_launches_a_process(
    tmp_path, monkeypatch
):
    manager, events = _manager(tmp_path)
    command = _command("name=input(); print('hello '+name, flush=True)")
    terminal_id = manager.start(command).terminal_id
    manager.write(terminal_id, "Ada")
    live_output, _final = _read_until(manager, terminal_id, "hello Ada")
    manager.close()

    replay = ReplayTerminalProcessManager(events)
    monkeypatch.setattr(
        "scripts.llm_solver.harness.terminal_process.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replay started an operating-system process")
        ),
    )
    replay_id = replay.start(command).terminal_id
    replay.write(replay_id, "Ada")
    replay_output = ""
    while not replay.consumed_all:
        replay_output += replay.read(replay_id).result
    assert replay_output == live_output
    assert replay.consumed_all is True

    changed = ReplayTerminalProcessManager(events)
    with pytest.raises(TerminalProcessError, match="command digest mismatch"):
        changed.start(command + " --changed")


def test_risky_terminal_start_uses_the_existing_approval_gate(tmp_path):
    trace_path = tmp_path / ".trace.jsonl"
    allowed, reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=trace_path,
        tool_name="terminal_start",
        tool_args={"cmd": "rm -rf build"},
        args_summary="terminal_start(cmd=<redacted>)",
    )
    assert allowed is False
    assert reason == "destructive file deletion via rm"
    request = json.loads((tmp_path / "approval_request.json").read_text())
    assert request["tool_name"] == "terminal_start"
    assert "cmd" not in request

    (tmp_path / "approval_request.json").unlink()
    allowed, reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=trace_path,
        tool_name="terminal_io",
        tool_args={"terminal_id": "t0001", "input": "rm -rf build"},
        args_summary="terminal_io(terminal_id=t0001, input=<redacted>)",
    )
    assert allowed is False
    assert reason == "destructive file deletion via rm"
    request = json.loads((tmp_path / "approval_request.json").read_text())
    assert request["tool_name"] == "terminal_io"
    assert "cmd" not in request
    assert "input" not in request
