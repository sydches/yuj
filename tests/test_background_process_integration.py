"""Public config, tool, session, trace, and replay seams for issue #29."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness._loop.profile_resolution import apply_profile_to_schemas
from llm_solver.harness.loop import Session
from llm_solver.harness.process_manager import ReplayProcessManager
from llm_solver.harness.schemas import get_tool_schemas
from llm_solver.harness.tools import dispatch
from llm_solver.server.replay_client import ReplayClient


def _tool_names(cfg) -> set[str]:
    schemas = apply_profile_to_schemas(
        get_tool_schemas(), cfg, SimpleNamespace()
    )
    return {schema["function"]["name"] for schema in schemas}


def test_background_config_defaults_overlay_and_validation(tmp_path):
    defaults = load_config()
    assert defaults.tools_background_enabled is False
    assert defaults.tools_background_max_procs == 4
    assert defaults.tools_background_poll_timeout == 300

    overlay = tmp_path / "enabled.toml"
    overlay.write_text(
        "[tools]\nbackground_enabled = true\n"
        "background_max_procs = 2\nbackground_poll_timeout = 1.5\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_background_enabled is True
    assert configured.tools_background_max_procs == 2
    assert configured.tools_background_poll_timeout == 1.5

    for body, match in (
        ('background_enabled = "yes"', "background_enabled"),
        ("background_max_procs = 0", "background_max_procs"),
        ("background_poll_timeout = -1", "background_poll_timeout"),
        ("background_poll_timeout = nan", "background_poll_timeout"),
    ):
        overlay.write_text(f"[tools]\n{body}\n")
        with pytest.raises(ValueError, match=match):
            load_config(user_config=overlay)


def test_background_tool_surface_is_opt_in_and_bash_shape_tracks_gate():
    disabled = make_config(tools_background_enabled=False)
    enabled = make_config(tools_background_enabled=True)
    assert "bash_poll" not in _tool_names(disabled)
    assert "bash_kill" not in _tool_names(disabled)
    assert {"bash_poll", "bash_kill"} <= _tool_names(enabled)

    def bash_properties(cfg):
        schemas = apply_profile_to_schemas(
            get_tool_schemas(), cfg, SimpleNamespace()
        )
        bash_schema = next(
            item for item in schemas if item["function"]["name"] == "bash"
        )
        return bash_schema["function"]["parameters"]["properties"]

    assert "background" not in bash_properties(disabled)
    assert bash_properties(enabled)["background"] == {"type": "boolean"}


def test_session_routes_background_calls_and_traces_exact_admitted_poll(tmp_path):
    trace = io.StringIO()
    cfg = make_config(
        tools_background_enabled=True,
        tools_background_poll_timeout=1.0,
        sandbox_bash=False,
        tools_unified_envelope_enabled=True,
    )
    session = Session(
        cfg,
        SimpleNamespace(),
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        trace_path=tmp_path / ".trace.jsonl",
        session_number=3,
        artifact_dir=tmp_path / "artifacts",
    )
    session._current_turn = 7
    common = {
        "cwd": str(tmp_path),
        "cfg": cfg,
        "tool_registry": session._tool_registry,
    }
    started = dispatch(
        "bash", {"cmd": "printf 'hello from child\\n'", "background": True},
        **common,
    )
    assert "Started background process p0001" in started
    polled = dispatch(
        "bash_poll", {"proc_id": "p0001", "timeout_s": 1}, **common
    )
    assert polled.count("<tool_result") == 1
    assert 'tool_name="bash_poll"' in polled
    assert "hello from child" in polled

    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    start_event = next(event for event in events if event["event"] == "proc_start")
    poll_event = next(event for event in events if event["event"] == "proc_poll")
    assert start_event["session_number"] == 3
    assert start_event["turn_number"] == 7
    assert poll_event["result"] == polled
    assert poll_event["output_sha256"] == hashlib.sha256(polled.encode()).hexdigest()
    assert poll_event["cursor_end"] > poll_event["cursor_start"]
    assert (tmp_path / "artifacts" / ".procs" / "p0001.log").read_text() == (
        "hello from child\n"
    )
    session._process_manager.close()


def test_session_end_closes_live_process_and_records_kill(tmp_path):
    trace = io.StringIO()
    cfg = make_config(
        tools_background_enabled=True,
        sandbox_bash=False,
        tools_unified_envelope_enabled=False,
    )
    session = Session(
        cfg,
        SimpleNamespace(),
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        trace_path=tmp_path / ".trace.jsonl",
        artifact_dir=tmp_path / "artifacts",
    )
    dispatch(
        "bash",
        {"cmd": "sleep 30", "background": True},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=session._tool_registry,
    )
    session._process_manager.close()
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    killed = next(event for event in events if event["event"] == "proc_kill")
    assert killed["reason"] == "session_end"
    assert killed["was_running"] is True


def test_replay_client_retains_proc_stream_and_session_never_starts_process(
    tmp_path, monkeypatch
):
    transcript = tmp_path / "recording.log"
    transcript.write_text(
        "=== turn 001 input ===\n{\"messages\": []}\n"
        "=== turn 001 output ===\n"
        "{\"choices\":[{\"message\":{\"content\":null,\"tool_calls\":[]},"
        "\"finish_reason\":\"stop\"}],\"usage\":{}}\n"
    )
    command = "never execute this"
    poll_result = "<tool_result tool_name=\"bash_poll\" status=\"ok\" v=\"1\">\n"
    poll_result += "recorded bytes\n</tool_result>"
    events = [
        {
            "event": "proc_start", "session_number": 0, "proc_id": "p0001",
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "log_path": ".procs/p0001.log", "result": "started",
        },
        {
            "event": "proc_poll", "session_number": 0, "proc_id": "p0001",
            "result": poll_result,
            "output_sha256": hashlib.sha256(poll_result.encode()).hexdigest(),
            "running": False, "exit_code": 0, "timed_out": False,
            "cursor_start": 0, "cursor_end": 14,
        },
    ]
    source_trace = tmp_path / "source.trace.jsonl"
    source_trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    client = ReplayClient(transcript, source_trace_path=source_trace)
    assert client.process_events == events

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("replay started an operating-system process")

    monkeypatch.setattr("subprocess.Popen", forbidden_popen)
    cfg = make_config(
        tools_background_enabled=True,
        sandbox_bash=False,
        tools_unified_envelope_enabled=True,
    )
    session = Session(cfg, client, "system", "task", str(tmp_path))
    assert isinstance(session._process_manager, ReplayProcessManager)
    common = {
        "cwd": str(tmp_path), "cfg": cfg,
        "tool_registry": session._tool_registry,
    }
    assert "started" in dispatch(
        "bash", {"cmd": command, "background": True}, **common
    )
    assert dispatch("bash_poll", {"proc_id": "p0001"}, **common) == poll_result
