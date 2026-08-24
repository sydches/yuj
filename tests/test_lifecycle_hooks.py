"""Acceptance coverage for trusted host-side lifecycle hooks."""
from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.hooks import (
    HookConfigurationError,
    HookRunner,
)
from scripts.llm_solver.harness.loop import Session, solve_task
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.replay_client import ReplayClient
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _stub_script(tmp_path: Path) -> Path:
    script = tmp_path / "hook_stub.py"
    script.write_text(
        """\
import json
import os
from pathlib import Path
import sys
import time

payload = json.load(sys.stdin)
mode = sys.argv[1]
marker = Path(sys.argv[2])
with marker.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "event": payload["event"],
        "run_id": payload["run_id"],
        "run_dir": os.environ["YUJ_RUN_DIR"],
        "tool_args": payload.get("tool_args"),
    }, sort_keys=True) + "\\n")

if mode == "block":
    print(json.dumps({"error": "operator policy rejected the event"}))
    raise SystemExit(2)
if mode == "rewrite":
    print(json.dumps({"updatedInput": {"path": "rewritten.txt"}}))
elif mode == "annotate":
    print(json.dumps({"additionalContext": "reviewed by lifecycle hook"}))
elif mode == "runtime":
    if payload["event"] == "pre_tool":
        print(json.dumps({"updated_input": {"path": "rewritten.txt"}}))
    elif payload["event"] == "post_tool":
        print(json.dumps({"additional_context": "post-tool annotation"}))
    else:
        print("{}")
elif mode == "timeout":
    time.sleep(1)
    print("{}")
elif mode == "error":
    print("ordinary hook failure", file=sys.stderr)
    raise SystemExit(7)
else:
    print("{}")
"""
    )
    return script


def _runner(
    tmp_path: Path,
    *,
    handlers: dict[str, object],
    events: list[dict[str, object]],
    replay: bool = False,
    recorded_events=(),
    sandbox_required: bool = False,
) -> HookRunner:
    return HookRunner(
        enabled=True,
        handlers=handlers,
        task_cwd=tmp_path,
        run_dir=tmp_path / "run-artifacts",
        run_id="acceptance-run",
        session_number=3,
        sandbox_required=sandbox_required,
        event_sink=events.append,
        replay=replay,
        recorded_events=recorded_events,
    )


def _run_tool_event(runner: HookRunner, event: str = "pre_tool"):
    return runner.run(
        event,
        turn=4,
        tool_call_id="call-32",
        tool_name="read",
        tool_args={"path": "original.txt"},
    )


def test_public_config_defaults_overlay_and_validation(tmp_path: Path) -> None:
    defaults = load_config()
    assert defaults.hooks_enabled is False
    assert defaults.hooks == {
        "pre_tool": [],
        "post_tool": [],
        "pre_model": [],
        "session_start": [],
        "session_end": [],
        "done": [],
    }

    overlay = tmp_path / "hooks.toml"
    overlay.write_text(
        """
[hooks]
enabled = true

[[hooks.pre_tool]]
matcher = "re:read|grep"
command = ["/usr/bin/env", "true"]
timeout_s = 1.5
"""
    )
    configured = load_config(user_config=overlay)
    assert configured.hooks_enabled is True
    assert configured.hooks["pre_tool"] == [{
        "matcher": "re:read|grep",
        "command": ["/usr/bin/env", "true"],
        "timeout_s": 1.5,
    }]
    assert configured.hooks["done"] == []

    overlay.write_text(
        """
[hooks]
enabled = true

[[hooks.pre_tool]]
command = ["/bin/true"]
timeout_s = 0
"""
    )
    with pytest.raises(ValueError, match="hooks.pre_tool.*timeout_s"):
        load_config(user_config=overlay)


def test_exit_two_blocks_and_records_trace(tmp_path: Path) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "block.jsonl"
    events: list[dict[str, object]] = []
    runner = _runner(
        tmp_path,
        handlers={
            "pre_tool": [{
                "matcher": "read",
                "command": [sys.executable, str(script), "block", str(marker)],
                "timeout_s": 1,
            }]
        },
        events=events,
    )

    effect = _run_tool_event(runner)

    assert effect.blocked is True
    assert effect.reason == "operator policy rejected the event"
    assert marker.is_file()
    assert events == [{
        "hook_event": "pre_tool",
        "hook_index": 0,
        "matcher": "read",
        "command": (
            f"{sys.executable} {script} block {marker}"
        ),
        "exit": 2,
        "ms": events[0]["ms"],
        "outcome": "block",
        "tool_call_id": "call-32",
        "tool_name": "read",
        "reason": "operator policy rejected the event",
    }]
    assert isinstance(events[0]["ms"], int)
    assert events[0]["ms"] >= 0


def test_rewrite_and_annotation_effects_use_normalized_context(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "effects.jsonl"
    events: list[dict[str, object]] = []
    runner = _runner(
        tmp_path,
        handlers={
            "pre_tool": [{
                "matcher": "read",
                "command": [sys.executable, str(script), "rewrite", str(marker)],
            }],
            "post_tool": [{
                "matcher": "read",
                "command": [sys.executable, str(script), "annotate", str(marker)],
            }],
        },
        events=events,
    )

    rewritten = _run_tool_event(runner)
    annotated = _run_tool_event(runner, "post_tool")

    assert rewritten.updated_input == {"path": "rewritten.txt"}
    assert rewritten.blocked is False
    assert annotated.context_block() == (
        '<injected-fragment source="hook">\n'
        "reviewed by lifecycle hook\n"
        "</injected-fragment>"
    )
    assert [event["outcome"] for event in events] == ["rewrite", "annotate"]


def test_timeout_and_other_nonzero_exit_fail_open_and_trace(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "fail-open.jsonl"
    events: list[dict[str, object]] = []
    runner = _runner(
        tmp_path,
        handlers={
            "pre_tool": [
                {
                    "matcher": "read",
                    "command": [
                        sys.executable, str(script), "timeout", str(marker)
                    ],
                    "timeout_s": 0.03,
                },
                {
                    "matcher": "read",
                    "command": [
                        sys.executable, str(script), "error", str(marker)
                    ],
                    "timeout_s": 1,
                },
            ]
        },
        events=events,
    )

    effect = _run_tool_event(runner)

    assert effect.blocked is False
    assert effect.updated_input is None
    assert [event["outcome"] for event in events] == ["timeout", "error"]
    assert [event["exit"] for event in events] == [None, 7]
    assert events[0]["reason"] == "hook timed out"
    assert events[1]["reason"] == "ordinary hook failure"


def test_pre_tool_block_never_enters_handler_and_is_model_visible(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "runtime-block.jsonl"
    cfg = make_config(
        max_turns=2,
        hooks_enabled=True,
        hooks={
            "pre_tool": [{
                "matcher": "read",
                "command": [
                    sys.executable, str(script), "block", str(marker)
                ],
            }],
        },
        error_nudge_threshold=99,
        error_abort_threshold=99,
        error_same_class_threshold=99,
        rumination_nudge_threshold=999,
    )
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(
            content="Try the read.",
            tool_calls=[ToolCall(
                id="blocked-read",
                name="read",
                arguments={"path": "never-read.txt"},
            )],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        ),
        TurnResult(
            content="Stop.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        ),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": "Try the read.", "tool_calls": []},
        {"role": "assistant", "content": "Stop."},
    ]
    trace = StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        session_number=2,
    )
    captured: list[str] = []
    original_add = session.context.add_tool_result

    def capture(tool_call_id, result, **kwargs):
        captured.append(result)
        return original_add(tool_call_id, result, **kwargs)

    session.context.add_tool_result = capture
    with (
        patch("scripts.llm_solver.harness.loop.dispatch") as dispatch_spy,
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
    ):
        result = session.run()

    assert result.done is True
    dispatch_spy.assert_not_called()
    assert "pre_tool hook blocked this call" in captured[0]
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    hook = next(event for event in events if event["event"] == "hook")
    attempted = next(event for event in events if event["event"] == "tool_call")
    assert hook["outcome"] == "block"
    assert attempted["gate_blocked"] is True
    assert attempted["gate_reason"] == "hook_block"
    assert not any(event["event"] == "tool_start" for event in events)


def test_pre_tool_rewrite_precedes_validation_and_post_tool_annotates(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "runtime.jsonl"
    command = [sys.executable, str(script), "runtime", str(marker)]
    cfg = make_config(
        max_turns=2,
        hooks_enabled=True,
        hooks={
            "pre_tool": [{"matcher": "read", "command": command}],
            "post_tool": [{"matcher": "read", "command": command}],
        },
        tools_schema_validation="reject",
        error_nudge_threshold=99,
        error_abort_threshold=99,
        error_same_class_threshold=99,
        rumination_nudge_threshold=999,
    )
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(
            content="Read the file.",
            tool_calls=[ToolCall(id="read-1", name="read", arguments={})],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        ),
        TurnResult(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        ),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": "Read the file.", "tool_calls": []},
        {"role": "assistant", "content": "Done."},
    ]
    trace = StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        session_number=2,
        artifact_dir=tmp_path / "run-artifacts",
    )
    captured: list[str] = []
    original_add = session.context.add_tool_result

    def capture(tool_call_id, result, **kwargs):
        captured.append(result)
        return original_add(tool_call_id, result, **kwargs)

    session.context.add_tool_result = capture
    with (
        patch("scripts.llm_solver.harness.loop.dispatch", return_value="FILE")
    ) as dispatch_spy, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        result = session.run()

    assert result.done is True
    assert dispatch_spy.call_args.args[:2] == (
        "read",
        {"path": "rewritten.txt"},
    )
    assert '<injected-fragment source="hook">' in captured[0]
    assert "post-tool annotation" in captured[0]
    trace_events = [json.loads(line) for line in trace.getvalue().splitlines()]
    hooks = [event for event in trace_events if event["event"] == "hook"]
    assert [event["hook_event"] for event in hooks] == ["pre_tool", "post_tool"]
    assert [event["outcome"] for event in hooks] == ["rewrite", "annotate"]
    assert not any(event["event"] == "schema_reject" for event in trace_events)
    assert TRACE_EVENT_REQUIRED_FIELDS["hook"] == frozenset({
        "hook_event", "command", "exit", "ms", "outcome",
    })


class _LifecycleClient:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.profile = load_profile("_base", PROJECT_ROOT / "profiles")
        self._turns = [TurnResult(
            content="Complete.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=7, completion_tokens=2),
        )]

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id

    def chat(self, messages, tools, turn=0):
        return self._turns.pop(0)

    def build_assistant_message(self, content, tool_calls):
        return {"role": "assistant", "content": content}


def test_driver_runs_all_non_tool_lifecycle_events_with_run_directory(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "lifecycle.jsonl"
    command = [sys.executable, str(script), "log", str(marker)]
    handlers = {
        event: [{"matcher": "*", "command": command}]
        for event in ("session_start", "pre_model", "done", "session_end")
    }
    cfg = make_config(
        profile_name="_base",
        max_sessions=1,
        max_turns=1,
        hooks_enabled=True,
        hooks=handlers,
    )
    client = _LifecycleClient(cfg)

    with (
        patch("scripts.llm_solver.harness.loop._auto_commit"),
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
    ):
        success = solve_task(tmp_path, cfg, client, initial_prompt="Finish it.")

    assert success is True
    invocations = [
        json.loads(line) for line in marker.read_text().splitlines()
    ]
    assert [row["event"] for row in invocations] == [
        "session_start", "pre_model", "done", "session_end",
    ]
    assert all(row["run_dir"] == str(tmp_path) for row in invocations)
    assert all(row["run_id"] == tmp_path.name for row in invocations)
    trace_events = [
        json.loads(line) for line in trace_path(tmp_path).read_text().splitlines()
    ]
    hooks = [row for row in trace_events if row.get("event") == "hook"]
    assert [row["hook_event"] for row in hooks] == [
        "session_start", "pre_model", "done", "session_end",
    ]


def test_replay_consumes_recorded_effect_without_launching_command(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path / "replay.jsonl"
    command = [sys.executable, str(script), "rewrite", str(marker)]
    handlers = {"pre_tool": [{"matcher": "read", "command": command}]}
    live_fields: list[dict[str, object]] = []
    live = _runner(tmp_path, handlers=handlers, events=live_fields)
    live_effect = _run_tool_event(live)
    assert live_effect.updated_input == {"path": "rewritten.txt"}
    before = marker.read_text()

    source_row = {
        "event": "hook",
        "session_number": 3,
        "turn_number": 4,
        **live_fields[0],
    }
    replay_fields: list[dict[str, object]] = []
    replay = _runner(
        tmp_path,
        handlers=handlers,
        events=replay_fields,
        replay=True,
        recorded_events=[source_row],
        sandbox_required=True,
    )

    replay_effect = _run_tool_event(replay)

    assert replay_effect == live_effect
    assert marker.read_text() == before
    assert replay_fields[0]["replayed"] is True
    assert replay_fields[0]["outcome"] == "rewrite"

    transcript = tmp_path / "recording.log"
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": []})
        + "\n=== turn 001 output ===\n"
        + json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
    )
    source_trace = tmp_path / "source.trace.jsonl"
    source_trace.write_text(json.dumps(source_row) + "\n")
    client = ReplayClient(transcript, source_trace_path=source_trace)
    assert client.hook_events == [source_row]


def test_required_sandbox_rejects_task_owned_hook_before_execution(
    tmp_path: Path,
) -> None:
    script = _stub_script(tmp_path)
    marker = tmp_path.parent / f"{tmp_path.name}-must-not-run.jsonl"
    handlers = {
        "pre_tool": [{
            "matcher": "read",
            "command": [sys.executable, str(script), "log", str(marker)],
        }]
    }

    cfg = make_config(
        hooks_enabled=True,
        hooks=handlers,
        sandbox_required=True,
    )

    with pytest.raises(HookConfigurationError, match="inside the task cwd"):
        Session(
            cfg,
            MagicMock(),
            "system",
            "task",
            str(tmp_path),
        )

    assert not marker.exists()


def test_hook_trace_rows_do_not_become_mechanical_state() -> None:
    row = {
        "event": "hook",
        "session_number": 1,
        "turn_number": 0,
        "hook_event": "pre_tool",
        "command": "/trusted/hook",
        "exit": 0,
        "ms": 4,
        "outcome": "rewrite",
        "updated_input": {"secret": "trace-only"},
        "additional_context": "trace-only annotation",
    }

    state = project([row], max_result_chars=2000)

    rendered = json.dumps(state, sort_keys=True)
    assert "trace-only" not in rendered
    assert state["meta"]["event_count"] == 1
    assert state["trace"] == []
