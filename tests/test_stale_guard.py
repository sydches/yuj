"""Focused tests for the read-before-edit ledger."""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver._shared.classification import is_error_result
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._guardrails.extractors import _error_signature
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.stale_guard import (
    StaleFileGuard,
    StaleGuardError,
    classify_single_file_read,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def test_block_mode_rejects_unread_file_with_trace_event(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)

    decision = guard.check_edit("src.py")

    assert decision.blocked is True
    assert decision.reason == "unread"
    assert decision.message == "ERROR: stale_file: read src.py first"
    assert events == [{
        "event": "stale_guard",
        "path": "src.py",
        "reason": "unread",
        "mode": "block",
        "blocked": True,
        "expected": None,
        "current": None,
    }]


def test_edit_after_read_is_fresh(tmp_path):
    (tmp_path / "src.py").write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    fingerprint = guard.observe_read("src.py")
    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert decision.reason == "fresh"
    assert guard.ledger_snapshot()["src.py"] == fingerprint


def test_external_content_change_after_read_is_blocked(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    guard.observe_read("src.py")

    target.write_text("changed\n")
    decision = guard.check_edit("src.py")

    assert decision.blocked is True
    assert decision.reason == "modified"
    assert decision.message == "ERROR: stale_file: read src.py first"
    assert events[-1]["event"] == "stale_guard"
    assert events[-1]["expected"]["sha256"] != events[-1]["current"]["sha256"]


def test_missing_observed_file_is_stale(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    guard.observe_read("src.py")
    target.unlink()

    decision = guard.check_edit("src.py")

    assert decision.reason == "missing"
    assert decision.blocked is True


def test_warn_mode_reports_hit_but_allows_edit(tmp_path):
    (tmp_path / "src.py").write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="warn")

    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert decision.message == "WARNING: stale_file: read src.py first"


def test_warn_is_the_default_mode(tmp_path):
    (tmp_path / "src.py").write_text("old\n")

    decision = StaleFileGuard(cwd=tmp_path).check_edit("src.py")

    assert decision.mode == "warn"
    assert decision.allowed is True


def test_off_mode_performs_no_file_check_or_trace(tmp_path):
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="off", event_sink=events.append)

    decision = guard.check_edit("does-not-exist.py")

    assert decision.allowed is True
    assert decision.reason == "off"
    assert events == []


@pytest.mark.parametrize(
    "source",
    ["write", "edit", "notebook_edit", "structural_edit", "apply_patch"],
)
def test_successful_mutation_refreshes_ledger(tmp_path, source):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    guard.observe_read("src.py")
    target.write_text("our edit\n")

    guard.observe_mutation("src.py", source=source)

    assert guard.check_edit("src.py").allowed is True


def test_metadata_only_touch_keeps_content_fresh(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("same\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    before = guard.observe_read("src.py")
    os.utime(target, ns=(before.mtime_ns + 1_000_000, before.mtime_ns + 1_000_000))

    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert events[-1]["source"] == "metadata_refresh"


@pytest.mark.parametrize(
    ("command", "verb", "path"),
    [
        ("cat src/app.py", "cat", "src/app.py"),
        ("LC_ALL=C head -n 20 -- src/app.py", "head", "src/app.py"),
        ("tail -n 5 src/app.py", "tail", "src/app.py"),
        ("sed -n '1,40p' src/app.py", "sed", "src/app.py"),
        ("grep -n 'needle' src/app.py", "grep", "src/app.py"),
        ("rg --line-number 'needle' src/app.py", "rg", "src/app.py"),
    ],
)
def test_single_file_shell_read_classifier_accepts_safe_reads(command, verb, path):
    classified = classify_single_file_read(command)

    assert classified is not None
    assert (classified.verb, classified.path) == (verb, path)


@pytest.mark.parametrize(
    "command",
    [
        "cat one.py two.py",
        "cat src.py && echo done",
        "cat src.py | head",
        "cat < src.py",
        "cat $(printf src.py)",
        "sed -i 's/a/b/' src.py",
        "sed -n '1p' one.py two.py",
        "grep -r needle src",
        "grep -c needle src.py",
        "rg --count needle src.py",
        "rg needle .",
        "wc -l src.py",
    ],
)
def test_single_file_shell_read_classifier_rejects_ambiguous_or_aggregate(command):
    assert classify_single_file_read(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "cat src.py",
        "sed -n '1p' src.py",
        "grep needle src.py",
    ],
)
def test_shell_classifier_observation_satisfies_guard(tmp_path, command):
    (tmp_path / "src.py").write_text("needle\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    assert guard.observe_shell_read(command) is not None
    assert guard.check_edit("src.py").allowed is True


def test_shell_observation_requires_existing_single_file(tmp_path):
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    assert guard.observe_shell_read("cat missing.py") is None
    assert guard.ledger_snapshot() == {}


def test_resume_rebuilds_ledger_from_trace_and_detects_later_change(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("observed\n")
    events = []
    live = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    live.observe_read("src.py")

    resumed = StaleFileGuard.from_trace(
        cwd=tmp_path, mode="block", events=events
    )
    assert resumed.check_edit("src.py").allowed is True
    target.write_text("external\n")

    assert resumed.check_edit("src.py").reason == "modified"


def test_deleted_path_event_removes_resumed_observation(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("observed\n")
    events = []
    live = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    live.observe_read("src.py")
    target.unlink()
    live.forget("src.py")

    resumed = StaleFileGuard.from_trace(
        cwd=tmp_path, mode="block", events=events
    )

    assert resumed.ledger_snapshot() == {}


def test_trace_rebuild_rejects_path_escape(tmp_path):
    events = [{
        "event": "stale_guard_observe",
        "path": "../secret",
        "source": "read",
        "fingerprint": {"mtime_ns": 1, "size": 1, "sha256": "x"},
    }]

    with pytest.raises(StaleGuardError, match="invalid path"):
        StaleFileGuard.from_trace(cwd=tmp_path, mode="block", events=events)


def _turn(*, tool_calls=(), content="", reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _client(*turns: TurnResult):
    client = MagicMock()
    client.chat.side_effect = turns
    client.build_assistant_message.side_effect = lambda content, tool_calls: {
        "role": "assistant", "content": content,
    }
    return client


def test_session_blocks_unread_edit_then_allows_edit_after_read(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    client = _client(
        _turn(tool_calls=[ToolCall(
            "edit-unread", "edit",
            {"path": "src.py", "old_str": "old", "new_str": "bad"},
        )]),
        _turn(tool_calls=[ToolCall("read", "read", {"path": "src.py"})]),
        _turn(tool_calls=[ToolCall(
            "edit-fresh", "edit",
            {"path": "src.py", "old_str": "old", "new_str": "fresh"},
        )]),
        _turn(content="done", reason="stop"),
    )
    cfg = make_config(
        max_turns=4,
        tools_stale_guard_mode="block",
        tools_unified_envelope_enabled=True,
        turn_snapshots_enabled=False,
    )
    trace = StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    captured: list[str] = []
    original_add = session.context.add_tool_result

    def capture(tool_call_id, result, **kwargs):
        captured.append(result)
        return original_add(tool_call_id, result, **kwargs)

    session.context.add_tool_result = capture
    with patch.object(session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.finish_reason == "stop"
    assert target.read_text() == "fresh\n"
    assert 'status="error" error_kind="stale_file"' in captured[0]
    assert "ERROR: stale_file: read src.py first" in captured[0]
    assert 'status="ok"' in captured[2]
    guard_events = [event for event in session._trace_events
                    if event.get("event") == "stale_guard"]
    observations = [event for event in session._trace_events
                    if event.get("event") == "stale_guard_observe"]
    assert guard_events[0]["reason"] == "unread"
    assert guard_events[0]["blocked"] is True
    assert {event["source"] for event in observations} >= {"read", "edit"}


def test_dispatch_blocks_edit_after_external_modification(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    cfg = make_config(tools_unified_envelope_enabled=True)

    read_result = dispatch(
        "read", {"path": "src.py"}, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    assert is_error_result(read_result) is False
    target.write_text("external\n")
    result = dispatch(
        "edit", {"path": "src.py", "old_str": "external", "new_str": "ours"},
        cwd=str(tmp_path), cfg=cfg, stale_guard=guard,
    )

    assert target.read_text() == "external\n"
    assert 'status="error" error_kind="stale_file"' in result
    assert events[-1]["reason"] == "modified"


def test_warn_mode_runs_edit_and_places_warning_inside_envelope(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="warn")
    cfg = make_config(tools_unified_envelope_enabled=True)

    result = dispatch(
        "edit", {"path": "src.py", "old_str": "old", "new_str": "new"},
        cwd=str(tmp_path), cfg=cfg, stale_guard=guard,
    )

    assert target.read_text() == "new\n"
    assert result.startswith('<tool_result tool_name="edit" status="ok"')
    assert "WARNING: stale_file: read src.py first" in result
    assert result.index("WARNING: stale_file") < result.rindex("</tool_result>")
    assert guard.check_edit("src.py").allowed is True


@pytest.mark.parametrize(
    "command",
    [
        "cat src.py",
        "sed -n '1p' src.py",
        "grep absent src.py",
    ],
)
def test_successful_shell_read_dispatch_credits_ledger(tmp_path, command):
    (tmp_path / "src.py").write_text("needle\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    cfg = make_config(sandbox_bash=False)

    dispatch(
        "bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard, execution_metadata={},
    )

    assert guard.check_edit("src.py").allowed is True


def test_successful_write_and_apply_patch_refresh_exact_paths(tmp_path):
    (tmp_path / "update.py").write_text("old\n")
    (tmp_path / "delete.py").write_text("remove\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    cfg = make_config(
        tools_apply_patch_enabled=True,
        tools_unified_envelope_enabled=True,
    )

    write_result = dispatch(
        "write", {"path": "written.py", "content": "written\n"},
        cwd=str(tmp_path), cfg=cfg, stale_guard=guard,
    )
    delete_header = "*** " + "Delete File: delete.py"
    patch_text = """*** Begin Patch
*** Add File: added.py
+added
*** Update File: update.py
@@
-old
+updated
{delete_header}
*** End Patch""".format(delete_header=delete_header)
    patch_result = dispatch(
        "apply_patch", {"patch": patch_text},
        cwd=str(tmp_path), cfg=cfg, stale_guard=guard,
    )

    assert is_error_result(write_result) is False
    assert patch_result.startswith('<apply_patch ok="true"')
    assert guard.check_edit("written.py").allowed is True
    assert guard.check_edit("added.py").allowed is True
    assert guard.check_edit("update.py").allowed is True
    assert "delete.py" not in guard.ledger_snapshot()


def test_session_rebuilds_guard_ledger_from_trace(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("observed\n")
    trace_path = tmp_path / "trace.jsonl"
    cfg = make_config(tools_stale_guard_mode="block")
    first_client = _client(_turn(content="done", reason="stop"))

    with trace_path.open("a", encoding="utf-8") as trace_file:
        first = Session(
            cfg, first_client, "system", "task", str(tmp_path),
            trace_file=trace_file, trace_path=trace_path,
        )
        first._stale_guard.observe_read("src.py")

    resumed = Session(
        cfg, _client(_turn(content="done", reason="stop")),
        "system", "task", str(tmp_path), trace_path=trace_path,
        session_number=1,
    )
    assert resumed._stale_guard.check_edit("src.py").allowed is True
    target.write_text("external\n")
    assert resumed._stale_guard.check_edit("src.py").reason == "modified"


def test_stale_error_has_stable_ladder_signature_across_paths(tmp_path):
    cfg = make_config(tools_unified_envelope_enabled=True)
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    results = []
    for name in ("one.py", "two.py"):
        (tmp_path / name).write_text("old\n")
        results.append(dispatch(
            "edit", {"path": name, "old_str": "old", "new_str": "new"},
            cwd=str(tmp_path), cfg=cfg, stale_guard=guard,
        ))

    assert all(is_error_result(result) for result in results)
    assert [_error_signature(result) for result in results] == [
        "stale_file", "stale_file",
    ]


def test_stale_guard_config_defaults_overlay_and_validation(tmp_path):
    assert load_config().tools_stale_guard_mode == "warn"
    overlay = tmp_path / "stale.toml"
    overlay.write_text('[tools]\nstale_guard_mode = "block"\n')
    assert load_config(user_config=overlay).tools_stale_guard_mode == "block"
    overlay.write_text('[tools]\nstale_guard_mode = "strict"\n')
    with pytest.raises(ValueError, match="tools.stale_guard_mode"):
        load_config(user_config=overlay)
