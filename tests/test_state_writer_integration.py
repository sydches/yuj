"""Tests for state_writer — edge cases, end-to-end Session integration, reasoning field.

Companion file: test_state_writer_projection.py covers the pure projection
function and file-write helpers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.llm_solver.harness.state_writer import (
    project,
    project_from_trace,
    write_state_from_trace,
)

_CAP = 20000


def _project(events):
    return project(events, max_result_chars=_CAP)


def _project_from_trace(path):
    return project_from_trace(path, max_result_chars=_CAP)


def _write_state_from_trace(trace_path, state_path):
    return write_state_from_trace(trace_path, state_path, max_result_chars=_CAP)


class TestProjectEdgeCases:
    """Guard against malformed events and missing fields."""

    def test_tool_call_with_missing_tool_name(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "args_summary": "a", "result_summary": "r"},
        ]
        out = _project(events)
        assert len(out["trace"]) == 1
        assert out["trace"][0]["action"].startswith("?(")

    def test_tool_call_with_missing_args_summary(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "result_summary": "r"},
        ]
        out = _project(events)
        assert out["trace"][0]["action"] == "bash()"

    def test_tool_call_with_none_result_summary(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": None},
        ]
        out = _project(events)
        assert out["trace"][0]["result"] == ""

    def test_unknown_event_type_is_ignored(self):
        events = [
            {"event": "noise", "session_number": 1},
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "r"},
        ]
        out = _project(events)
        assert len(out["trace"]) == 1

    def test_session_end_without_matching_start(self):
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 1},
        ]
        out = _project(events)
        assert "stop" in out["state"]["last_verify"]
        assert out["trace"] == []

    def test_unicode_in_args_and_results(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='echo café ñ 日本語 🔥'",
             "result_summary": "café ñ 日本語 🔥"},
        ]
        out = _project(events)
        assert "café" in out["trace"][0]["action"]
        assert "日本語" in out["trace"][0]["result"]
        assert "🔥" in out["trace"][0]["result"]

    def test_trailing_partial_json_line_raises(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(
            json.dumps({"event": "tool_call", "session_number": 1,
                        "turn_number": 0, "tool_name": "bash",
                        "args_summary": "a", "result_summary": "r"}) + "\n"
            + '{"event": "tool_call", "session_number": 1, "turn_numb'
        )
        with pytest.raises(json.JSONDecodeError):
            _project_from_trace(trace)

    def test_atomic_write_leaves_no_partial_file(self, tmp_path: Path, monkeypatch):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / ".solver" / "state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text('{"state": {"current_attempt": "pre-existing"}, '
                              '"trace": [], "gates": [], "evidence": [], '
                              '"inference": []}')
        pre_existing = state_path.read_text()

        def fail_replace(self, target):
            raise OSError("simulated rename failure")
        monkeypatch.setattr(Path, "replace", fail_replace)

        try:
            _write_state_from_trace(trace, state_path)
        except OSError:
            pass

        assert state_path.read_text() == pre_existing

    def test_projection_is_not_affected_by_event_order_within_type(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 5,
             "tool_name": "bash", "args_summary": "B", "result_summary": "r"},
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "bash", "args_summary": "A", "result_summary": "r"},
        ]
        out = _project(events)
        assert out["trace"][0]["action"] == "bash(B)"
        assert out["trace"][1]["action"] == "bash(A)"


def _make_session_cfg():
    from scripts.llm_solver.config import Config
    return Config(
        base_url="http://localhost:8080/v1",
        api_key="local",
        timeout_connect=10,
        timeout_read=120,
        health_poll_interval=2, health_timeout=2, launch_timeout=120, stop_settle=2,
        model="test",
        # Fields required by this direct Config fixture.
        profile_name="test",
        sandbox_required=False,
        unreadable_paths=(),
        context_size=8000,
        context_fill_ratio=0.85,
        max_tokens_fraction=0.5,
        max_tokens=1024,
        tokenizer_id="",
        max_turns=10,
        max_sessions=1,
        duplicate_abort=3,
        error_nudge_threshold=3, rumination_nudge_threshold=200, require_intent=False,
        intent_grace_turns=3,
        min_turns_before_context=2,
        max_output_chars=_CAP,
        truncate_head_ratio=0.6,
        truncate_head_lines=100,
        truncate_tail_lines=50,
        args_summary_chars=80, trace_args_summary_chars=200,
        trace_reasoning_store_chars=800,
        solver_trace_lines=50,
        solver_evidence_lines=30,
        solver_inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        trace_reasoning_chars=150,
        pretest_head_chars=2000,
        pretest_tail_chars=1500,
        bash_timeout=60,
        grep_timeout=30,
        pretest_timeout=240,
        llama_server_bin="/bin/true",
        sandbox_bash=False,
        strip_ansi=True,
        collapse_blank_lines=True,
        collapse_duplicate_lines=True,
        collapse_similar_lines=True,
        bwrap_bin="/usr/bin/bwrap",
        max_transient_retries=0,
        retry_backoff=(1, 4, 16),
        system_header="You are a solver.",
        state_context_suffix="Continue working.",
        intent_gate_first="[harness] silent rejected",
        intent_gate_repeat="[intent gate: {count} since {first_turn}]",
        resume_base="Continue.",
        error_nudge="{count} errors",
        rumination_nudge="{count} non-write",
        rumination_gate="blocked",
        rumination_same_target_nudge="same target {target}",
        rumination_outside_cwd_nudge="outside {target}",
        test_read_nudge="read test {target}",
        contract_commit_warn="warn {source}",
        contract_commit_block="block {source}",
        contract_recovery_block="recover {reason} {target}",
        mutation_repeat_warn="warn mutation {target}",
        mutation_repeat_block="block mutation {target}",
        resume_duplicate_abort="{n} identical: {call}",
        resume_context_full="{pct}% full",
        resume_max_turns="{n}: {actions}",
        resume_length="truncated",
        resume_last_n_actions=3,
        tool_desc="minimal",
        prompt_addendum="",
        variant_name="",
    )


class TestWriteTraceIntegration:
    """Exercise the actual harness loop path: Session._write_trace → write-
    through to trace.jsonl → _refresh_state → write_state_from_trace →
    state.json updated."""

    def _make_session(self, tmp_path: Path):
        from scripts.llm_solver.harness.loop import Session
        from scripts.llm_solver.harness.context import FullTranscript
        cfg = _make_session_cfg()
        trace_path = tmp_path / ".trace.jsonl"
        state_path = tmp_path / ".solver" / "state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(
            '{"state": {}, "trace": [], "gates": [], "evidence": [], "inference": []}'
        )
        trace_file = open(trace_path, "a")
        session = Session(
            cfg=cfg,
            client=None,
            system_prompt="sys",
            initial_message="task",
            cwd=str(tmp_path),
            context_manager=FullTranscript(),
            trace_file=trace_file,
            session_number=1,
            trace_path=trace_path,
            state_path=state_path,
        )
        return session, trace_file, trace_path, state_path

    def test_write_trace_updates_state_json(self, tmp_path: Path):
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            session._write_trace({
                "event": "tool_call", "session_number": 1, "turn_number": 0,
                "tool_name": "bash", "args_summary": "ls",
                "result_summary": "out",
            })
            data = json.loads(state_path.read_text())
            assert len(data["trace"]) == 1
            assert data["trace"][0]["action"] == "bash(ls)"
            assert data["state"]["current_attempt"] == "bash(ls)"
        finally:
            trace_file.close()

    def test_no_state_write_when_state_path_is_none(self, tmp_path: Path):
        from scripts.llm_solver.harness.loop import Session
        from scripts.llm_solver.harness.context import FullTranscript
        cfg = _make_session_cfg()
        trace_path = tmp_path / ".trace.jsonl"
        trace_file = open(trace_path, "a")
        try:
            session = Session(
                cfg=cfg, client=None, system_prompt="sys",
                initial_message="task", cwd=str(tmp_path),
                context_manager=FullTranscript(),
                trace_file=trace_file, session_number=1,
                trace_path=trace_path, state_path=None,
            )
            session._write_trace({
                "event": "tool_call", "session_number": 1, "turn_number": 0,
                "tool_name": "bash", "args_summary": "ls",
                "result_summary": "r",
            })
            assert not (tmp_path / ".solver").exists()
        finally:
            trace_file.close()

    def test_multiple_writes_grow_state_trace(self, tmp_path: Path):
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            for i in range(5):
                session._write_trace({
                    "event": "tool_call", "session_number": 1, "turn_number": i,
                    "tool_name": "bash", "args_summary": f"cmd{i}",
                    "result_summary": f"r{i}",
                })
            data = json.loads(state_path.read_text())
            assert len(data["trace"]) == 5
            assert [t["action"] for t in data["trace"]] == [
                f"bash(cmd{i})" for i in range(5)
            ]
            assert data["state"]["current_attempt"] == "bash(cmd4)"
        finally:
            trace_file.close()

    def test_state_matches_offline_replay_after_live_writes(self, tmp_path: Path):
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            for i in range(10):
                session._write_trace({
                    "event": "tool_call", "session_number": 1, "turn_number": i,
                    "tool_name": "bash" if i % 2 == 0 else "read",
                    "args_summary": f"arg{i}",
                    "result_summary": f"result{i}",
                })
        finally:
            trace_file.close()

        live = json.loads(state_path.read_text())
        replay = _project_from_trace(trace_path)
        assert live == replay


class TestProjectReasoning:
    def test_reasoning_defaults_to_empty_when_field_missing(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='ls'",
             "result_summary": "file1\n"},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == ""

    def test_reasoning_propagates_from_event_to_trace_entry(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='ls'",
             "result_summary": "file1\n",
             "reasoning": "Let me check what files are in the repo first."},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == \
            "Let me check what files are in the repo first."

    def test_multi_tool_turn_shares_reasoning_across_entries(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "read", "args_summary": "path='a.py'",
             "result_summary": "...", "reasoning": "I need a.py and b.py"},
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "read", "args_summary": "path='b.py'",
             "result_summary": "...", "reasoning": "I need a.py and b.py"},
        ]
        out = _project(events)
        assert len(out["trace"]) == 2
        assert out["trace"][0]["reasoning"] == out["trace"][1]["reasoning"]
        assert out["trace"][0]["reasoning"] == "I need a.py and b.py"

    def test_reasoning_none_becomes_empty_string(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "x", "result_summary": "y",
             "reasoning": None},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == ""
