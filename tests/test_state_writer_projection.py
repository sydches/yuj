"""Tests for state_writer.project — pure projection function + file helpers.

Companion file: test_state_writer_integration.py covers edge cases,
end-to-end Session._write_trace integration, and the reasoning field.

Validates the pure projection function (`project`) and the file helpers
(`project_from_trace`, `write_state_from_trace`) against fixture trace event
streams. No LLM dependency, no filesystem state beyond temp dirs.

The projection does not parse task output. Trace entries carry actions and
results as recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

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


class TestProjectEmpty:
    def test_empty_events_yields_empty_schema(self):
        out = _project([])
        # The meta block reports the projection schema and input position.
        assert out == {
            "meta": {
                "schema_version": 1,
                "event_count": 0,
                "last_session": None,
                "last_turn": None,
            },
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "tools": {
                "lazy_loading_enabled": False,
                "active_limit": None,
                "registered": [],
                "active": [],
                "activations": [],
            },
            "trace": [],
            "gates": [],
            "evidence": [],
            "inference": [],
        }

    def test_only_session_start_leaves_state_blank(self):
        out = _project([{"event": "session_start", "session_number": 1}])
        assert out["state"]["current_attempt"] == ""
        assert out["state"]["last_verify"] == ""
        assert out["trace"] == []
        assert out["evidence"] == []


class TestProjectTraceAccumulation:
    def test_single_tool_call_becomes_trace_entry(self):
        events = [
            {"event": "session_start", "session_number": 1},
            {
                "event": "tool_call",
                "session_number": 1,
                "turn_number": 0,
                "tool_name": "bash",
                "args_summary": "cmd='ls'",
                "result_summary": "file1\nfile2\n",
            },
        ]
        out = _project(events)
        assert len(out["trace"]) == 1
        entry = out["trace"][0]
        assert entry["step"] == 1
        assert entry["action"] == "bash(cmd='ls')"
        assert entry["result"] == "file1\nfile2\n"
        assert entry["next"] == ""
        assert entry["session"] == 1
        assert entry["turn"] == 0
        assert "is_test" not in entry
        assert "verdict" not in entry

    def test_step_counter_is_monotonic_across_sessions(self):
        events = [
            {"event": "session_start", "session_number": 1},
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "x"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "b", "result_summary": "y"},
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 2},
            {"event": "session_start", "session_number": 2},
            {"event": "tool_call", "session_number": 2, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='f'", "result_summary": "z"},
        ]
        out = _project(events)
        assert [t["step"] for t in out["trace"]] == [1, 2, 3]
        assert out["trace"][2]["session"] == 2

    def test_current_attempt_tracks_latest_tool_call(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "first", "result_summary": "r"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "read", "args_summary": "second", "result_summary": "r"},
        ]
        out = _project(events)
        assert out["state"]["current_attempt"] == "read(second)"


class TestProjectLastVerify:
    def test_session_end_populates_last_verify(self):
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "max_turns", "turns": 60},
        ]
        out = _project(events)
        assert "max_turns" in out["state"]["last_verify"]
        assert "60" in out["state"]["last_verify"]
        assert "session 1" in out["state"]["last_verify"]

    def test_last_verify_reflects_most_recent_session_end(self):
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 5},
            {"event": "session_end", "session_number": 2,
             "finish_reason": "max_turns", "turns": 60},
        ]
        out = _project(events)
        assert "session 2" in out["state"]["last_verify"]
        assert "max_turns" in out["state"]["last_verify"]


class TestProjectEvidence:
    """Evidence population is content-blind."""

    def test_bash_exit_zero_is_ok_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='./whatever'",
             "result_summary": "done"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        ev = out["evidence"][0]
        assert ev["step"] == 1
        assert ev["action"] == "bash(cmd='./whatever')"
        assert ev["verdict"] == "OK"
        assert ev["gate_blocked"] is False

    def test_bash_nonzero_exit_is_fail_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='./anything'",
             "result_summary": "some output\n[exit code: 1]"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["verdict"] == "FAIL"

    def test_bounded_result_uses_explicit_pass_fail_for_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='./anything'",
             "result_summary": "head of a long failing output...",
             "output_snippet": "head of a long failing output...",
             "output_sha256": "abc123",
             "output_full_path": ".tool_output/1_0001_t0_trace.log",
             "pass_fail": "fail"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["verdict"] == "FAIL"
        assert out["trace"][0]["output_sha256"] == "abc123"
        assert out["trace"][0]["output_full_path"] == ".tool_output/1_0001_t0_trace.log"

    def test_error_wrapper_is_fail_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='run'",
             "result_summary": "ERROR: command timed out after 60s"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["verdict"] == "FAIL"

    def test_read_tool_does_not_generate_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='anything.txt'",
             "result_summary": "content"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_write_tool_does_not_generate_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "write", "args_summary": "path='x.py'",
             "result_summary": "OK: wrote 10 bytes to x.py"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_edit_tool_does_not_generate_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "edit", "args_summary": "path='x.py'",
             "result_summary": "OK"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_gate_blocked_bash_is_not_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='grep x'",
             "result_summary": "[harness gate] blocked",
             "gate_blocked": True},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_mixed_stream_preserves_order(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='a'",
             "result_summary": "ok"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "read", "args_summary": "path='b'", "result_summary": "data"},
            {"event": "tool_call", "session_number": 1, "turn_number": 2,
             "tool_name": "bash", "args_summary": "cmd='c'",
             "result_summary": "oops\n[exit code: 2]"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 2
        assert out["evidence"][0]["verdict"] == "OK"
        assert out["evidence"][1]["verdict"] == "FAIL"

    def test_evidence_result_is_truncated_to_evidence_cap(self):
        big = "x" * 5000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='a'",
             "result_summary": big},
        ]
        out = _project(events)
        ev_result = out["evidence"][0]["result"]
        assert len(ev_result) < len(big)
        assert ev_result.endswith("...")


class TestProjectImperativeProcess:
    def test_optional_process_block_is_content_blind_and_imperative(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source",
             "reasoning": "The fix should replace the old branch in src/app.py."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["meta"]["schema_version"] == 2
        assert out["process"]["phase"] == "candidate_edit_pending"
        assert out["process"]["pending_edit_step"] == 2
        assert out["process"]["required_next_action"] == "apply the pending source edit"
        assert out["process"]["target_paths"] == ["src/app.py"]

    def test_failed_bash_write_is_not_successful_mutation(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash",
             "args_summary": "cmd=\"python - <<'PY'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nPY\"",
             "result_summary": "PermissionError: denied\n[exit code: 1]",
             "reasoning": "The fix is to change old to new in src/app.py."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "mutation_attempt_failed"
        assert out["process"]["last_mutation_step"] is None
        assert out["process"]["last_failed_mutation_step"] == 2
        assert (
            out["process"]["required_next_action"]
            == "retry the source edit with a write method that succeeds"
        )

    def test_successful_bash_write_counts_as_mutation(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"cd /testbed && sed -i 's/old/new/' src/app.py\"",
             "result_summary": "",
             "reasoning": "The fix is to change old to new in src/app.py."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "post_mutation_unverified"
        assert out["process"]["last_mutation_step"] == 1
        assert out["process"]["last_failed_mutation_step"] is None

    def test_trace_source_write_metadata_counts_as_mutation(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\"",
             "result_summary": "SUCCESS",
             "reasoning": "The fix is to change old to new in src/app.py.",
             "source_write_like": True,
             "source_write_paths": ["src/app.py"]},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["trace"][0]["source_write_like"] is True
        assert out["trace"][0]["source_write_paths"] == ["src/app.py"]
        assert out["process"]["phase"] == "post_mutation_unverified"
        assert out["process"]["last_mutation_step"] == 1
        assert out["process"]["target_paths"] == ["src/app.py"]

    def test_exploratory_reasoning_is_not_pending_edit(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source",
             "reasoning": "I need to understand this code before editing it."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "pre_mutation_discovery"
        assert out["process"]["pending_edit_step"] is None

    def test_applied_edit_recap_is_not_pending_edit(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"python - <<'PY'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nPY\"",
             "result_summary": "",
             "source_write_like": True,
             "source_write_paths": ["src/app.py"]},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source",
             "reasoning": "Looking at the trace, step 1 already made a successful edit. I need to verify it."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "post_mutation_unverified"
        assert out["process"]["pending_edit_step"] is None

    def test_applied_edit_recap_with_have_been_is_not_pending_edit(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"python - <<'PY'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nPY\"",
             "result_summary": "",
             "source_write_like": True,
             "source_write_paths": ["src/app.py"]},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p src/app.py'",
             "result_summary": "source",
             "reasoning": "Looking at the trace, changes have already been made to src/app.py, so I should verify the patch."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "post_mutation_unverified"
        assert out["process"]["pending_edit_step"] is None

    def test_post_mutation_test_plan_is_not_pending_edit(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"python - <<'PY'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nPY\"",
             "result_summary": "",
             "source_write_like": True,
             "source_write_paths": ["src/app.py"]},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "cmd='git diff src/app.py'",
             "result_summary": "diff",
             "reasoning": "Let me check the current state of the fix and try running the tests with the correct Python environment."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "post_mutation_unverified"
        assert out["process"]["pending_edit_step"] is None

    def test_truncated_python_command_counts_as_verification(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash",
             "args_summary": "cmd=\"python - <<'PY'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nPY\"",
             "result_summary": "",
             "source_write_like": True,
             "source_write_paths": ["src/app.py"]},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash",
             "args_summary": "cmd='cd /testbed && python3 -c \"import sympy...",
             "result_summary": "ImportError: cannot import name Mapping"},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["phase"] == "post_verification"
        assert out["process"]["last_verification_step"] == 2

    def test_go_and_rust_file_tokens_are_recognized_as_target_paths(self):
        """_FILE_TOKEN_RE now reuses bash_write_classification's superset of
        source extensions (F10), so non-Python source paths surface too."""
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p foo.go'",
             "result_summary": "source"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "cmd='sed -n 1,40p bar.rs'",
             "result_summary": "source",
             "reasoning": "The fix should replace the old branch in bar.rs."},
        ]
        out = project(events, max_result_chars=_CAP, imperative_projection=True)

        assert out["process"]["target_paths"] == ["foo.go", "bar.rs"]

    def test_go_rust_js_test_runners_count_as_verification(self):
        """_VERIFICATION_RE (F10) now recognizes jest/vitest/npx-jest/ctest/
        pnpm-test/yarn-test in addition to the existing go/cargo/npm test."""
        for verify_cmd in (
            "go test ./...", "cargo test", "npx jest", "vitest run",
            "yarn test", "pnpm test", "ctest --output-on-failure",
        ):
            events = [
                {"event": "tool_call", "session_number": 1, "turn_number": 0,
                 "tool_name": "bash",
                 "args_summary": "cmd=\"cd /testbed && sed -i 's/old/new/' src/app.py\"",
                 "result_summary": "",
                 "reasoning": "The fix is to change old to new in src/app.py."},
                {"event": "tool_call", "session_number": 1, "turn_number": 1,
                 "tool_name": "bash", "args_summary": f"cmd='{verify_cmd}'",
                 "result_summary": "ok"},
            ]
            out = project(events, max_result_chars=_CAP, imperative_projection=True)
            assert out["process"]["phase"] == "post_verification", verify_cmd

    def test_default_projection_does_not_add_process_block(self):
        out = _project([
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='ls'",
             "result_summary": "x"},
        ])
        assert out["meta"]["schema_version"] == 1
        assert "process" not in out


class TestProjectTruncation:
    def test_long_args_summary_is_truncated(self):
        long_args = "x" * 500
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": long_args,
             "result_summary": "r"},
        ]
        out = _project(events)
        entry = out["trace"][0]
        assert len(entry["action"]) < 500
        assert entry["action"].endswith("...)")

    def test_long_result_summary_is_truncated(self):
        long_result = "y" * 25000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "short",
             "result_summary": long_result},
        ]
        out = _project(events)
        assert len(out["trace"][0]["result"]) < 25000
        assert len(out["trace"][0]["result"]) == _CAP
        assert out["trace"][0]["result"].endswith("...")

    def test_mid_length_result_is_not_truncated(self):
        code_read = "def foo():\n    pass\n" * 250
        assert len(code_read) == 5000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='foo.py'",
             "result_summary": code_read},
        ]
        out = _project(events)
        assert out["trace"][0]["result"] == code_read
        assert not out["trace"][0]["result"].endswith("...")


class TestProjectDeterminism:
    def test_same_input_yields_same_output(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "r"},
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 1},
        ]
        assert _project(events) == _project(events)


class TestProjectFromTrace:
    def test_missing_file_yields_empty_schema(self, tmp_path: Path):
        out = _project_from_trace(tmp_path / "does-not-exist.jsonl")
        assert out["state"]["current_attempt"] == ""
        assert out["trace"] == []

    def test_reads_multiple_events_from_file(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(
            "\n".join([
                json.dumps({"event": "session_start", "session_number": 1}),
                json.dumps({
                    "event": "tool_call", "session_number": 1, "turn_number": 0,
                    "tool_name": "bash", "args_summary": "echo hi",
                    "result_summary": "hi",
                }),
            ]) + "\n"
        )
        out = _project_from_trace(trace)
        assert len(out["trace"]) == 1
        assert out["state"]["current_attempt"] == "bash(echo hi)"

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("\n\n" + json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "a", "result_summary": "r",
        }) + "\n\n")
        out = _project_from_trace(trace)
        assert len(out["trace"]) == 1


class TestWriteStateFromTrace:
    def test_writes_state_to_target_path(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "ls", "result_summary": "r",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        assert state_path.is_file()
        data = json.loads(state_path.read_text())
        assert len(data["trace"]) == 1

    def test_creates_parent_directories(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / "a" / "b" / "c" / "state.json"
        _write_state_from_trace(trace, state_path)
        assert state_path.is_file()

    def test_idempotent_overwrite(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "a", "result_summary": "r",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        first = state_path.read_text()
        _write_state_from_trace(trace, state_path)
        second = state_path.read_text()
        assert first == second

    def test_no_tmp_file_left_behind(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        leftover = list((tmp_path / ".solver").glob("*.tmp"))
        assert leftover == []

    def test_round_trip_through_solver_state_context_schema(self, tmp_path: Path):
        """The projection output must match the keys SolverStateContext reads."""
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "echo",
            "result_summary": "out",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        data = json.loads(state_path.read_text())
        # The output includes the top-level meta block.
        assert set(data.keys()) == {
            "meta", "state", "tools", "trace", "gates", "evidence",
            "inference",
        }
        assert set(data["state"].keys()) >= {"current_attempt", "last_verify", "next_action"}
        for entry in data["trace"]:
            assert {"step", "action", "result", "next"} <= set(entry.keys())
        for ev in data["evidence"]:
            assert {"step", "action", "result", "verdict"} <= set(ev.keys())
