from __future__ import annotations

import json
from pathlib import Path

from scripts.llm_solver.harness.context_strategies import (
    ConciseTranscript,
    SlotTranscript,
    YconciseContext,
    YslotContext,
)


def _assistant(content: str, tool_name: str, tool_args: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_args),
            },
        }],
    }


def _make_concise(
    tmp_path: Path,
    prompt: str = "Fix the bug.",
    *,
    min_turns: int = 2,
    inspect_repeat_threshold: int = 0,
) -> ConciseTranscript:
    ctx = ConciseTranscript(
        cwd=str(tmp_path),
        original_prompt=prompt,
        recent_results_chars=12000,
        trace_reasoning_chars=120,
        min_turns=min_turns,
        args_summary_chars=120,
        inspect_repeat_threshold=inspect_repeat_threshold,
    )
    ctx.add_system("You are a solver.")
    ctx.add_user(prompt)
    return ctx


def _make_yconcise(
    tmp_path: Path,
    prompt: str = "Fix the bug.",
    *,
    min_turns: int = 2,
    inspect_repeat_threshold: int = 0,
) -> YconciseContext:
    ctx = YconciseContext(
        cwd=str(tmp_path),
        original_prompt=prompt,
        trace_lines=20,
        evidence_lines=10,
        recent_tool_results_chars=12000,
        trace_stub_chars=200,
        trace_reasoning_chars=120,
        min_turns=min_turns,
        args_summary_chars=120,
        suffix="Continue working.",
        inspect_repeat_threshold=inspect_repeat_threshold,
    )
    ctx.add_system("You are a solver.")
    ctx.add_user(prompt)
    return ctx


def _make_slot(
    tmp_path: Path,
    prompt: str = "Fix the bug.",
    *,
    min_turns: int = 2,
    inspect_repeat_threshold: int = 0,
    recovery_same_target_threshold: int = 0,
    recovery_verify_repeat_threshold: int = 0,
) -> SlotTranscript:
    ctx = SlotTranscript(
        cwd=str(tmp_path),
        original_prompt=prompt,
        recent_results_chars=12000,
        trace_reasoning_chars=120,
        min_turns=min_turns,
        args_summary_chars=120,
        inspect_repeat_threshold=inspect_repeat_threshold,
        recovery_same_target_threshold=recovery_same_target_threshold,
        recovery_verify_repeat_threshold=recovery_verify_repeat_threshold,
        slot_max_candidates=1,
        slot_inline_files=1,
    )
    ctx.add_system("You are a solver.")
    ctx.add_user(prompt)
    return ctx


def _make_yslot(
    tmp_path: Path,
    prompt: str = "Fix the bug.",
    *,
    min_turns: int = 2,
    inspect_repeat_threshold: int = 0,
    recovery_same_target_threshold: int = 0,
    recovery_verify_repeat_threshold: int = 0,
) -> YslotContext:
    ctx = YslotContext(
        cwd=str(tmp_path),
        original_prompt=prompt,
        trace_lines=20,
        evidence_lines=10,
        recent_tool_results_chars=12000,
        trace_stub_chars=200,
        trace_reasoning_chars=120,
        min_turns=min_turns,
        args_summary_chars=120,
        suffix="Continue working.",
        inspect_repeat_threshold=inspect_repeat_threshold,
        recovery_same_target_threshold=recovery_same_target_threshold,
        recovery_verify_repeat_threshold=recovery_verify_repeat_threshold,
        slot_max_candidates=1,
        slot_inline_files=1,
    )
    ctx.add_system("You are a solver.")
    ctx.add_user(prompt)
    return ctx


def _write_state(tmp_path: Path, state: dict) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / "state.json").write_text(json.dumps(state))


class TestConciseBaseline:
    def test_concise_renders_action_sections_without_loop_banner(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hi')\n")

        ctx = _make_concise(tmp_path, prompt="Repair src/app.py", min_turns=0)

        ctx.add_assistant(_assistant("Read the file first", "read", {"path": "src/app.py"}, "c1"))
        ctx.add_tool_result("c1", "print('hi')\n")

        ctx.add_assistant(_assistant("Run the focused test", "bash", {"cmd": "pytest -q tests/test_app.py"}, "c2"))
        ctx.add_tool_result("c2", "E   AssertionError: SENTINEL_FAIL\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "State:" in user
        assert "Blocking output:" in user
        assert "Files (current content):" in user
        assert "Checks:" in user
        assert "Progress:" in user
        assert "Phase:" in user
        assert "Focus files: src/app.py" in user
        assert "Next obligation:" in user
        assert "SENTINEL_FAIL" in user
        assert "src/app.py" in user
        assert "print('hi')" in user
        assert "Loop detected" not in user

    def test_concise_collapses_repeated_actions(self, tmp_path: Path):
        ctx = _make_concise(tmp_path)
        for idx in range(3):
            ctx.add_assistant(_assistant(
                "Retry the same check",
                "bash",
                {"cmd": "pytest -q tests/test_app.py"},
                f"c{idx}",
            ))
            ctx.add_tool_result(f"c{idx}", "still failing\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "×3" in user
        assert "unchanged" in user

    def test_concise_exits_inspect_phase_after_repeated_inspection(self, tmp_path: Path):
        ctx = _make_concise(
            tmp_path,
            prompt="Inspect seaborn and fix the bug.",
            min_turns=0,
            inspect_repeat_threshold=3,
        )
        for idx in range(3):
            ctx.add_assistant(_assistant(
                "Inspect the package root again",
                "bash",
                {"cmd": "ls -la seaborn/"},
                f"c{idx}",
            ))
            ctx.add_tool_result(f"c{idx}", "listing\n[exit code: 0]")

        user = ctx.get_messages()[1]["content"]
        assert "Working root: . (current directory already set)" in user
        assert "Phase: leave inspection and choose a concrete file or check" in user
        assert "Focus files: seaborn/" in user
        assert "Next obligation: stop repeating seaborn/" in user

    def test_failed_write_does_not_mark_file_changed(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hi')\n")

        ctx = _make_concise(tmp_path, prompt="Repair src/app.py", min_turns=0)
        ctx.add_assistant(_assistant(
            "Try an edit",
            "edit",
            {"path": "src/app.py", "old_str": "missing", "new_str": "x"},
            "c1",
        ))
        ctx.add_tool_result("c1", "ERROR: old_str not found")

        user = ctx.get_messages()[1]["content"]
        assert "Files changed:" not in user
        assert "ERROR: old_str not found" in user


class TestYconciseBaseline:
    def test_yconcise_reads_state_json_and_renders_protocol_sections(self, tmp_path: Path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text("value = 1\n")
        _write_state(tmp_path, {
            "state": {
                "current_attempt": "edit(path='pkg/mod.py')",
                "last_verify": "session 1 ended: gate_escalation after 33 turns",
                "next_action": "",
            },
            "trace": [
                {
                    "step": 1,
                    "session": 1,
                    "turn": 1,
                    "reasoning": "Inspect the module before editing",
                    "action": "read(path='pkg/mod.py')",
                    "result": "value = 1",
                    "next": "",
                    "gate_blocked": False,
                },
                {
                    "step": 2,
                    "session": 1,
                    "turn": 2,
                    "reasoning": "Run the focused test",
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "SENTINEL_TRACE_FAIL\n[exit code: 1]",
                    "next": "",
                    "gate_blocked": False,
                },
            ],
            "gates": [],
            "evidence": [
                {
                    "step": 1,
                    "action": "bash(cmd='python -m compileall pkg')",
                    "result": "ok",
                    "verdict": "OK",
                    "gate_blocked": False,
                },
                {
                    "step": 2,
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "SENTINEL_GATE_FAIL\n[exit code: 1]",
                    "verdict": "FAIL",
                    "gate_blocked": False,
                },
            ],
            "inference": [],
        })

        ctx = _make_yconcise(tmp_path, min_turns=0)
        ctx._ws.record_read("pkg/mod.py", "value = 1\n", turn=1)

        user = ctx.get_messages()[1]["content"]
        assert "=== State ===" in user
        assert "=== Gate (blocking) ===" in user
        assert "=== Evidence ===" in user
        assert "=== Files ===" in user
        assert "=== Trace ===" in user
        assert "Current attempt: edit(path='pkg/mod.py')" in user
        assert "session 1 ended: gate_escalation after 33 turns" in user
        assert "Phase:" in user
        assert "Focus files: pkg/mod.py" in user
        assert "Next obligation:" in user
        assert "SENTINEL_GATE_FAIL" in user
        assert "-- unresolved --" in user
        assert "-- resolved --" in user
        assert "pkg/mod.py" in user
        assert "Inspect the module before editing" in user

    def test_yconcise_falls_back_without_state_json(self, tmp_path: Path):
        ctx = _make_yconcise(tmp_path)
        ctx.add_assistant(_assistant("Read the file", "read", {"path": "a.py"}, "c1"))
        ctx.add_tool_result("c1", "x = 1\n")
        ctx.add_assistant(_assistant("Run the test", "bash", {"cmd": "pytest -q"}, "c2"))
        ctx.add_tool_result("c2", "boom\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "=== State ===" in user
        assert "=== Gate (blocking) ===" in user
        assert "=== Trace ===" in user
        assert "pytest -q" in user

    def test_yconcise_surfaces_test_target_before_more_checks(self, tmp_path: Path):
        ctx = _make_yconcise(tmp_path, min_turns=0)
        ctx.add_assistant(_assistant(
            "Run the focused test",
            "bash",
            {"cmd": "pytest -q tests/test_mod.py"},
            "c1",
        ))
        ctx.add_tool_result("c1", "boom\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "Test target: tests/test_mod.py" in user
        assert "Next obligation: read tests/test_mod.py before another verification run" in user


class TestSlotContract:
    def test_slot_renders_decision_surface_with_single_candidate_file(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('hi')\n")

        ctx = _make_slot(tmp_path, prompt="Repair src/app.py", min_turns=0)
        ctx.add_assistant(_assistant("Read the file first", "read", {"path": "src/app.py"}, "c1"))
        ctx.add_tool_result("c1", "print('hi')\n")
        ctx.add_assistant(_assistant("Run the focused test", "bash", {"cmd": "pytest -q tests/test_app.py"}, "c2"))
        ctx.add_tool_result("c2", "E   AssertionError: SENTINEL_FAIL\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "State:" in user
        assert "Candidate file:" in user
        assert "Blocking output:" in user
        assert "candidate_source: src/app.py" in user
        assert "candidate_test: tests/test_app.py" in user
        assert "next_action:" in user
        assert "src/app.py" in user
        assert "print('hi')" in user
        assert "Checks:" not in user
        assert "Progress:" not in user

    def test_slot_enters_recovery_menu_on_same_target_loop(self, tmp_path: Path):
        ctx = _make_slot(
            tmp_path,
            prompt="Inspect seaborn and fix the bug.",
            min_turns=0,
            inspect_repeat_threshold=3,
            recovery_same_target_threshold=3,
        )
        for idx in range(3):
            ctx.add_assistant(_assistant(
                "Inspect the package root again",
                "bash",
                {"cmd": "ls -la seaborn/"},
                f"c{idx}",
            ))
            ctx.add_tool_result(f"c{idx}", "listing\n[exit code: 0]")

        user = ctx.get_messages()[1]["content"]
        assert "phase: recovery" in user
        assert "stuck_reason: repeated same-target inspection" in user
        assert "focused_target: seaborn/" in user
        assert "candidate_source:" not in user
        assert "allowed_moves:" in user

    def test_slot_does_not_surface_outside_root_as_candidate_source(self, tmp_path: Path):
        ctx = _make_slot(tmp_path, prompt="Fix pandas groupby", min_turns=0)

        ctx.add_assistant(_assistant(
            "Try the wrong absolute path",
            "bash",
            {"cmd": "cd /opt/miniconda3/envs./lib/python3.11/site-packages/pandas/core/groupby && ls -la"},
            "c1",
        ))
        ctx.add_tool_result("c1", "bash: line 1: cd: /opt/miniconda3/...: No such file or directory\n[exit code: 1]")

        user = ctx.get_messages()[1]["content"]
        assert "candidate_source: /opt/miniconda3" not in user
        assert "next_action: edit /opt/miniconda3" not in user

    def test_yslot_reads_state_and_surfaces_recovery_from_verify_repeats(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {
                "current_attempt": "edit(path='pkg/mod.py')",
                "last_verify": "pytest -q tests/test_mod.py",
                "next_action": "",
            },
            "trace": [
                {
                    "step": 1,
                    "session": 1,
                    "turn": 1,
                    "reasoning": "Run the focused test",
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "boom\n[exit code: 1]",
                    "next": "",
                    "gate_blocked": False,
                },
                {
                    "step": 2,
                    "session": 1,
                    "turn": 2,
                    "reasoning": "Run the focused test again",
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "boom\n[exit code: 1]",
                    "next": "",
                    "gate_blocked": False,
                },
                {
                    "step": 3,
                    "session": 1,
                    "turn": 3,
                    "reasoning": "Run the focused test again",
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "boom\n[exit code: 1]",
                    "next": "",
                    "gate_blocked": False,
                },
            ],
            "gates": [],
            "evidence": [
                {
                    "step": 3,
                    "action": "bash(cmd='pytest -q tests/test_mod.py')",
                    "result": "boom\n[exit code: 1]",
                    "verdict": "FAIL",
                    "gate_blocked": False,
                },
            ],
            "inference": [],
        })

        ctx = _make_yslot(
            tmp_path,
            min_turns=0,
            recovery_verify_repeat_threshold=3,
        )
        user = ctx.get_messages()[1]["content"]
        assert "=== State ===" in user
        assert "phase: recovery" in user
        assert "stuck_reason: repeated verification without refinement" in user
        assert "candidate_test: tests/test_mod.py" in user
