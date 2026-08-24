"""Tests for the compound_selective context mode."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.llm_solver.harness.context_strategies import (
    CompoundSelectiveContext,
    SalienceContext,
)


def _make_context(
    tmp_path: Path,
    *,
    tool_budget: int = 60,
    trace_repeat_cap: int = 1,
    resolved_repeat_cap: int = 1,
    trace_lines: int = 2,
    resolved_lines: int = 2,
    trace_anchor_lines: int = 0,
    resolved_anchor_lines: int = 0,
    trace_source_anchor_lines: int = 0,
    trace_test_anchor_lines: int = 0,
    resolved_source_anchor_lines: int = 0,
    resolved_test_anchor_lines: int = 0,
) -> CompoundSelectiveContext:
    ctx = CompoundSelectiveContext(
        cwd=str(tmp_path),
        original_prompt="Solve the task.",
        trace_lines=50,
        evidence_lines=30,
        inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        min_turns=2,
        suffix="Continue working. Your progress is tracked in .solver/state.json.",
        selective_trace_lines=trace_lines,
        selective_unresolved_evidence_lines=2,
        selective_resolved_evidence_lines=resolved_lines,
        selective_resolved_evidence_stub_chars=40,
        selective_recent_tool_results_chars=tool_budget,
        selective_trace_action_repeat_cap=trace_repeat_cap,
        selective_resolved_action_repeat_cap=resolved_repeat_cap,
        selective_trace_anchor_lines=trace_anchor_lines,
        selective_resolved_anchor_lines=resolved_anchor_lines,
        selective_trace_source_anchor_lines=trace_source_anchor_lines,
        selective_trace_test_anchor_lines=trace_test_anchor_lines,
        selective_resolved_source_anchor_lines=resolved_source_anchor_lines,
        selective_resolved_test_anchor_lines=resolved_test_anchor_lines,
    )
    ctx.add_system("You are a solver.")
    ctx._turn_count = 5
    return ctx


def _make_salience_context(
    tmp_path: Path,
    *,
    trace_lines: int = 6,
    ignore_state: bool = False,
) -> SalienceContext:
    ctx = SalienceContext(
        cwd=str(tmp_path),
        original_prompt="Solve the task.",
        trace_lines=50,
        evidence_lines=30,
        inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        min_turns=2,
        suffix="Continue working. Your progress is tracked in .solver/state.json.",
        selective_trace_lines=trace_lines,
        selective_unresolved_evidence_lines=2,
        selective_resolved_evidence_lines=2,
        selective_resolved_evidence_stub_chars=40,
        selective_recent_tool_results_chars=60,
        selective_trace_action_repeat_cap=1,
        selective_resolved_action_repeat_cap=1,
        selective_trace_anchor_lines=0,
        selective_resolved_anchor_lines=0,
        selective_trace_source_anchor_lines=0,
        selective_trace_test_anchor_lines=0,
        selective_resolved_source_anchor_lines=0,
        selective_resolved_test_anchor_lines=0,
        ignore_state=ignore_state,
    )
    ctx.add_system("You are a solver.")
    ctx._turn_count = 5
    return ctx


def _write_state(tmp_path: Path, state: dict) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / "state.json").write_text(json.dumps(state))


def test_selective_trace_budget_is_smaller(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "first", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "second", "action": "read(path='b.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "third", "action": "edit(path='c.py')", "result": "OK", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path)
    user_text = ctx.get_messages()[1]["content"]
    assert "read(path='a.py')" not in user_text
    assert "read(path='b.py')" in user_text
    assert "edit(path='c.py')" in user_text


def test_selective_evidence_keeps_fails_and_stubbed_resolved_tail(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "bash(cmd='find a')", "result": "OLDEST_RESOLVED_" + ("x" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "bash(cmd='find b')", "result": "KEEP_RESOLVED_ONE_" + ("y" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "bash(cmd='find c')", "result": "KEEP_RESOLVED_TWO_" + ("z" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "bash(cmd='pytest a')", "result": "FAIL_ONE\ntraceback line", "verdict": "FAIL", "gate_blocked": False},
            {"step": 5, "action": "bash(cmd='pytest b')", "result": "FAIL_TWO\nassert line", "verdict": "FAIL", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(tmp_path)
    user_text = ctx.get_messages()[1]["content"]
    assert "FAIL_ONE\ntraceback line" in user_text
    assert "FAIL_TWO\nassert line" in user_text
    assert "OLDEST_RESOLVED_" not in user_text
    assert "KEEP_RESOLVED_ONE_" in user_text
    assert "KEEP_RESOLVED_TWO_" in user_text
    assert "KEEP_RESOLVED_ONE_" + ("y" * 50) not in user_text
    assert "-- resolved --" in user_text


def test_selective_trace_keeps_diverse_actions_under_repeat_cap(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "bash(cmd='find tests')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 4, "session": 1, "turn": 3, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 5, "session": 1, "turn": 4, "reasoning": "", "action": "grep(pattern='foo')", "result": "", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path, trace_lines=3, trace_anchor_lines=1)
    user_text = ctx.get_messages()[1]["content"]
    assert "bash(cmd='find tests')" in user_text
    assert user_text.count("read(path='a.py')") == 1
    assert "grep(pattern='foo')" in user_text


def test_selective_trace_preserves_source_and_test_transition_anchors(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd='grep -n \"target\" tests/test_module.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 4, "session": 1, "turn": 3, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 5, "session": 1, "turn": 4, "reasoning": "", "action": "grep(pattern='symbol', path='src')", "result": "", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(
        tmp_path,
        trace_lines=3,
        trace_repeat_cap=1,
        trace_anchor_lines=0,
        trace_source_anchor_lines=1,
        trace_test_anchor_lines=1,
    )
    user_text = ctx.get_messages()[1]["content"]
    assert "read(path='src/module.py')" in user_text
    assert "grep -n \"target\" tests/test_module.py" in user_text
    assert "grep(pattern='symbol', path='src')" in user_text


def test_selective_resolved_evidence_keeps_diverse_actions_under_repeat_cap(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "bash(cmd='find tests')", "result": "FOUND_TEST", "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "read(path='build.py')", "result": "READ_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "read(path='build.py')", "result": "READ_TWO", "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "grep(pattern='get_parser')", "result": "FOUND_SYMBOL", "verdict": "OK", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(tmp_path, resolved_lines=3, resolved_anchor_lines=1)
    user_text = ctx.get_messages()[1]["content"]
    assert "FOUND_TEST" in user_text
    assert "FOUND_SYMBOL" in user_text
    assert user_text.count("read(path='build.py')") == 1
    assert "READ_TWO" in user_text
    assert "READ_ONE" not in user_text


def test_selective_resolved_evidence_preserves_test_anchor(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "read(path='src/module.py')", "result": "SOURCE_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "bash(cmd='pytest -q tests/test_module.py')", "result": "VERIFY_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "read(path='src/module.py')", "result": "SOURCE_TWO", "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "grep(pattern='symbol', path='src')", "result": "RECENT_SEARCH", "verdict": "OK", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(
        tmp_path,
        resolved_lines=3,
        resolved_repeat_cap=1,
        resolved_anchor_lines=0,
        resolved_source_anchor_lines=1,
        resolved_test_anchor_lines=1,
    )
    user_text = ctx.get_messages()[1]["content"]
    assert "SOURCE_ONE" in user_text
    assert "VERIFY_ONE" in user_text
    assert "RECENT_SEARCH" in user_text


def test_selective_tool_result_window_is_smaller(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path, tool_budget=50)
    older = "OLDER_RESULT_" + ("x" * 30)
    newer = "NEWER_RESULT_" + ("y" * 30)
    ctx.add_tool_result("call-1", older, tool_name="bash", cmd_signature='{"cmd":"echo older"}')
    ctx.add_tool_result("call-2", newer, tool_name="bash", cmd_signature='{"cmd":"echo newer"}')
    user_text = ctx.get_messages()[1]["content"]
    assert "NEWER_RESULT_" in user_text
    assert "OLDER_RESULT_" not in user_text


def test_salience_pressure_surfaces_no_mutation_test_and_repeats(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "read(path='src/a.py')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "read(path='src/b.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "", "action": "bash(cmd='grep -n foo src/a.py')", "result": "", "next": ""},
            {"step": 4, "session": 1, "turn": 3, "reasoning": "", "action": "bash(cmd='grep -n foo src/a.py')", "result": "", "next": ""},
            {"step": 5, "session": 1, "turn": 4, "reasoning": "", "action": "bash(cmd='grep -n foo src/a.py')", "result": "", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=5)
    ctx._turn_count = 20

    user_text = ctx.get_messages()[1]["content"]
    assert "=== Salience Pressure ===" in user_text
    assert "No file-mutation action" in user_text
    assert "No test/verification-like command" in user_text
    assert "Newest action repeated 3 consecutive times" in user_text


def test_salience_classifies_double_quoted_bash_actions(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "bash(cmd=\"cd /testbed && sed -i 's/a/b/' pkg/mod.py\")", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd=\"cd /testbed && python -m pytest tests/test_mod.py\")", "result": "FAILED", "next": ""},
        ] + [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "",
                "action": f"bash(cmd=\"sed -n '{idx},{idx + 5}p' pkg/mod.py\")",
                "result": "",
                "next": "",
            }
            for idx in range(3, 31)
        ],
        "gates": [],
        "evidence": [
            {"step": 2, "action": "bash(cmd=\"cd /testbed && python -m pytest tests/test_mod.py\")", "result": "FAILED", "verdict": "FAIL", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=8)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "No file-mutation action" not in user_text
    assert "No test/verification-like command" not in user_text
    assert "29 steps since the last file mutation." in user_text
    assert "28 steps since the last test/verification-like command." in user_text
    assert "sed -i" in user_text
    assert "python -m pytest tests/test_mod.py" in user_text


def test_salience_pressure_flags_read_search_loop(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "",
                "action": f"bash(cmd=\"sed -n '{idx},{idx + 5}p' pkg/mod.py\")",
                "result": "",
                "next": "",
            }
            for idx in range(1, 13)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Recent actions are read/search-only (12/12)." in user_text
    assert "avoid another broad cat/sed/grep/read" in user_text
    assert "Calling done now would submit an empty patch" in user_text
    assert "edit through bash with a short python/perl/sed script" in user_text


def test_salience_hard_read_loop_does_not_invite_reproducer(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "",
                "action": "bash(cmd='cat pkg/mod.py')",
                "result": "source",
                "next": "",
            }
            for idx in range(1, 13)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "hard read/search loop before any recorded source mutation" in user_text
    assert "next move should make a minimal source edit" in user_text
    assert "tiny reproducer" not in user_text


def test_salience_classifies_truncated_python_command_as_verification(tmp_path: Path):
    item = {
        "action": "bash(cmd='cd /testbed && python3 -c \"import sympy...",
        "result": "ImportError: cannot import name Mapping",
    }
    assert SalienceContext._extract_action_cmd(item).startswith(
        "cd /testbed && python3 -c"
    )
    assert SalienceContext._is_verification_item(item)


def test_salience_pressure_surfaces_pending_edit_intent(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": (
                    "I need to apply the fix by changing the helper to use "
                    "the subclass instead of the base class."
                    if idx >= 8
                    else "I am inspecting the implementation."
                ),
                "action": "bash(cmd='grep -n \"helper\" pkg/mod.py')",
                "result": "",
                "next": "",
            }
            for idx in range(1, 13)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Latest model intent says it is ready to edit" in user_text
    assert "changing the helper" in user_text
    assert "Repeated read/search actions already seen" in user_text


def test_salience_trace_compresses_recent_read_only_loop(tmp_path: Path):
    trace = [
        {
            "step": idx,
            "session": 1,
            "turn": idx,
            "reasoning": "",
            "action": "bash(cmd='cat pkg/mod.py')",
            "result": "SOURCE",
            "next": "",
        }
        for idx in range(1, 13)
    ]
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": trace,
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    rendered_trace = ctx._format_salience_trace(trace, [], trace_limit=6)
    assert rendered_trace.count("bash(cmd='cat pkg/mod.py')") <= 2


def test_salience_suppresses_raw_tool_results_during_repeated_read_loop(tmp_path: Path):
    trace = [
        {
            "step": idx,
            "session": 1,
            "turn": idx,
            "reasoning": "",
            "action": "bash(cmd='cat pkg/mod.py')",
            "result": "SOURCE",
            "next": "",
        }
        for idx in range(1, 13)
    ]
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": trace,
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30
    ctx.add_tool_result(
        "call-1",
        "FULL_SOURCE_CONTENT_SHOULD_NOT_REPEAT",
        tool_name="bash",
        cmd_signature='{"cmd":"cat pkg/mod.py"}',
    )

    user_text = ctx.get_messages()[1]["content"]
    assert "=== Tool results suppressed ===" in user_text
    assert "FULL_SOURCE_CONTENT_SHOULD_NOT_REPEAT" not in user_text
    assert "same read/search actions are looping" in user_text


def test_salience_pressure_surfaces_patch_hygiene_from_mutations_and_diff(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "I should edit the main implementation.",
                "action": "bash(cmd=\"cd /testbed && sed -i 's/a/b/' pkg/mod.py\")",
                "result": "",
                "next": "",
            },
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "The local probe has an import issue.",
                "action": "bash(cmd=\"cd /testbed && sed -i 's/from collections import Mapping/from collections.abc import Mapping/' pkg/compat.py\")",
                "result": "",
                "next": "",
            },
            {
                "step": 3,
                "session": 1,
                "turn": 3,
                "reasoning": "Let me inspect the diff again.",
                "action": "bash(cmd='cd /testbed && git diff --stat')",
                "result": " pkg/mod.py    | 2 +-\n pkg/compat.py | 2 +-\n 2 files changed, 2 insertions(+), 2 deletions(-)",
                "next": "",
            },
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Mutation targets recorded: pkg/mod.py, pkg/compat.py." in user_text
    assert "Latest diff paths visible in trace: pkg/mod.py, pkg/compat.py." in user_text
    assert "Patch hygiene: keep only task-relevant source edits" in user_text
    assert "mutations without a later verification-like command" in user_text


def test_salience_keeps_the_trace_when_it_fits_the_budget(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": idx, "session": 1, "turn": idx, "reasoning": "", "action": f"read(path='old_{idx}.py')", "result": f"OLD_{idx}", "next": ""}
            for idx in range(1, 12)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=4)
    ctx._turn_count = 80

    user_text = ctx.get_messages()[1]["content"]
    assert "read(path='old_1.py')" in user_text
    assert "read(path='old_11.py')" in user_text


def test_salience_action_contract_frontloads_bash_write_skeleton(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")", "result": "def helper():\n    return Base(value)", "next": ""},
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "The fix is to change helper to use cls instead of Base when creating the object.",
                "action": "bash(cmd='grep -n helper pkg/mod.py')",
                "result": "100:def helper():",
                "next": "",
            },
        ] + [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "I am ready to apply the edit.",
                "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")",
                "result": "def helper():\n    return Base(value)",
                "next": "",
            }
            for idx in range(3, 13)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "=== Next Action Contract ===" in user_text
    assert "pending source mutation" in user_text
    assert "next bash command must write the target source file" in user_text
    assert "do not use cat, sed -n, grep" in user_text
    assert "cd /testbed && python - <<'PY'" in user_text
    assert "os.replace(tmp, path)" in user_text
    assert "path = Path('pkg/mod.py')" in user_text
    assert "path.write_text" not in user_text


def test_salience_action_contract_recovers_from_failed_bash_write(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")", "result": "def helper():\n    return Base(value)", "next": ""},
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "The fix is to change helper to use cls instead of Base when creating the object.",
                "action": "bash(cmd=\"cd /testbed && python - <<'PY'\nfrom pathlib import Path\nPath('pkg/mod.py').write_text('x')\nPY\")",
                "result": "PermissionError: [Errno 13] Permission denied: 'pkg/mod.py'\n[exit code: 1]",
                "next": "",
            },
            {"step": 3, "session": 1, "turn": 3, "reasoning": "", "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")", "result": "def helper():\n    return Base(value)", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "last source write attempt failed" in user_text
    assert "no source mutation is recorded" in user_text
    assert "PermissionError" in user_text
    assert "retry the source edit" in user_text
    assert "not run verification" in user_text
    assert "Status: source mutation exists and needs verification" not in user_text


def test_salience_uses_trace_source_write_metadata(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "The fix is to change helper to use cls instead of Base.",
                "action": "bash(cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\")",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["pkg/mod.py"],
            },
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "Looking at the trace, step 1 already made a successful edit. I need to verify it.",
                "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")",
                "result": "def helper():\n    return cls(value)",
                "next": "",
            },
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Status: source mutation exists and needs verification" in user_text
    assert "patch paths: pkg/mod.py" in user_text
    assert "read/search loop without a recorded source mutation" not in user_text
    assert "Latest model intent says it is ready to edit" not in user_text


def test_salience_applied_edit_recap_phrase_is_not_edit_intent(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "The fix is to change helper to use cls instead of Base.",
                "action": "bash(cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\")",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["pkg/mod.py"],
            },
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "Looking at the trace, changes have already been made to pkg/mod.py, so I should verify the patch.",
                "action": "bash(cmd=\"sed -n '90,110p' pkg/mod.py\")",
                "result": "def helper():\n    return cls(value)",
                "next": "",
            },
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Status: source mutation exists and needs verification" in user_text
    assert "Latest model intent says it is ready to edit" not in user_text
    assert "pending source mutation" not in user_text


def test_salience_post_mutation_test_intent_is_not_edit_intent(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "The fix is to change helper to use cls instead of Base.",
                "action": "bash(cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\")",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["pkg/mod.py"],
            },
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "Let me check the current state of the fix and try running the tests with the correct Python environment.",
                "action": "bash(cmd='git diff pkg/mod.py')",
                "result": "diff --git a/pkg/mod.py b/pkg/mod.py\n",
                "next": "",
            },
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "Latest model intent says it is ready to edit" not in user_text
    assert "pending source mutation" not in user_text


def test_salience_post_verification_stall_contract(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "The fix is to change helper to use cls instead of Base.",
                "action": "bash(cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\")",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["pkg/mod.py", "sympy/core/basic.py"],
            },
            {
                "step": 2,
                "session": 1,
                "turn": 2,
                "reasoning": "Now I need to verify the fix.",
                "action": "bash(cmd='python -m pytest tests/test_mod.py')",
                "result": "FAILED",
                "next": "",
            },
        ] + [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "Looking at the trace, changes have already been made.",
                "action": "bash(cmd='git diff -- pkg/mod.py | head -200')",
                "result": "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n",
                "next": "",
            }
            for idx in range(3, 26)
        ],
        "gates": [],
        "evidence": [
            {
                "step": 2,
                "action": "bash(cmd='python -m pytest tests/test_mod.py')",
                "result": "FAILED",
                "verdict": "FAIL",
                "gate_blocked": False,
            },
        ],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=8)
    ctx._turn_count = 60

    user_text = ctx.get_messages()[1]["content"]
    assert "Status: source mutation has been verified or probed" in user_text
    assert "latest verification/probe step: 2" in user_text
    assert "patch paths: pkg/mod.py, sympy/core/basic.py" in user_text
    assert "repeated post-mutation inspections" in user_text
    assert "remove unrelated compatibility/setup edits" in user_text
    assert "call done if the current source patch is intended" in user_text
    assert "Latest model intent says it is ready to edit" not in user_text


def test_salience_repeated_verification_requires_revision(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "The fix is to change helper to use cls instead of Base.",
                "action": "bash(cmd=\"cd /testbed && python3 << 'PY'\\nfrom pathlib import Path\\n...\")",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["pkg/mod.py"],
            },
        ] + [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "Let me run the same focused test again.",
                "action": "bash(cmd='python -m pytest tests/test_mod.py::test_case -q')",
                "result": "FAILED tests/test_mod.py::test_case\nAssertionError: concrete failure",
                "next": "",
            }
            for idx in range(2, 7)
        ],
        "gates": [],
        "evidence": [
            {
                "step": 6,
                "action": "bash(cmd='python -m pytest tests/test_mod.py::test_case -q')",
                "result": "FAILED tests/test_mod.py::test_case\nAssertionError: concrete failure",
                "verdict": "FAIL",
                "gate_blocked": False,
            },
        ],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=8)
    ctx._turn_count = 60

    user_text = ctx.get_messages()[1]["content"]
    assert "source mutation already has repeated verification/probe results" in user_text
    assert "verification/probe count after mutation: 5" in user_text
    assert "repeated verification/probe commands" in user_text
    assert "revise the source patch from the concrete failure" in user_text
    assert "do not rerun the same verification/probe" in user_text


def test_salience_env_blocker_discourages_setup_only_patch(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "I changed an import so the local probe can run.",
                "action": "bash(cmd='python - <<\\'PY\\'\\nfrom pathlib import Path\\nPath(\"sympy/core/basic.py\").write_text(\"x\")\\nPY')",
                "result": "SUCCESS",
                "next": "",
                "source_write_like": True,
                "source_write_paths": ["sympy/core/basic.py"],
            },
        ] + [
            {
                "step": idx,
                "session": 1,
                "turn": idx,
                "reasoning": "I will try the reproducer again.",
                "action": "bash(cmd='python -c \"import sympy\"')",
                "result": "ImportError: cannot import name 'Mapping' from 'collections'\n[exit code: 1]",
                "next": "",
            }
            for idx in range(2, 6)
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=8)
    ctx._turn_count = 60

    user_text = ctx.get_messages()[1]["content"]
    assert "verification blocker looks environmental" in user_text
    assert "do not patch unrelated compatibility/import files" in user_text


def test_salience_candidate_edit_ignores_exploratory_reasoning(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 1,
                "reasoning": "I need to understand how Complement handles mixed symbols. Let me examine the relevant code.",
                "action": "bash(cmd=\"sed -n '100,180p' pkg/mod.py\")",
                "result": "source",
                "next": "",
            },
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=4)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "pending source mutation" not in user_text
    assert "next bash command must write" not in user_text


def test_salience_action_contract_does_not_label_plain_failure_as_gate(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd='python repro.py')", "result": "Traceback", "next": ""},
        ],
        "gates": [],
        "evidence": [
            {
                "step": 1,
                "action": "bash(cmd='python repro.py')",
                "result": "Traceback",
                "verdict": "FAIL",
                "gate_blocked": False,
            },
        ],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, trace_lines=6)
    ctx._turn_count = 30

    user_text = ctx.get_messages()[1]["content"]
    assert "=== Gate (blocking) ===" not in user_text
    assert "=== Evidence ===" in user_text


def test_salience_ignore_state_uses_append_log_not_synthetic_projection(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "STATE_SHOULD_NOT_APPEAR"},
        "trace": [],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_salience_context(tmp_path, ignore_state=True)
    ctx.add_assistant({
        "role": "assistant",
        "content": "I will inspect a file.",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"cmd": "sed -n '1,20p' pkg/module.py"}),
            },
        }],
    })
    ctx.add_tool_result(
        "call-1",
        "READ_RESULT",
        tool_name="bash",
        cmd_signature='{"cmd": "sed"}',
    )

    messages = ctx.get_messages()
    assert [m["role"] for m in messages] == ["system", "assistant", "tool"]
    rendered = json.dumps(messages)
    assert "STATE_SHOULD_NOT_APPEAR" not in rendered
    assert "Task: Solve the task." not in rendered
    assert "READ_RESULT" in messages[-1]["content"]
