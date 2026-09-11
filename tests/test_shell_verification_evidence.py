"""A recognized command and successful shell are not proof a check passed."""
import shlex
import sys

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._guardrails.checks_post import mark_bash_verified
from scripts.llm_solver.harness._guardrails.state import GuardrailState
from scripts.llm_solver.harness._guardrails.verification import observe_post_mutation_verification
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize("case", ["verbose_read", "skipped_runner", "masked_failure", "quiet_custom"])
def test_actual_shell_evidence_controls_verification(tmp_path, monkeypatch, case):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    # A harmless runner fixture records whether shell control flow reached it.
    runner = tmp_path / "pytest"
    runner.write_text('#!/bin/sh\nprintf invoked > invocation\nexit 1\n')
    runner.chmod(0o755)
    (tmp_path / "notes.txt").write_text("\n".join(
        f"Documentation item {index}: a permitted task fact" for index in range(20)
    ))
    commands = {
        "verbose_read": "cat notes.txt",
        "skipped_runner": "true || " + shlex.quote(str(runner)),
        "masked_failure": shlex.quote(str(runner)) + " || true",
        "quiet_custom": shlex.join([sys.executable, "-c", "assert 2 + 2 == 4"]),
    }
    cfg = make_config(sandbox_bash=False, bash_transforms_universal_enabled=False,
                      bash_quirks_forbidden_enabled=False)
    facts = {}
    args = {"cmd": commands[case]}
    result = dispatch("bash", args, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["executed"] and facts["exit_status"] == 0
    if case == "verbose_read":
        assert len(result) > cfg.done_verified_bash_min_chars
    assert (tmp_path / "invocation").exists() is (case == "masked_failure")
    state = GuardrailState(has_mutated=True)
    for observer in (mark_bash_verified, observe_post_mutation_verification):
        observer(state, cfg, tc_name="bash", tc_args=args, result=result,
                 gate_blocked=False, execution_metadata=facts, cwd=tmp_path)
    assert state.verified_since_mutation is (case == "quiet_custom")
    assert state.formal_verification_passed_since_mutation is False


@pytest.mark.parametrize("case", ["quiet_pass", "failed", "empty", "conjunction"])
def test_actual_registered_shell_check_reaches_trace_and_observers(tmp_path, monkeypatch, case):
    from _verification_fixture import seed_edited_input, account_check_changes
    from types import SimpleNamespace
    from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
    from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    if case != "empty":
        (tmp_path / "test_check.py").write_text(f"def test_check():\n    assert {case != 'failed'}\n")
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    if case == "conjunction":
        command = "cd . && " + command
    cfg = make_config(sandbox_bash=False, turn_snapshots_enabled=False)
    state = GuardrailState(has_mutated=True)
    seed_edited_input(state, cfg, tmp_path, 'test_check.py')
    facts = {}
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    passed = case in {"quiet_pass", "conjunction"}
    assert facts["verification_status"] == ("passed" if passed else "failed")
    account_check_changes(state, cfg, tmp_path, 'bash', {'cmd': command}, result, facts)
    for observer in (mark_bash_verified, observe_post_mutation_verification):
        observer(state, cfg, tc_name="bash", tc_args={"cmd": command}, result=result,
                 gate_blocked=False, execution_metadata=facts, cwd=tmp_path)
    assert state.verified_since_mutation is passed
    assert state.formal_verification_passed_since_mutation is passed
    session = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    fields = build_tool_call_trace_fields(session, tool_name="bash", args_summary=f"cmd={command!r}",
                                         result=result, turn=1, gate_blocked=False, execution_metadata=facts)
    assert fields["verification_status"] == facts["verification_status"]
    slot = project_tool_event({"tool_name": "bash", "args_summary": f"cmd={command!r}", **fields})
    assert slot["test_execution_action"] == "true"
    assert slot["test_exit_status"] == ("pass" if passed else "fail")


@pytest.mark.parametrize("command", [
    "true || pytest", "pytest || true", "pytest; true", "pytest &", "! pytest",
    "if false; then pytest; fi", "echo $(pytest)", "pytest $(false)",
    "CHOICE=$(false) pytest", "pytest > output", "pytest | tail", "pytest # skipped comment",
    "pytest --h*", "pytest tests/[ab].py", "pytest ~/tests",
])
def test_unattributable_shell_result_stays_unresolved(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status
    assert shell_verification_status(command, 0) not in {"passed", "custom_passed"}


@pytest.mark.parametrize("command", ["pytest --help", "pytest --collect-only", "cargo test --help",
                                     "ctest -N", "go test -list=.", "npx jest --listTests"])
def test_runner_inspection_modes_do_not_become_executed_checks(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status
    assert shell_verification_status(command, 0) == "not_a_check"


@pytest.mark.parametrize("command", ["cargo test --no-run", "go test -c"])
def test_compile_only_success_is_useful_without_becoming_test_execution(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status
    assert shell_verification_status(command, 0) == "custom_passed"
