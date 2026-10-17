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


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.parametrize("filter_command", ["tail -30", "grep -E '(OK|FAILED)'"])
def test_prepared_runner_pipeline_uses_actual_pipefail_result(tmp_path, monkeypatch, exit_code, filter_command):
    from _verification_fixture import seed_edited_input, account_check_changes
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    (tmp_path / "runtests.py").write_text(f"print('OK')\nraise SystemExit({exit_code})\n")
    (tmp_path / "module.py").write_text("edited input")
    command = f"cd . && timeout 5 {shlex.quote(sys.executable)} runtests.py 2>&1 | {filter_command}"
    cfg = make_config(sandbox_bash=False, turn_snapshots_enabled=False)
    state = GuardrailState(has_mutated=True)
    seed_edited_input(state, cfg, tmp_path, "module.py")
    facts = {}
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["exit_status"] == exit_code
    assert facts["verification_status"] == ("passed" if exit_code == 0 else "shell_unresolved")
    account_check_changes(state, cfg, tmp_path, "bash", {"cmd": command}, result, facts)
    mark_bash_verified(state, cfg, tc_name="bash", tc_args={"cmd": command}, result=result,
                       gate_blocked=False, execution_metadata=facts, cwd=tmp_path)
    assert state.verified_since_mutation is (exit_code == 0)


@pytest.mark.parametrize("command", [
    "set +o pipefail; pytest | tail", "bash -c 'pytest | tail'",
    "pytest | tail || true", "timeout --help pytest", "timeout unknown pytest",
    "timeout 5 pytest --help | tail", "timeout 5 pytest --collect-only | tail",
    "pytest | sh -c 'true'", "pytest | tail && true", "pytest > report",
    "timeout --preserve-status 5 pytest | tail",
])
def test_pipeline_attribution_does_not_credit_masks_or_non_execution(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status
    assert shell_verification_status(command, 0, pipefail=True) not in {"passed", "custom_passed"}


@pytest.mark.parametrize("command", [
    "timeout 5 cargo test | tail", "timeout --signal TERM --kill-after=2s 5 go test ./... | tail",
    "env KEY=value timeout 5 python runtests.py 2>&1", "timeout 5 env KEY=value python runtests.py",
])
def test_timeout_wrapper_is_independent_of_task_runner(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status
    assert shell_verification_status(command, 0, pipefail=True) == "passed"


@pytest.mark.parametrize("path,previously_edited,passed", [
    (".pytest_cache/v/cache/nodeids", False, True),
    (".pytest_cache/v/cache/nodeids", True, False),
    ("other_module.py", False, False),
    (".pytest_cache/custom_source.py", False, False),
    ("result_images/test_text/result.png", False, False),
])
def test_runner_cache_does_not_hide_real_changed_inputs(tmp_path, monkeypatch, path, previously_edited, passed):
    from _verification_fixture import seed_edited_input, account_check_changes
    from scripts.llm_solver.harness.file_changes import observed_mutation
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before")
    (tmp_path / "module.py").write_text("edited input")
    (tmp_path / "test_check.py").write_text(
        "from pathlib import Path\ndef test_check():\n"
        f"    Path({path!r}).write_text('changed')\n"
    )
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    cfg = make_config(sandbox_bash=False, turn_snapshots_enabled=False)
    state = GuardrailState(has_mutated=True)
    seed_edited_input(state, cfg, tmp_path, path if previously_edited else "module.py")
    facts = {}
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["verification_status"] == "passed"
    assert path in facts["file_changes"]["changed_paths"]
    account_check_changes(state, cfg, tmp_path, "bash", {"cmd": command}, result, facts)
    for observer in (mark_bash_verified, observe_post_mutation_verification):
        observer(state, cfg, tc_name="bash", tc_args={"cmd": command}, result=result,
                 gate_blocked=False, execution_metadata=facts, cwd=tmp_path)
    assert state.verified_since_mutation is passed
    assert state.formal_verification_passed_since_mutation is passed
    assert observed_mutation(facts) is not passed


def test_masked_check_cache_does_not_grant_verification(tmp_path, monkeypatch):
    from _verification_fixture import seed_edited_input, account_check_changes
    from scripts.llm_solver.harness.file_changes import observed_mutation
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    (tmp_path / "test_check.py").write_text("def test_check():\n    assert False\n")
    command = "set +o pipefail; " + shlex.join([sys.executable, "-m", "pytest", "-q"]) + " | wc -l"
    cfg = make_config(sandbox_bash=False, turn_snapshots_enabled=False,
                      bash_transforms_universal_enabled=False)
    state = GuardrailState(has_mutated=True)
    seed_edited_input(state, cfg, tmp_path, "test_check.py")
    facts = {}
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["exit_status"] == 0
    assert facts["verification_status"] not in {"passed", "custom_passed"}
    account_check_changes(state, cfg, tmp_path, "bash", {"cmd": command}, result, facts)
    assert ".pytest_cache/v/cache/nodeids" in facts["file_changes"]["test_artifact_paths"]
    assert not observed_mutation(facts)
    for observer in (mark_bash_verified, observe_post_mutation_verification):
        observer(state, cfg, tc_name="bash", tc_args={"cmd": command}, result=result,
                 gate_blocked=False, execution_metadata=facts, cwd=tmp_path)
    assert not state.verified_since_mutation
    assert not state.formal_verification_passed_since_mutation
