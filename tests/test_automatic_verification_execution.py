"""Automatic checks require a completed submission and available run time."""
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.guardrails import init_guardrail_state, mark_bash_verified
from scripts.llm_solver.harness._guardrails.verification import (
    observe_post_mutation_verification, observed_component_runner_base_cmd,
)
from scripts.llm_solver.harness._loop._dispatch_tool_call import _run_automatic_component_verification
from scripts.llm_solver.harness.time_budget import run_time_budget


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    cfg = make_config(sandbox_bash=False, analysis_task_format="pytest",
                      post_mutation_verification_gate_after=2)
    guards = init_guardrail_state(cfg)
    guards.has_mutated = True
    (tmp_path / "core.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text("def test_core():\n    assert True\n")
    guards.post_mutation_source_paths = ("core.py",)
    return tmp_path, cfg, guards


def observed(rig, command, *, original=None):
    root, cfg, guards = rig
    facts = {}
    output = dispatch("bash", {"cmd": command}, cwd=str(root), cfg=cfg, execution_metadata=facts)
    observe_post_mutation_verification(guards, cfg, tc_name="bash", result=output,
        gate_blocked=False, tc_args={"cmd": original or command}, execution_metadata=facts, cwd=str(root))
    return facts


@pytest.mark.parametrize("operator", ["true ||", "false &&"])
def test_skipped_executable_never_counts_or_supplies_runtime(rig, operator):
    root, _, guards = rig
    executable = root / "python"
    executable.write_text("#!/bin/sh\ntouch invoked\n")
    executable.chmod(0o755)
    for _ in range(3):
        facts = observed(rig, f"{operator} {shlex.quote(str(executable))}")
        assert "shell_submission" not in facts
    assert not (root / "invoked").exists()
    assert guards.post_mutation_non_test_bash_count == 0
    assert guards.post_mutation_observed_runtime_executable == ""
    assert not guards.post_mutation_verification_gate_armed


def test_completed_submission_wins_over_old_request_and_stdout_markers(rig):
    _, cfg, guards = rig
    actual = shlex.quote(sys.executable) + " -c 'print(\"[exit code: 127]\"); raise SystemExit(1)'"
    for _ in range(2):
        facts = observed(rig, actual, original="echo never-called-request")
        assert facts["exit_status"] == 1
    assert guards.post_mutation_non_test_bash_count == 2
    assert guards.post_mutation_verification_gate_armed
    assert guards.post_mutation_observed_runtime_executable == sys.executable
    assert observed_component_runner_base_cmd(guards, "pytest", cwd=str(rig[0]), cfg=cfg).startswith(sys.executable)
    from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
    fields = build_tool_call_trace_fields(
        SimpleNamespace(cfg=cfg, cwd=str(rig[0]), _sink_counter=0, _session_number=1),
        tool_name="bash", args_summary="original request", result="display",
        turn=1, gate_blocked=False, execution_metadata=facts,
    )
    assert fields["shell_submission"] == facts["shell_submission"]
    assert "arguments" not in fields["shell_submission"]


def test_printed_success_cannot_override_native_unavailable_exit(rig):
    command = shlex.quote(sys.executable) + " -c 'print(\"passed\"); raise SystemExit(127)'"
    observed(rig, command)
    assert rig[2].post_mutation_non_test_bash_count == 0


def test_inline_environment_does_not_reuse_an_earlier_runtime_spelling(rig):
    command = shlex.quote(sys.executable) + " -c 'assert True'"
    observed(rig, command)
    assert rig[2].post_mutation_observed_runtime_executable
    observed(rig, "FIXTURE_VALUE=1 " + command)
    assert rig[2].post_mutation_non_test_bash_count == 2
    assert rig[2].post_mutation_observed_runtime_executable == ""


def test_runtime_reuse_requires_current_task_and_container_binding(rig, monkeypatch):
    root, cfg, guards = rig
    observed(rig, shlex.quote(sys.executable) + " -c 'assert True'")
    assert not observed_component_runner_base_cmd(guards, "pytest", cwd=str(root.parent), cfg=cfg)
    monkeypatch.setenv("YUJ_CONTAINER", "not-contacted")
    assert not observed_component_runner_base_cmd(guards, "pytest", cwd=str(root), cfg=cfg)


def test_native_collection_error_invalidates_earlier_formal_pass(rig):
    root, cfg, guards = rig
    guards.formal_verification_passed_since_mutation = True
    guards.verified_since_mutation = True
    observe_post_mutation_verification(guards, cfg, tc_name="bash", result="display is not authority",
        gate_blocked=False, tc_args={"cmd": "old request"}, cwd=root,
        execution_metadata={"executed": True, "exit_status_known": True, "exit_status": 2,
                            "verification_status": "collection_error",
                            "runner_request": {"check_intent": True}})
    assert not guards.formal_verification_passed_since_mutation
    assert not guards.verified_since_mutation


@pytest.mark.parametrize("change", [{"executed": False}, {"timed_out": True},
                                   {"security_blocked_stage": "result"}, {"exit_status_known": False}])
def test_incomplete_or_blocked_metadata_does_not_count(rig, change):
    from scripts.llm_solver.harness.runner_invocations import describe_shell_submission
    root, cfg, guards = rig
    facts = dict(executed=True, exit_status_known=True, exit_status=0,
                 verification_status="custom_passed",
                 shell_submission=describe_shell_submission("python -c 'assert True'", root))
    facts.update(change)
    observe_post_mutation_verification(guards, cfg, tc_name="bash", result="passed",
        gate_blocked=False, tc_args={"cmd": "python -c 'assert True'"}, execution_metadata=facts, cwd=root)
    assert guards.post_mutation_non_test_bash_count == 0


def automatic_state(rig):
    root, cfg, guards = rig
    guards.post_mutation_verification_gate_armed = True
    session = SimpleNamespace(_guards=guards, cfg=cfg, _emit=Mock(), cwd=str(root), _ignore_policy=None,
        output_control=None, universal_rewrites=None, forbidden_rules=None,
        redactions=None, _tool_registry=None, _effective_env=None, _allow_login_shell=False,
        _queue_user_turn_injection=Mock(), _queue_execution_user_turn_injections=Mock())
    return SimpleNamespace(session=session, cfg=cfg, turn=1, dispatch=dispatch,
        observers={"mark_bash_verified": mark_bash_verified,
                   "observe_post_mutation_verification": observe_post_mutation_verification})


def test_exhausted_budget_refuses_before_target_discovery(rig, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("scripts.llm_solver.harness.time_budget.time.monotonic", lambda: now[0])
    state = automatic_state(rig)
    metadata = {}
    with run_time_budget(10), patch(
        "scripts.llm_solver.harness._guardrails.verification.resolve_component_verification_target",
        side_effect=AssertionError("discovery must not start"),
    ):
        now[0] = 10
        output, _ = _run_automatic_component_verification(SimpleNamespace(id="call"), state, "custom output", metadata)
    assert metadata["automatic_verification"] == "budget_exhausted"
    assert metadata["automatic_verification_execution_budget"]["effective_seconds"] == 0
    assert "did not execute" in output
    assert not rig[2].post_mutation_automatic_verification_attempted


def test_budget_exhaustion_after_discovery_is_not_a_failed_test(rig, monkeypatch):
    from scripts.llm_solver.harness._guardrails.verification import resolve_component_verification_target
    now = [0.0]
    monkeypatch.setattr("scripts.llm_solver.harness.time_budget.time.monotonic", lambda: now[0])
    state = automatic_state(rig)
    metadata = {}
    def late_target(*args, **kwargs):
        target = resolve_component_verification_target(*args, **kwargs)
        now[0] = 10
        return target
    with run_time_budget(10), patch(
        "scripts.llm_solver.harness._guardrails.verification.resolve_component_verification_target", late_target,
    ), patch("scripts.llm_solver.harness.tools._run_in_sandbox", side_effect=AssertionError("must not launch")):
        output, _ = _run_automatic_component_verification(SimpleNamespace(id="call"), state, "custom output", metadata)
    assert metadata["automatic_verification"] == "budget_exhausted"
    assert "did not execute" in output
    assert not rig[2].post_mutation_automatic_verification_attempted
    assert not state.session._queue_user_turn_injection.called


def test_automatic_run_uses_remaining_time_and_keeps_pending_limits_explicit(rig, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("scripts.llm_solver.harness.time_budget.time.monotonic", lambda: now[0])
    state = automatic_state(rig)
    metadata = {}
    with run_time_budget(10), patch("scripts.llm_solver.harness.tools._run_in_sandbox",
                                   return_value=("1 passed", 0, False)) as backend:
        now[0] = 3
        output, _ = _run_automatic_component_verification(SimpleNamespace(id="call"), state, "custom output", metadata)
    assert backend.call_args.kwargs["timeout"] == 7
    # This backend stub writes no collection report; text cannot supply one.
    assert metadata["automatic_verification"] == "selection_unavailable"
    assert metadata["automatic_verification_execution_budget"]["effective_seconds"] == 7
    assert 'scope="conventional_component" task_requirement="not_checked"' in output


def test_automatic_run_preserves_selected_configuration_and_collects_before_choosing(rig):
    from dataclasses import replace
    root, cfg, guards = rig
    (root / "first.py").write_text("VALUE = 0\n")
    guards.post_mutation_source_paths = ("first.py", "core.py")
    (root / "checks").mkdir()
    (root / "checks/verify_core.py").write_text("def test_declared(): assert False\n")
    (root / "chosen.ini").write_text("[pytest]\ntestpaths = checks\npython_files = verify_*.py\n")
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "-c", "chosen.ini"])
    cfg = replace(cfg, runtime_test_selection={
        "status": "selected", "task_root": str(root),
        "selected": {"runner": "pytest", "base_cmd": command},
    })
    state = automatic_state((root, cfg, guards))
    metadata = {}
    output, _ = _run_automatic_component_verification(SimpleNamespace(id="call"), state, "custom output", metadata)
    assert metadata["automatic_verification"] == "failed"
    assert metadata["component_selection"]["candidates"] == ["checks/verify_core.py"]
    assert metadata["component_selection"]["source"] == "core.py"
    assert metadata["automatic_verification_source"] == "core.py"
    assert metadata["automatic_verification_target"] == "checks/verify_core.py"
    assert not metadata["automatic_verification_base_cmd_reused"]
    assert not guards.formal_verification_passed_since_mutation
    assert "checks/verify_core.py" in output
