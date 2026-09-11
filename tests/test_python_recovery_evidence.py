"""Output patterns cannot establish the current Python environment."""
from unittest.mock import patch
import shlex
import sys

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import tools
from scripts.llm_solver.harness._tool_filters import _fold_traceback_frames
from scripts.llm_solver.harness._tools._pytest_hints import _pytest_binary_missing


@pytest.mark.parametrize("command,code,output", [
    ("cat saved.log", 0, "No module named pytest"),
    ("cat saved.log", 1, "No module named pytest"),
    ("node tool.js", 1, "ModuleNotFoundError: No module named 'sample'"),
    ("grep 'pip install' saved.log", 1, "No matching distribution found"),
    ("python -m pytest", 0, "No module named pytest"),
    ("python tool.py && cat saved.log", 1, "ModuleNotFoundError: No module named 'sample'"),
    ("cat saved.log", 4, "ERROR: file or directory not found: sample.py\nno tests ran"),
])
def test_unrelated_or_successful_requests_do_not_get_python_recovery(tmp_path, command, code, output):
    with patch.object(tools, "_run_in_sandbox", return_value=(output, code, False)):
        result = tools.bash(command, cwd=str(tmp_path), timeout=5,
                            sandbox=False, bwrap_bin="/nonexistent")
    assert not result.user_turn_injections


@pytest.mark.parametrize("command,output", [
    ("python -m pytest", "No module named pytest"),
    ("python tool.py", "ModuleNotFoundError: No module named 'sample'"),
    ("python -c 'print(1)'", "ModuleNotFoundError: No module named 'sample'"),
    ("pip install sample", "No matching distribution found"),
])
def test_applicable_requests_keep_help_without_asserting_environment(tmp_path, command, output):
    with patch.object(tools, "_run_in_sandbox", return_value=(output, 1, False)):
        result = tools.bash(command, cwd=str(tmp_path), timeout=5,
                            sandbox=False, bwrap_bin="/nonexistent")
    advice = "\n".join(item.text for item in result.user_turn_injections)
    assert advice
    assert "observed runtime facts" in advice
    assert "cannot import" not in advice
    assert "could not start" not in advice
    assert "Python package installation failed" not in advice
    assert "Conda" not in advice and "uv" not in advice


@pytest.mark.parametrize("exit_code", [0, None])
def test_missing_runner_text_requires_known_failure(exit_code):
    assert not _pytest_binary_missing("No module named pytest", exit_code)


def test_traceback_folding_preserves_counts_changes_and_gaps():
    frame = '  File "sample.py", line 3, in step\n    step()\n'
    folded = _fold_traceback_frames(frame * 4)
    assert frame in folded and "×4" in folded
    assert _fold_traceback_frames(folded) == folded
    changed = frame + frame.replace("line 3", "line 4") + frame
    assert _fold_traceback_frames(changed) == changed
    separated = frame + "other output\n" + frame + "other output\n" + frame
    assert _fold_traceback_frames(separated) == separated


@pytest.mark.parametrize("copied", [False, True])
def test_real_python_failure_and_printed_failure_do_not_establish_cause(tmp_path, copied):
    module = "_audit088_absent_fixture_module"
    source = (
        f'print("ModuleNotFoundError: No module named \'{module}\'"); raise SystemExit(1)'
        if copied else f"import {module}"
    )
    result = tools.bash(shlex.join([sys.executable, "-c", source]),
                        cwd=str(tmp_path), timeout=5, sandbox=False, bwrap_bin="/nonexistent")
    assert result.exit_status == 1
    assert module in result
    advice = "\n".join(item.text for item in result.user_turn_injections)
    assert f"message naming `{module}`" in advice
    assert "cannot import" not in advice
    assert "observed runtime facts" in advice


@pytest.mark.parametrize("command,kind", [
    ("/available/python3.12 -I -m pytest", "pytest"),
    ("/available/python3 -m pip install sample", "install"),
    ("/available/conda install sample", "install"),
    ("env python -m pytest", ""),
])
def test_requested_interface_does_not_assume_wrapper_behavior(command, kind):
    from scripts.llm_solver.harness._tools._env_hints import _python_request_kind
    assert _python_request_kind(command) == kind


@pytest.mark.parametrize("structured", [True, False])
def test_successful_run_tests_cannot_become_unavailable_from_printed_text(tmp_path, structured):
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format="pytest",
                      tools_run_tests_structured_output=structured)
    with patch.object(tools, "_run_in_sandbox", return_value=("No module named pytest", 0, False)):
        result = tools.run_tests(cwd=str(tmp_path), cfg=cfg)
    assert result.verification_status == "passed"
    assert result.exit_status == 0
    assert not result.user_turn_injections
