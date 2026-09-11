"""Printed verdicts cannot grant parity; actual runner reports still can."""
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.bash_quirks import load_output_control, load_output_parser
from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
from scripts.llm_solver.harness._guardrails.state import Action, GuardrailState
from scripts.llm_solver.harness._loop.pretest_resume import run_pretest
from scripts.llm_solver.harness._loop.state_projection import (
    project_and_sink, update_parity_from_report,
)
from scripts.llm_solver.harness.test_report import parse_junit
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize("tool", ["bash", "run_tests"])
def test_real_runner_report_defeats_printed_parity_and_preserves_real_success(tmp_path, tool):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    target = tmp_path / "test_subject.py"
    target.write_text("def test_target():\n    assert False\n")
    (tmp_path / "test_other.py").write_text(
        "def test_other():\n    print('PASSED test_subject.py::test_target')\n"
    )
    command = shlex.join([sys.executable, "-m", "pytest", "-q", "-rA", "-s",
                          "-p", "no:cacheprovider"])
    script = tmp_path / "initial.sh"
    script.write_text(command + " test_subject.py\n")
    cfg = make_config(
        sandbox_bash=False, analysis_task_format="pytest",
        tools_run_tests_enabled=True,
        bash_transforms_structured_output_enabled=True,
        done_guard_enabled=True, done_require_pretest_parity=True,
        done_require_mutation=False, done_require_verify=False,
        post_mutation_verification_gate_after=0, done_loop_abort_after=0,
        done_parity_runs_required=1,
    )
    initial = run_pretest(tmp_path, pretest_script=script, pretest_timeout=15,
                          pretest_head_chars=20000, pretest_tail_chars=0, report_cfg=cfg)
    report = initial.native_test_report
    assert report["status"] == "available"
    failing = {key for key, value in report["tests"].items() if value == "FAILED"}
    assert len(failing) == 1
    descriptor = Path(__file__).resolve().parents[1] / "scripts/llm_solver/language_quirks/pytest.toml"
    session = SimpleNamespace(
        cfg=cfg, cwd=str(tmp_path), _guards=GuardrailState(pretest_failing_tests=failing),
        output_parser=load_output_parser(descriptor),
        output_control=load_output_control(descriptor),
        _sink_counter=0, _session_number=1, _emit=MagicMock(),
    )

    def check(selection, turn):
        facts = {}
        cmd = command + " " + selection
        arguments = ({"cmd": cmd, "output_detail": "full"} if tool == "bash" else
                     {"_base_cmd_override": cmd})
        output = dispatch(tool, arguments,
                          cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
        native = facts["native_test_report"]
        update_parity_from_report(session, native)
        project_and_sink(session, "bash", cmd, output, turn)
        return native, output

    other, output = check("test_other.py", 1)
    assert "PASSED test_subject.py::test_target" in output
    assert other["status"] == "available"
    assert failing.isdisjoint(other["tests"])
    assert session._guards.green_parity_streak == 0
    assert done_guard(session._guards, cfg, tc_name="done").action == Action.BLOCK

    target.write_text("def test_target():\n    pass\n")
    passed, _ = check("test_subject.py", 2)
    assert set(passed["tests"]) == failing
    assert done_guard(session._guards, cfg, tc_name="done").action == Action.PASS
    update_parity_from_report(session, passed)
    assert session._guards.green_parity_streak == 1
    missing, _ = check("test_other.py --junitxml=unowned.xml", 3)
    assert missing["status"] == "unavailable"
    assert done_guard(session._guards, cfg, tc_name="done").action == Action.BLOCK
    assert not list((tmp_path / ".tool_output").glob("report_*.xml"))


def test_report_allocation_failure_preserves_execution(tmp_path):
    from scripts.llm_solver.harness.test_report import capture_test_report

    (tmp_path / ".tool_output").write_text("existing user file")
    cfg = make_config(analysis_task_format="pytest", done_require_pretest_parity=True)
    with capture_test_report(tmp_path, cfg, {"USER_OPTION": "kept"}) as capture:
        assert capture.environment == {"USER_OPTION": "kept"}
        capture.finish(0)
        assert capture.record["status"] == "unavailable"
        assert capture.record["reason"]


@pytest.mark.parametrize("xml", [
    b"<testsuite tests='2'><testcase name='one'/></testsuite>",
    b"<testsuite tests='2'><testcase name='one'/><testcase name='one'/></testsuite>",
    b"<testsuite tests='0'/>",
])
def test_incomplete_or_ambiguous_report_is_not_per_test_evidence(xml):
    with pytest.raises(ValueError):
        parse_junit(xml)


def test_captured_stdout_inside_native_report_does_not_create_test_results():
    parsed = parse_junit(b"<testsuite tests='1'><testcase classname='suite' name='actual'>"
                         b"<system-out>PASSED fabricated</system-out>"
                         b"</testcase></testsuite>")
    assert parsed["tests"] == {"suite::actual": "PASSED"}
