"""Regression tests for refusals and safe command rewrites."""
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from scripts.llm_solver.bash_quirks import load_output_control
from scripts.llm_solver.bash_quirks._forbidden import load_forbidden_rules
from scripts.llm_solver.bash_quirks._rewrites import load_universal_rewrites
from scripts.llm_solver.bash_quirks.transforms import (
    OutputControl,
    condense_output,
    rewrite_command,
)
from scripts.llm_solver.harness import tools as tools_mod
from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox

UR = load_universal_rewrites()
FR = load_forbidden_rules()


@pytest.mark.parametrize("flags", ["-xvs", "-x -vv -s -rA", "-x --verbose --capture=no -r fEsxX", "-x --capture no"])
def test_routine_pytest_flags_are_quiet(flags):
    oc = load_output_control(PROJECT_ROOT / "scripts/llm_solver/language_quirks/pytest.toml")
    rewritten = rewrite_command("pytest tests " + flags, oc)
    assert shlex.split(rewritten) == ["pytest", "tests", "-x", "--tb=short", "-q", "--no-header"]
    assert rewrite_command(rewritten, oc) == rewritten


def test_quiet_flags_preserve_selection_and_other_runners():
    oc = load_output_control(PROJECT_ROOT / "scripts/llm_solver/language_quirks/pytest.toml")
    command = 'pytest -k "has -s or -v" -m fast -xvs tests'
    rewritten = rewrite_command(command, oc)
    assert '-k "has -s or -v" -m fast -x tests' in rewritten
    assert rewrite_command("python -m unittest -v", oc) == "python -m unittest -v"
    assert rewrite_command('python -c "run_tests(verbosity=2)"', oc) == 'python -c "run_tests(verbosity=2)"'
    assert shlex.split(rewrite_command("pytest -v -- -s", oc)) == [
        "pytest", "--tb=short", "-q", "--no-header", "--", "-s"
    ]


def test_full_detail_preserves_flags_for_one_call_and_keeps_output_cap(tmp_path):
    oc = load_output_control(PROJECT_ROOT / "scripts/llm_solver/language_quirks/pytest.toml")
    cfg = make_config(max_output_chars=1000, collapse_similar_lines=False)
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return "\n".join(f"build detail {i}" for i in range(200)), 0, False

    with mock.patch.object(tools_mod, "_run_in_sandbox", side_effect=run):
        full = tools_mod.dispatch("bash", {"cmd": "pytest -xvs", "output_detail": "full"},
                                  cwd=str(tmp_path), cfg=cfg, output_control=oc)
        tools_mod.dispatch("bash", {"cmd": "pytest -xvs"},
                           cwd=str(tmp_path), cfg=cfg, output_control=oc)
    assert commands[0] == "pytest -xvs"
    assert shlex.split(commands[1]) == ["pytest", "-x", "--tb=short", "-q", "--no-header"]
    assert "chars omitted" in full

HEREDOC_WRITE = """python << 'EOF'
# make the edit and write it back
with open('/tmp/x.py', 'w') as f:
    f.write('x')
EOF"""

HEREDOC_READ = """python << 'EOF'
# make the edit in memory only
with open('/tmp/x.py', 'r') as f:
    content = f.read()
print(content)
EOF"""

FORBIDDEN_CD = "cd /home/other && pwd"


def test_containment_refusal_surfaces_reason_to_model():
    out = rewrite_command(FORBIDDEN_CD, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    assert out.startswith('echo "[HARNESS refused')
    # and the refusal actually prints the reason and exits nonzero
    r = subprocess.run(out, shell=True, capture_output=True, text=True)
    assert r.returncode != 0
    assert "HARNESS refused" in r.stderr


def test_refusal_shell_safe_despite_quotes_in_reason():
    out = rewrite_command(FORBIDDEN_CD, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    r = subprocess.run(out, shell=True, capture_output=True, text=True)
    # no shell parse error text
    assert "unexpected" not in r.stderr.lower()


def test_external_heredoc_write_passes_through():
    out = rewrite_command(HEREDOC_WRITE, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    assert out == HEREDOC_WRITE


def test_multiline_commands_never_get_flag_appends():
    out = rewrite_command(HEREDOC_READ, None, universal_rewrites=UR,
                          forbidden_rules=FR)
    assert out == HEREDOC_READ  # 'make' in prose must not append -s


def test_single_line_rewrites_still_work():
    out = rewrite_command("pip install requests", None,
                          universal_rewrites=UR, forbidden_rules=FR)
    assert out != "pip install requests"  # -q (or similar) appended


def test_rewrite_targets_matching_compound_fragment():
    assert rewrite_command(
        "pip install demo && echo done", None, universal_rewrites=UR
    ) == "pip install demo -q && echo done"
    assert rewrite_command(
        "echo ok | pip install demo", None, universal_rewrites=UR
    ) == "echo ok | pip install demo -q"


def test_quoted_command_mentions_are_not_rewritten():
    for command in (
        'echo "pip install demo"',
        'grep "npm install" README.md',
        'echo "make all"',
        "which make",
        'echo "x && pip install demo"',
    ):
        assert rewrite_command(command, None, universal_rewrites=UR) == command


def test_rewrites_do_not_inject_pipelines_or_result_caps():
    assert rewrite_command(
        "npx tsc --noEmit", None, universal_rewrites=UR
    ) == "npx tsc --noEmit --pretty false"
    assert rewrite_command(
        "cmake --build build", None, universal_rewrites=UR
    ) == "cmake --build build"
    assert rewrite_command("rg pattern .", None, universal_rewrites=UR) == "rg pattern ."


def test_task_flag_targets_test_fragment_only():
    oc = OutputControl(
        failure_only_flag="--tb=short -q",
        passed_marker="PASSED",
        failed_marker="FAILED",
        verification_patterns=(re.compile(r"(?:^|[\s/'\"])(?:pytest)\b"),),
    )
    assert rewrite_command("cd /repo && pytest tests", oc) == (
        "cd /repo && pytest tests --tb=short -q"
    )
    command = 'echo "pytest tests && echo done"'
    assert rewrite_command(command, oc) == command


def test_task_format_removes_display_only_test_pipeline():
    oc = OutputControl(
        failure_only_flag="--tb=short -q",
        passed_marker="PASSED",
        failed_marker="FAILED",
        verification_patterns=(re.compile(r"^pytest\b"),),
    )
    rules: list[dict] = []
    transformed = rewrite_command(
        'pytest tests -v 2>&1 | grep -i "doctest" | head -20',
        oc,
        rule_log=rules,
    )

    assert transformed == "pytest tests -v 2>&1 --tb=short -q"
    assert rules[-1] == {"kind": "test_output_filter_removed"}


def test_task_format_preserves_side_effecting_or_compound_test_pipeline():
    oc = OutputControl(
        failure_only_flag="--tb=short -q",
        passed_marker="PASSED",
        failed_marker="FAILED",
        verification_patterns=(re.compile(r"^pytest\b"),),
    )

    assert rewrite_command("pytest tests | tee test.log", oc) == (
        "pytest tests --tb=short -q | tee test.log"
    )
    assert rewrite_command("pytest tests | tail -20; echo done", oc) == (
        "pytest tests --tb=short -q | tail -20; echo done"
    )
    assert rewrite_command("pytest tests | grep fail > failures.txt", oc) == (
        "pytest tests --tb=short -q | grep fail > failures.txt"
    )


def test_host_runner_preserves_pipeline_failure_status(tmp_path):
    _output, exit_code, timed_out = _run_in_sandbox(
        "false | true",
        cwd=str(tmp_path),
        timeout=10,
        sandbox=False,
        bwrap_bin="/usr/bin/bwrap",
    )

    assert timed_out is False
    assert exit_code != 0


def test_removed_test_filter_surfaces_complete_failure_summary(tmp_path):
    output_control = load_output_control(
        PROJECT_ROOT / "scripts/llm_solver/language_quirks/pytest.toml"
    )
    cfg = make_config(
        max_output_chars=1000,
        truncate_head_ratio=0.5,
        collapse_similar_lines=False,
    )
    raw = (
        "FAILED tests/test_x.py::test_bad - "
        "AttributeError: missing compile\n"
        + "\n".join(f"trace detail {index}" for index in range(200))
        + "\n1 failed, 1 passed in 0.10s"
    )
    captured: dict = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return raw, 1, False

    with mock.patch.object(tools_mod, "_run_in_sandbox", side_effect=fake_run):
        result = tools_mod.dispatch(
            "bash",
            {
                "cmd": (
                    "python -m pytest tests -v 2>&1 "
                    "| grep -i doctest | head -20"
                )
            },
            cwd=str(tmp_path),
            cfg=cfg,
            output_control=output_control,
        )

    assert "| grep" not in captured["cmd"]
    assert "| head" not in captured["cmd"]
    assert "[test evidence] 1 passed, 1 failed" in result
    assert "first failure: FAILED tests/test_x.py::test_bad" in result
    assert "AttributeError: missing compile" in result
    assert "chars omitted" in result


def test_condensation_does_not_claim_the_suite_passed():
    oc = OutputControl(
        failure_only_flag="",
        passed_marker="PASSED",
        failed_marker="FAILED",
        verification_patterns=(re.compile(r"^pytest\b"),),
    )
    output = "PASSED tests/test_ok.py::test_ok\nFAILED tests/test_bad.py::test_bad\n"
    condensed = condense_output(output, "pytest tests", oc)
    assert "[1 passing-result lines omitted]" in condensed
    assert "tests passed" not in condensed
    assert "FAILED tests/test_bad.py::test_bad" in condensed


def test_rule_log_records_kind():
    log = []
    rewrite_command(FORBIDDEN_CD, None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log == [{"kind": "forbidden"}]
    log = []
    rewrite_command("pip install requests", None, universal_rewrites=UR,
                    forbidden_rules=FR, rule_log=log)
    assert log and log[0]["kind"] == "universal"
