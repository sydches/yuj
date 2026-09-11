"""Runner changes must reach each command-tagging consumer consistently."""
import pytest

from scripts.llm_solver.harness import _shell_patterns, state_writer
from scripts.llm_solver.harness._guardrails.extractors import _is_test_command
from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event
from scripts.llm_solver.harness.context_strategies._solver_state_helpers import _classify_cmd
from scripts.llm_solver.harness.context_strategies.salience_context import SalienceContext
from scripts.llm_solver.language_quirks import detect_runner, load_run_tests_quirk_object


@pytest.mark.parametrize("command", [
    "python -m pytest tests/", "python manage.py test app", "tox -e test",
    "go test ./...", "cargo test", "pnpm test", "npx --no-install jest",
    "ctest --output-on-failure", "make check", "mvn test",
    "cat note.txt; cargo test", "rg value src/ && go test ./...",
    "head -1 note.txt && python -m pytest tests/",
])
def test_registered_commands_reach_all_consumers(command):
    item = {"action": f"bash(cmd={command!r})"}
    assert state_writer._is_check_item(item)
    assert SalienceContext._is_verification_item(item)
    assert not state_writer._is_read_only_item(item)
    assert not SalienceContext._is_read_only_item(item)
    assert _classify_cmd(command) == "test"
    assert _is_test_command("bash", {"cmd": command})
    slot = project_tool_event({"tool_name": "bash", "args_summary": command})
    assert slot["test_like_action"] == "true"
    assert slot["test_execution_action"] == "false"


@pytest.mark.parametrize("prefix", ["", "cat note.txt; ", "rg value src/ && "])
def test_custom_probe_is_context_evidence_but_not_a_registered_runner(prefix):
    command = prefix + 'python -c "assert 1 + 1 == 2"'
    item = {"action": f"bash(cmd={command!r})"}
    assert state_writer._is_check_item(item)
    assert SalienceContext._is_verification_item(item)
    assert not _is_test_command("bash", {"cmd": command})
    assert _classify_cmd(command) != "test"
    slot = project_tool_event({"tool_name": "bash", "args_summary": command})
    assert slot["test_execution_action"] == "false"


def test_reading_runner_output_does_not_become_an_executed_test():
    command = "cat pytest.log"
    item = {"action": f"bash(cmd={command!r})"}
    assert not state_writer._is_check_item(item)
    assert not SalienceContext._is_verification_item(item)
    assert not _is_test_command("bash", {"cmd": command})
    assert _classify_cmd(command) == "read"
    slot = project_tool_event({"tool_name": "bash", "args_summary": command,
                               "result_summary": "1 passed"})
    assert slot["test_execution_action"] == "false"


@pytest.mark.parametrize("command", ["cat note.txt; cargo test", "rg value src/ && go test ./..."])
def test_recognized_mixed_call_retains_unresolved_execution_status(command):
    from scripts.llm_solver.harness.shell_verification import shell_verification_status

    status = shell_verification_status(command, 0)
    assert status == "shell_unresolved"
    slot = project_tool_event({
        "tool_name": "bash", "args_summary": command, "result_summary": "1 passed",
        "verification_status": status, "exit_status": 0,
    })
    assert _is_test_command("bash", {"cmd": command})
    assert slot["test_execution_action"] == "false"
    assert slot["test_exit_status"] == ""


@pytest.mark.parametrize("command", [
    'echo "pytest tests/"', 'python -c "print(\'pytest\')"',
    "cat <<'EOF'\npytest tests/\nEOF", "# pytest tests/\ncat result.txt",
    "echo harmless # comment; pytest tests/",
    'python3 -m "pytest nonsense"', 'npm "test nonsense"',
    "pytest.log", "cargo test-helper", "npm test:watch",
])
def test_mentions_are_not_formal_runner_invocations(command):
    assert not _is_test_command("bash", {"cmd": command})
    slot = project_tool_event({"tool_name": "bash", "args_summary": f"cmd={command!r}"})
    assert slot["test_execution_action"] == "false"


@pytest.mark.parametrize("command", [
    "env -- /opt/env/bin/python3.12 -m pytest tests/",
    "echo '<<'; pytest tests/",
    "echo harmless # comment; ignored\npytest tests/",
    "echo '#'; pytest tests/",
])
def test_shell_syntax_keeps_real_invocations(command):
    assert _is_test_command("bash", {"cmd": command})


def test_quoted_module_and_runner_suffix_are_not_context_checks():
    for command in ['python3 -m "pytest nonsense"', 'pytest.log']:
        item = {"action": f"bash(cmd={command!r})"}
        assert not state_writer._is_check_item(item)
        assert not SalienceContext._is_verification_item(item)


def test_shell_execution_agrees_with_literal_invocation_recognition(tmp_path):
    """Run a harmless fake runner to distinguish syntax from printed mentions."""
    import os
    import subprocess

    marker = tmp_path / "invoked"
    runner = tmp_path / "pytest"
    runner.write_text('#!/bin/sh\nprintf invoked > "$MARKER"\n')
    runner.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin", "MARKER": str(marker)}
    for command, invoked in [
        ('echo "pytest tests/"', False),
        ('echo harmless # comment; pytest tests/', False),
        ("cat <<'EOF'\npytest tests/\nEOF", False),
        ("echo '<<'; pytest tests/", True),
        ('env -- pytest tests/', True),
        ('pytest tests/', True),
        ('cat /dev/null; pytest tests/', True),
    ]:
        if marker.exists():
            marker.unlink()
        result = subprocess.run(['bash', '-c', command], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0
        assert marker.exists() is invoked
        assert _is_test_command("bash", {"cmd": command}) is invoked


def test_new_descriptor_reaches_consumers_in_a_fresh_process(tmp_path):
    """Adding vocabulary needs no per-consumer edit or analysis task format."""
    import subprocess
    import sys
    from pathlib import Path

    formats = tmp_path / "formats"
    formats.mkdir()
    (formats / "sample.toml").write_text(
        'name = "sample"\nrecognition_only = true\n'
        "verification_patterns = ['^sample_runner\\s+--check']\n"
    )
    script = r'''
import sys
from pathlib import Path
from scripts.llm_solver import language_quirks
language_quirks.FORMATS_DIR = Path(sys.argv[1])
language_quirks._load_runner_quirk_dict.cache_clear()
language_quirks.all_verification_patterns.cache_clear()
language_quirks.all_custom_check_patterns.cache_clear()
from scripts.llm_solver.harness import state_writer
from scripts.llm_solver.harness._guardrails.extractors import _is_test_command
from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event
from scripts.llm_solver.harness.context_strategies._solver_state_helpers import _classify_cmd
from scripts.llm_solver.harness.context_strategies.salience_context import SalienceContext
for command, expected in [
    ('sample_runner --check', True),
    ('cat note.txt; sample_runner --check', True),
    ('rg term src/ && sample_runner --check', True),
    ('echo "sample_runner --check"', False),
    ('cat sample_runner', False),
    ('sample_runner --check-extra', False),
    ('pytest tests/', False),
]:
    item = {'action': f'bash(cmd={command!r})'}
    assert state_writer._is_check_item(item) is expected, command
    assert SalienceContext._is_verification_item(item) is expected, command
    assert (_classify_cmd(command) == 'test') is expected, command
    assert _is_test_command('bash', {'cmd': command}) is expected, command
    slot = project_tool_event({'tool_name': 'bash', 'args_summary': command})
    assert (slot['test_like_action'] == 'true') is expected, command
    assert slot['test_execution_action'] == 'false', command
print('seven commands agree across five consumers without an analysis format')
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(formats)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_environment_and_absolute_interpreter_are_recognized():
    command = "source /opt/env/activate; cd /workspace && env -i KEY=value /opt/env/bin/python3.12 -m pytest tests/"
    assert _is_test_command("bash", {"cmd": command})
    slot = project_tool_event({"tool_name": "bash", "args_summary": f"cmd={command!r}"})
    assert slot["test_like_action"] == "true"
    assert slot["test_execution_action"] == "false"


def test_recognition_only_descriptor_cannot_invent_an_automatic_runner(tmp_path):
    (tmp_path / "pom.xml").write_text("<project/>")
    assert detect_runner(tmp_path) == "generic"
    assert load_run_tests_quirk_object(tmp_path).base_cmd == ""


def test_absent_runner_patterns_never_fall_back_to_pytest(monkeypatch):
    monkeypatch.setattr(_shell_patterns, "all_verification_patterns", lambda: ())
    assert not _shell_patterns._build_test_command_re().search("pytest tests/")


def test_broken_runner_data_does_not_select_a_builtin_fallback(monkeypatch):
    def broken():
        raise OSError("unreadable descriptor")
    monkeypatch.setattr(_shell_patterns, "all_verification_patterns", broken)
    with pytest.raises(OSError, match="unreadable descriptor"):
        _shell_patterns._build_test_command_re()
