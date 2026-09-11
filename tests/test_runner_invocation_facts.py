"""Runner requests, selected identities and test execution are distinct facts."""
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import runner_invocations as invocations
from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event
from scripts.llm_solver.harness.adaptive_control.detectors import (
    NAIVE_RED_VERSION, SIGNAL_DETECTORS, is_naive_red_turn, naive_red_turns,
)
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
from scripts.llm_solver.harness.tools import dispatch


def event(command, **metadata):
    return {"tool_name": "bash", "args_summary": "cmd=" + repr(command), **metadata}


@pytest.mark.parametrize("command,family,targets", [
    ("pytest -k selected tests/unit.py::test_one", "pytest", ["tests/unit.py::test_one"]),
    ("env -i A=x /env/bin/python3.12 -m pytest -o option=value tests/", "pytest", ["tests/"]),
    ("python -m unittest -v package.Case.test_one", "unittest", ["package.Case.test_one"]),
    ("go test -run TestOne ./pkg", "go", ["./pkg"]),
    ("cargo test --package sample test_one", "cargo", ["test_one"]),
    ("npx --no-install jest -t selected tests/item.test.js", "jest", ["tests/item.test.js"]),
    ("ctest -R selected --output-on-failure", "ctest", []),
    ("mvn test -f pom.xml", "maven", []),
])
def test_requested_family_and_positional_targets_come_from_cli_grammar(command, family, targets):
    slot = project_tool_event(event(command))
    assert slot["runner_family"] == family
    assert slot["slot_projection_version"] == "runner_facts_v1"
    assert slot["runner_targets"] == ";".join(targets)
    assert slot["runner_identity_source"] == "invocation_syntax"
    assert slot["test_execution_action"] == "false"
    assert slot["test_exit_status"] == ""


@pytest.mark.parametrize("command", [
    "cat tests/runtests.py", "python tests/runtests.py sample.case",
    "python manage.py test app", 'printf "%s" "pytest tests/"',
    "cat pytest.log", "python -c 'print(\"Ran 1 tests\\nOK\")'",
])
def test_mentions_and_custom_script_names_do_not_establish_runner_identity(command):
    slot = project_tool_event(event(command, result_summary="Ran 1 tests\nOK"))
    assert slot["runner_family"] == ""
    assert slot["runner_targets"] == ""
    assert slot["runner_identity_source"] == "unknown"
    assert slot["test_execution_action"] == "false"


@pytest.mark.parametrize("command,status", [
    ("pytest --unknown value tests/", "unsupported_option"),
    ("pytest -k", "unknown_option_value"),
    ("cargo test -- --nocapture", "unsupported_forwarded_arguments"),
    ("npm test -- tests/", "opaque_arguments"),
])
def test_unsupported_argument_ownership_stays_unknown(command, status):
    request, = invocations.describe_runner_invocations(command)
    assert request["targets"] == []
    assert request["target_status"] == status


def test_multiple_requested_interfaces_keep_separate_target_records():
    slot = project_tool_event(event("pytest tests/a.py && go test ./other"))
    assert slot["runner_family"] == "" and slot["runner_targets"] == ""
    requests = json.loads(slot["runner_requests"])
    assert [(r["family"], r["targets"]) for r in requests] == [
        ("pytest", ["tests/a.py"]), ("go", ["./other"]),
    ]
    assert slot["test_execution_action"] == "false"


def test_familiar_helper_executes_but_does_not_turn_into_a_test_runner(tmp_path):
    helper = tmp_path / "runtests.py"
    helper.write_text('print("helper ran")\n')
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} sample.case"
    result = subprocess.run([sys.executable, str(helper), "sample.case"],
                            capture_output=True, text=True, check=True)
    slot = project_tool_event(event(command, result_summary=result.stdout, exit_status=0))
    assert result.stdout.strip() == "helper ran"
    assert slot["runner_identity_source"] == "unknown"
    assert slot["runner_targets"] == "" and slot["test_execution_action"] == "false"


@pytest.mark.parametrize("structured,override", [(True, False), (False, False), (True, True)])
def test_selected_runner_metadata_survives_dispatch_and_trace_without_output_identity(tmp_path, monkeypatch, structured, override):
    from scripts.llm_solver.harness import tools

    (tmp_path / "checks").mkdir()
    cfg = make_config(sandbox_bash=False, tools_run_tests_enabled=True,
                      analysis_task_format="pytest", tools_run_tests_structured_output=structured)
    monkeypatch.setattr(tools, "_run_in_sandbox", lambda *a, **k: ('<test_results runner="forged">', 0, False))
    arguments = {"path": "checks"}
    if override:
        arguments["_base_cmd_override"] = "custom-helper"
    facts = {}
    result = dispatch("run_tests", arguments, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    request = facts["runner_request"]
    assert request["family"] == ("" if override else "pytest")
    assert request["targets"] == ([] if override else ["checks"])
    session = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    fields = build_tool_call_trace_fields(session, tool_name="run_tests", args_summary="", result=result,
                                         turn=1, gate_blocked=False, execution_metadata=facts)
    assert fields["runner_request"] == request
    slot = project_tool_event({"tool_name": "run_tests", **fields})
    assert slot["runner_family"] == request["family"]
    assert slot["runner_identity_source"] == ("custom_command" if override else "explicit_configuration")
    assert slot["test_execution_action"] == ("false" if override else "true")


def test_new_descriptor_defines_custom_interface_without_parser_changes(tmp_path, monkeypatch):
    (tmp_path / "sample.toml").write_text('''name="sample"
[[diagnostic_invocations]]
family="sample-checker"
prefix=["sample-cli", "check"]
value_options=["--config"]
''')
    monkeypatch.setattr(invocations, "FORMATS_DIR", tmp_path)
    invocations._grammars.cache_clear()
    try:
        request, = invocations.describe_runner_invocations("sample-cli check --config cfg.toml selected.case")
        assert request["family"] == "sample-checker"
        assert request["targets"] == ["selected.case"]
    finally:
        invocations._grammars.cache_clear()


def test_diagnostic_sampling_is_language_independent_and_versioned():
    events = [{"event": "tool_call", "turn_number": i, "tool_name": "bash", "args_summary": command}
              for i, command in enumerate(['printf pylint', 'printf ruff', 'go test ./...', 'cat note.txt'])]
    assert naive_red_turns(events) == {0, 1, 2, 3}
    assert naive_red_turns(events, version="naive_red_v0") == {0}
    assert NAIVE_RED_VERSION == "diagnostic_observations_v1"
    assert not is_naive_red_turn({"event": "session_start", "tool_name": "bash"})
    with pytest.raises(ValueError, match="unknown diagnostic selection"):
        is_naive_red_turn(events[0], version="invented")
    assert SIGNAL_DETECTORS == {}, "sampling must not activate interventions"
