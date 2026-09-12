"""Inspection advice follows executed requests and delivered file revisions."""
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.guardrails import init_guardrail_state, Action
from scripts.llm_solver.harness._guardrails.test_inspection import (
    test_read_ladder as advise, observe_test_file_read,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.task_file_runtime import task_file_scope


@pytest.mark.parametrize('explicit', [False, True])
def test_public_loader_fallback_preserves_directory_qualification(tmp_path, monkeypatch, explicit):
    from scripts.llm_solver.config import load_config

    root = Path(__file__).resolve().parents[1]
    source = (root / 'config.toml').read_text()
    expected = tomllib.loads(source)['prompts']['test_read_nudge']
    source = source.replace('patterns_file = "security/patterns.toml"',
                            f'patterns_file = "{root / "security/patterns.toml"}"')
    source = source.replace('test_read_warn_after = 0', 'test_read_warn_after = 1')
    custom = 'Owner-selected inspection advice for {target}'
    source = ''.join((f'test_read_nudge = "{custom}"\n' if explicit else '')
                     if line.startswith('test_read_nudge = ') else line
                     for line in source.splitlines(keepends=True))
    config = tmp_path / 'alternate.toml'
    config.write_text(source)
    monkeypatch.setenv('YUJ_CONFIG', str(config))
    monkeypatch.setenv('YUJ_CONFIG_LOCAL', str(tmp_path / 'absent.local.toml'))
    # This suite already imported config; resolve its startup paths again with
    # the fixture environment, as a fresh process would.
    from scripts.llm_solver import config as config_module
    from scripts.llm_solver._shared.paths import default_config_path, local_config_path
    monkeypatch.setattr(config_module, '_DEFAULT_CONFIG', default_config_path())
    monkeypatch.setattr(config_module, '_LOCAL_CONFIG', local_config_path())
    cfg = load_config()
    assert cfg.test_read_nudge == (custom if explicit else expected)
    if not explicit:
        assert make_config().test_read_nudge == expected
    (tmp_path / 'checks').mkdir()
    facts = dict(executed=True, exit_status_known=True, exit_status=0,
        verification_status='passed', timed_out=False,
        runner_request=dict(family='pytest', targets=['checks'], target_status='recorded',
            check_intent=True, workspace_cwd=str(tmp_path), workspace_namespace='local_filesystem'))
    decision = advise(init_guardrail_state(cfg), cfg, tc_name='bash', gate_blocked=False,
                      execution_metadata=facts, cwd=str(tmp_path))
    assert decision.action == Action.WARN
    if explicit:
        assert decision.text == custom.format(target='checks')
    else:
        assert 'directory' in decision.text
        assert 'do not establish suite coverage' in decision.text
        assert 'does not establish understanding or per-test execution' in decision.text


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "cases.py").write_text("# comment\ndef check():\n    assert True\n")
    (tmp_path / "checks" / "other.py").write_text("assert True\n")
    cfg = make_config(test_read_warn_after=1, sandbox_bash=False,
                      test_read_nudge="{runner}|{count}|{target}|{kind}|{coverage}")
    state = init_guardrail_state(cfg)
    return [tmp_path, cfg, state]


def call(rig, name, arguments, *, output="ok", exit_code=0, timed_out=False):
    root, cfg, state = rig
    metadata = {}
    with task_file_scope(str(root), cfg), patch("scripts.llm_solver.harness.tools._run_in_sandbox",
               return_value=(output, exit_code, timed_out)):
        result = dispatch(name, arguments, cwd=str(root), cfg=cfg, execution_metadata=metadata)
        decision = advise(state, cfg, tc_name=name, result=result, tc_args=arguments,
                          gate_blocked=False, execution_metadata=metadata, cwd=str(root))
        observe_test_file_read(state, cfg, tc_name=name, result=result, tc_args=arguments,
                               gate_blocked=False, execution_metadata=metadata)
    return decision, metadata


def run(rig, target="checks/cases.py"):
    return call(rig, "bash", {"cmd": f"pytest -q {target}"})[0]


@pytest.mark.parametrize("cmd", [
    "ls checks", "printf 'tests/test_missing.py'", "echo pytest checks/cases.py",
    "false && pytest checks/cases.py", "cd checks && pytest cases.py",
    "pytest --collect-only checks/cases.py", "pytest --version checks/cases.py",
    "pytest checks/cases.py | cat",
])
def test_mentions_nonexecution_and_compound_shell_do_not_count(rig, cmd):
    decision, _ = call(rig, "bash", {"cmd": cmd}, output="exit code: 0")
    assert decision.action == Action.PASS
    assert rig[2].test_target_observations == {}


def test_no_excerpt_partial_complete_and_stale_revision(rig):
    root, _, _ = rig
    assert "no matching recorded excerpt" in run(rig).text
    call(rig, "read", {"path": "checks/cases.py", "limit": 1})
    assert "partial recorded excerpt: lines 1 of 3" in run(rig).text
    call(rig, "read", {"path": "checks/cases.py", "offset": 1})
    assert run(rig).action == Action.PASS
    (root / "checks/cases.py").write_text("assert False\n")
    assert "different file revision" in run(rig).text


def test_directory_is_not_covered_by_one_descendant(rig):
    call(rig, "read", {"path": "checks/cases.py"})
    decision = run(rig, "checks")
    assert "directory spelling" in decision.text
    assert "do not establish suite coverage" in decision.text


@pytest.mark.parametrize("target", ["checks/missing.py", "package.TestCase", "checks/cases.py::check"])
def test_missing_and_nonfile_selectors_do_not_invent_files(rig, target):
    assert "missing or non-file" in run(rig, target).text


def test_counts_and_warning_suppression_are_per_target(rig):
    rig[1] = replace(rig[1], test_read_warn_after=2)
    assert run(rig).action == Action.PASS
    assert run(rig, "checks/other.py").action == Action.PASS
    assert "|2|checks/cases.py|" in run(rig).text
    assert run(rig).action == Action.PASS
    assert "|2|checks/other.py|" in run(rig, "checks/other.py").text


def test_clipped_excerpt_does_not_get_whole_file_credit(rig):
    root, cfg, state = rig
    rig[1] = replace(cfg, max_output_chars=100)
    (root / "checks/cases.py").write_text("different content\n" * 100)
    _, metadata = call(rig, "read", {"path": "checks/cases.py"})
    record = metadata["inspection_evidence"]
    assert record["delivery"] == "unknown"
    assert "inspection_body" not in metadata
    assert state.inspected_files[record["path"]]["intervals"] == []
    assert "extent is unknown" in run(rig).text


def test_trivial_cat_and_head_record_content_not_request_names(rig):
    from scripts.llm_solver.harness.sandbox import bwrap_preflight
    ready, reason = bwrap_preflight('/usr/bin/bwrap')
    if not ready:
        pytest.skip(reason)
    rig[1] = replace(rig[1], sandbox_bash=True)
    _, metadata = call(rig, "bash", {"cmd": "head -n 1 checks/cases.py"})
    assert metadata["inspection_evidence"]["line_count"] == 1
    assert "partial recorded excerpt" in run(rig).text
    call(rig, "bash", {"cmd": "cat checks/cases.py"})
    assert run(rig).action == Action.PASS


def test_unrecorded_shell_content_does_not_claim_it_was_not_read(rig):
    # Host shell path is deliberately mocked, without a content receipt.
    call(rig, "bash", {"cmd": "cat checks/cases.py"}, output="assert True")
    assert "other inspection is unknown" in run(rig).text


def test_remote_namespace_cannot_use_a_matching_host_file(rig, monkeypatch):
    call(rig, "read", {"path": "checks/cases.py"})
    monkeypatch.setenv("YUJ_CONTAINER", "uncontacted-container")
    decision = run(rig)
    assert "namespace or working directory is unbound" in decision.text


def test_denied_target_is_not_hashed_or_called_readable(rig):
    root, cfg, _ = rig
    rig[1] = replace(cfg, unreadable_paths=(str(root / "checks/cases.py"),))
    assert "not permitted for inspection" in run(rig).text


def test_symlink_loop_leaves_target_unknown(rig):
    (rig[0] / "loop").symlink_to("loop")
    assert "|unknown|" in run(rig, "loop").text


def test_structured_runner_uses_recorded_request(rig):
    rig[1] = replace(rig[1], tools_run_tests_enabled=True, analysis_task_format="pytest")
    decision, metadata = call(rig, "run_tests", {"path": "checks/cases.py"})
    assert metadata["runner_request"]["targets"] == ["checks/cases.py"]
    assert "checks/cases.py" in decision.text


@pytest.mark.parametrize("code,timed_out", [(127, False), (0, True)])
def test_unavailable_or_incomplete_execution_does_not_count(rig, code, timed_out):
    decision, _ = call(rig, "bash", {"cmd": "pytest checks/cases.py"},
                       exit_code=code, timed_out=timed_out)
    assert decision.action == Action.PASS


def test_disabled_guard_has_no_inspection_probe_or_state(rig):
    rig[1] = replace(rig[1], test_read_warn_after=0)
    call(rig, "read", {"path": "checks/cases.py"})
    assert run(rig).action == Action.PASS
    assert rig[2].inspected_files == {}
    assert rig[2].test_target_observations == {}


@pytest.mark.parametrize("command", [
    "pytest --collect-only checks/cases.py", "false && pytest checks/cases.py",
])
def test_structured_custom_command_does_not_certify_skipped_runner(rig, command):
    rig[1] = replace(rig[1], tools_run_tests_enabled=True, analysis_task_format="pytest")
    decision, _ = call(rig, "run_tests", {"_base_cmd_override": command})
    assert decision.action == Action.PASS


def test_downstream_replacement_cannot_keep_admitted_read_credit(rig):
    root, cfg, state = rig
    metadata = {}
    dispatch("read", {"path": "checks/cases.py"}, cwd=str(root), cfg=cfg,
             execution_metadata=metadata)
    observe_test_file_read(state, cfg, gate_blocked=False, result="replacement",
                           execution_metadata=metadata)
    assert "extent is unknown" in run(rig).text


def test_read_revision_is_from_bytes_returned_even_if_file_changes(rig):
    from pathlib import Path
    root, _, _ = rig
    original = Path.read_bytes
    target = root / "checks/cases.py"
    def changing_read(path):
        data = original(path)
        if path == target:
            path.write_text("changed after read\n")
        return data
    with patch.object(Path, "read_bytes", changing_read):
        _, metadata = call(rig, "read", {"path": "checks/cases.py"})
    assert metadata["inspection_evidence"]["line_count"] == 3
    assert "different file revision" in run(rig).text


@pytest.mark.parametrize("parallel", [False, True])
def test_session_delivers_partial_inspection_advice_and_traces_evidence(rig, parallel):
    import io
    import json
    from unittest.mock import MagicMock
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.types import TurnResult, ToolCall, Usage
    root, cfg, _ = rig
    cfg = replace(cfg, max_turns=3, loop_detect_enabled=False,
                  duplicate_guard_enabled=False, parallel_readonly_enabled=parallel)
    client = MagicMock()
    reads = [ToolCall(id="r", name="read", arguments={"path": "checks/cases.py", "limit": 1})]
    if parallel:
        reads.append(ToolCall(id="s", name="read", arguments={"path": "checks/other.py"}))
    requests = [reads, [ToolCall(id="v", name="bash", arguments={"cmd": "pytest checks/cases.py"})], []]
    client.chat.side_effect = [TurnResult(content=None, tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5)) for calls in requests]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    with patch("scripts.llm_solver.harness.tools._run_in_sandbox", return_value=("ok", 0, False)):
        Session(cfg, client, "system", "task", str(root), trace_file=trace).run()
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    rows = [row for row in events if row["event"] == "tool_call"]
    assert any(row.get("inspection_evidence", {}).get("line_count") == 1 for row in rows)
    # The next prepared model request must contain the truthful partial-read advice.
    assert "partial recorded excerpt: lines 1 of 3" in str(client.chat.call_args_list[-1])
