"""Requested syntax must survive display limits without inventing execution."""
import json
import shlex
import sys
from types import SimpleNamespace

import pytest
from _config_helpers import make_config
from scripts.llm_solver.harness import tools
from scripts.llm_solver.harness._loop.session_io import _summarize_args
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event
from scripts.llm_solver.harness.runner_invocations import describe_runner_invocations


def run_and_project(root, command, *, tool='bash', cfg=None):
    cfg = cfg or make_config(sandbox_bash=False, tools_run_tests_enabled=True,
                            analysis_task_format='pytest', args_summary_chars=80)
    args = {'cmd': command} if tool == 'bash' else {'_base_cmd_override': command}
    metadata = {}
    result = tools.dispatch(tool, args, cwd=str(root), cfg=cfg, execution_metadata=metadata)
    summary = _summarize_args(args, cfg.args_summary_chars)
    session = SimpleNamespace(cfg=cfg, cwd=str(root), _sink_counter=0, _session_number=1)
    fields = build_tool_call_trace_fields(session, tool_name=tool, args_summary=summary,
        result=result, turn=1, gate_blocked=False, execution_metadata=metadata)
    event = {'tool_name': tool, 'args_summary': summary, **fields}
    return result, event, project_tool_event(event)


@pytest.mark.parametrize('incomplete', ['compound', 'timeout', 'error'])
def test_full_requests_survive_clipped_and_incomplete_results(tmp_path, monkeypatch, incomplete):
    target = 'package.' + 'case_' * 40
    command = f'{shlex.quote(sys.executable)} -m unittest {target}'
    if incomplete == 'compound':
        command += ' && go test ./other'
    response = ('', 0, False) if incomplete == 'compound' else (
        ('partial', None, True) if incomplete == 'timeout' else ('ERROR: unavailable', None, False))
    monkeypatch.setattr(tools, '_run_in_sandbox', lambda *a, **k: response)
    _, event, slot = run_and_project(tmp_path, command)
    assert target not in event['args_summary']
    requests = json.loads(slot['runner_requests'])
    assert requests[0]['family'] == 'unittest' and requests[0]['targets'] == [target]
    if incomplete == 'compound':
        assert requests[1]['family'] == 'go' and requests[1]['targets'] == ['./other']
    else:
        assert len(requests) == 1
    assert slot['test_execution_action'] == 'false'


@pytest.mark.parametrize('filename,options', [
    ('test_plain.py', '-q'), ('test_--help.py', '-q'), ('test_plain.py', '-q --color=no'),
    ('test_plain.py', '-q -k ' + shlex.quote('test_one or ' * 20 + 'test_one'))])
def test_completed_request_overrides_display_heuristics(tmp_path, filename, options):
    (tmp_path / filename).write_text('def test_one():\n    assert True\n')
    command = f'{shlex.quote(sys.executable)} -m pytest {options} ' + ' '.join([filename] * 12)
    _, event, slot = run_and_project(tmp_path, command)
    assert event['verification_status'] == 'passed'
    assert slot['runner_execution_source'] == 'recorded_status'
    assert slot['test_execution_action'] == 'true' and slot['test_exit_status'] == 'pass'
    assert project_tool_event({**event, 'gate_blocked': True})['test_execution_action'] == 'false'
    assert project_tool_event({**event, 'error_class': 'security_block'})['test_execution_action'] == 'false'


def test_unidentified_helper_has_no_formal_execution_credit(tmp_path):
    (tmp_path / 'runtests.py').write_text('print("helper ran")\n')
    result, _, slot = run_and_project(tmp_path, f'{shlex.quote(sys.executable)} runtests.py')
    assert 'helper ran' in result
    assert slot['runner_family'] == '' and slot['test_execution_action'] == 'false'


@pytest.mark.parametrize('tool', ['bash', 'run_tests'])
def test_heredoc_data_has_no_runner_request_or_execution(tmp_path, tool):
    command = "cat <<'TEXT'\npytest tests/not_executed.py\nTEXT"
    result, _, slot = run_and_project(tmp_path, command, tool=tool)
    assert 'pytest tests/not_executed.py' in result
    assert slot['runner_family'] == '' and slot['test_execution_action'] == 'false'
    assert not any(r['family'] for r in json.loads(slot['runner_requests']))
    requests = describe_runner_invocations('go test ./before; ' + command)
    assert [(r['family'], r['targets']) for r in requests] == [('go', ['./before'])]


def test_short_circuit_override_retains_request_without_execution(tmp_path):
    runner = tmp_path / 'pytest'
    runner.write_text('#!/bin/sh\ntouch runner-started\n')
    runner.chmod(0o700)
    _, event, slot = run_and_project(tmp_path, f'true || {shlex.quote(str(runner))}', tool='run_tests')
    assert not (tmp_path / 'runner-started').exists()
    assert slot['runner_family'] == 'pytest'
    assert slot['test_execution_action'] == 'false'
    assert event['verification_status'] == 'shell_unresolved'


def test_literal_override_keeps_attributed_test_completion(tmp_path):
    (tmp_path / 'test_one.py').write_text('def test_one():\n    assert True\n')
    command = f'{shlex.quote(sys.executable)} -m pytest -q test_one.py'
    _, event, slot = run_and_project(tmp_path, command, tool='run_tests')
    assert event['verification_status'] == 'passed'
    assert slot['test_execution_action'] == 'true' and slot['test_exit_status'] == 'pass'
