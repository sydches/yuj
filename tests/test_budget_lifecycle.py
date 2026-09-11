"""Dispatch budgets preserve preparation scope and actual launch outcomes."""
import importlib
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import time_budget as budgets, tools, file_changes, runtime_discovery
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields


def config(**changes):
    return make_config(tools_run_tests_enabled=True, analysis_task_format='pytest',
                       sandbox_bash=False, **changes)


@pytest.mark.parametrize('preparation', [0.25, 2.0])
def test_call_scope_starts_before_inventory_and_covers_postprocessing(tmp_path, monkeypatch, preparation):
    clock, deadlines, launches = [0.0], [], []
    monkeypatch.setattr(budgets.time, 'monotonic', lambda: clock[0])

    def inventory(*args):
        deadlines.append(budgets.execution_deadline())
        budgets.remaining_before(budgets.execution_deadline())
        clock[0] += preparation
        return {}, []

    def launch(*args, **kwargs):
        launches.append(kwargs['timeout'])
        return 'completed', 0, False

    monkeypatch.setattr(file_changes, '_inventory', inventory)
    monkeypatch.setattr(tools, '_run_in_sandbox', launch)
    facts = {}
    tools.dispatch('run_tests', {}, cwd=str(tmp_path), cfg=config(tools_run_tests_timeout=1),
                   execution_metadata=facts)
    assert deadlines == [1, 1]
    if preparation < 1:
        assert launches == [0.75]
        assert facts['executed'] and facts['exit_status'] == 0
    else:
        assert not launches
        assert not facts['executed']
        assert facts['verification_status'] == 'budget_exhausted'
    assert budgets.execution_deadline() is None


@pytest.mark.parametrize('spent', [0.25, 2.0])
def test_discovery_exhaustion_is_distinct_from_unresolved_selection(tmp_path, monkeypatch, spent):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(file_changes, '_inventory', lambda *args: ({}, []))

    def discover(*args, **kwargs):
        clock[0] = spent
        return {'runner_selection': {'status': 'unresolved', 'candidates': []}, 'probes': []}

    monkeypatch.setattr(runtime_discovery, 'discover_runtime', discover)
    monkeypatch.setattr(tools, '_run_in_sandbox', lambda *args, **kwargs: pytest.fail('must not launch'))
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format='auto',
                      sandbox_bash=False, tools_run_tests_timeout=1)
    facts = {}
    tools.dispatch('run_tests', {}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts['executed'] is False
    assert facts['verification_status'] == ('budget_exhausted' if spent > 1 else 'selection_unresolved')


@pytest.mark.parametrize('stage', ['lock', 'start'])
def test_persistent_prelaunch_exhaustion_reaches_dispatch_and_trace(tmp_path, monkeypatch, stage):
    from scripts.llm_solver.harness.sandbox import PersistentBashSession
    from scripts.llm_solver.harness.sandbox import _filesystem
    backend = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    runner = PersistentBashSession.__new__(PersistentBashSession)
    runner.cwd, runner._filesystem_view = str(tmp_path), None
    runner._proc = runner._namespace = None
    runner._lock = threading.Lock()
    runner._configure = lambda *args, **kwargs: None

    def start():
        if stage == 'lock':
            pytest.fail('lock refusal must not start the shell')
        raise subprocess.TimeoutExpired('fixture shell startup', 0.03)

    runner.start = start
    if stage == 'lock':
        runner._lock.acquire()
    monkeypatch.setattr(backend, 'container_mode', lambda: None)
    monkeypatch.setattr(backend, 'get_persistent_runner', lambda: runner)
    monkeypatch.setattr(file_changes, '_inventory', lambda *args: ({}, []))
    monkeypatch.setattr(_filesystem, 'capture_frozen_filesystem_view', lambda *args: None)
    monkeypatch.setattr(backend.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('must not launch'))
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format='pytest',
                      sandbox_bash=True, sandbox_required=True, bwrap_bin=sys.executable,
                      tools_run_tests_timeout=0.03)
    facts = {}
    try:
        result = tools.dispatch('run_tests', {}, cwd=str(tmp_path), cfg=cfg,
            effective_env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)}, execution_metadata=facts)
    finally:
        if stage == 'lock':
            runner._lock.release()
    assert facts['executed'] is False and facts['timed_out'] is False
    assert facts['exit_status'] is None and facts['verification_status'] == 'budget_exhausted'
    fields = build_tool_call_trace_fields(
        SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1),
        tool_name='run_tests', args_summary='', result=result, turn=1,
        gate_blocked=False, execution_metadata=facts)
    assert fields['verification_status'] == 'budget_exhausted'
    assert fields['execution_budget']['status'] == 'exhausted'


@pytest.mark.parametrize('sandbox', [False, True])
def test_launched_timeout_survives_supplemental_report_exhaustion(tmp_path, sandbox):
    if sandbox:
        from scripts.llm_solver.harness.sandbox import bwrap_preflight
        ok, reason = bwrap_preflight('/usr/bin/bwrap')
        if not ok:
            pytest.skip(reason)
    (tmp_path / 'widget.py').write_text('value = 1\n')
    (tmp_path / 'test_widget.py').write_text(
        'from pathlib import Path\nimport time\n'
        'def test_wait():\n    Path("runner-started").touch()\n    time.sleep(10)\n')
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format='pytest',
        sandbox_bash=sandbox, sandbox_required=sandbox, done_require_pretest_parity=True,
        sandbox_env_set={'PATH': str(Path(sys.executable).parent) + ':/usr/bin:/bin',
                         'HOME': str(tmp_path)})
    facts = {}
    # Allow native setup and collection before testing the launched timeout.
    with budgets.run_time_budget(5):
        tools.dispatch('run_tests', {'_component_source': 'widget.py',
            '_base_cmd_override': shlex.join([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'])},
            cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert (tmp_path / 'runner-started').exists()
    assert facts['executed'] and facts['timed_out']
    assert facts['verification_status'] == 'timed_out'
    assert facts['native_test_report']['status'] == 'unavailable'
    if sandbox:
        component = facts['runner_request']['component_selection']
        assert component['status'] == 'unavailable'
        assert component['cleanup_error']
        assert facts['native_test_report']['cleanup_error']


def test_inherited_allocation_status_matches_its_remaining_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, 'monotonic', lambda: clock[0])
    with budgets.command_time_budget(2) as parent:
        for now, status in [(0.5, 'allocated'), (2, 'exhausted')]:
            clock[0] = now
            with budgets.command_time_budget(0) as child:
                assert child.deadline == parent.deadline
                assert child.record['status'] == status
                assert child.record['effective_seconds'] == max(0, 2 - now)
        assert budgets.execution_deadline() == parent.deadline
    assert budgets.execution_deadline() is None
