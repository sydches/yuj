"""Retained process and language-server launchers own their environment."""
import os
import subprocess
import importlib

import pytest

from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
from scripts.llm_solver.harness.process_manager import ProcessManager
from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable
from tests.test_task_files import bwrap


@pytest.mark.parametrize('consumer', ['background', 'lsp'])
@pytest.mark.parametrize('launched_before', [False, True])
def test_manager_launches_do_not_follow_caller_environment_changes(bwrap, tmp_path, consumer, launched_before):
    task = tmp_path / 'task'
    task.mkdir()
    environment = {'PATH': os.environ['PATH'], 'SELECTED_VALUE': 'original'}
    options = dict(cwd=str(task), bwrap_bin=bwrap, sandbox_required=True,
                   effective_env=environment)
    command = 'printf %s "$SELECTED_VALUE"'
    if consumer == 'background':
        manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                           poll_timeout_s=20, **options)
        prepare = lambda: manager.argv_builder(command)
    else:
        spec = LspServerSpec('fixture', ('bash', '-c', command), ('.txt',))
        manager = LspManager.sandboxed(servers=(spec,), **options)
        prepare = lambda: manager.argv_builder(spec, task)

    def launch():
        # Execute the real launcher arguments; the LSP fixture is a literal
        # command, not a language-server protocol or model session.
        argv = prepare()
        result = subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return result.stdout

    try:
        if launched_before:
            assert launch() == b'original'
        environment['SELECTED_VALUE'] = 'replacement'
        assert launch() == b'original'
    finally:
        manager.close()


@pytest.mark.parametrize('consumer', ['background', 'lsp'])
@pytest.mark.parametrize('initial_mode', [None, 'ambient'])
def test_retained_manager_refuses_changed_execution_mode(
        bwrap, tmp_path, monkeypatch, consumer, initial_mode):
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    # Only prepare transport arguments; no container or namespace is launched.
    monkeypatch.setattr(execution, '_ambient_network_prefix', lambda: ())
    if initial_mode is None:
        monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    else:
        monkeypatch.setenv('YUJ_CONTAINER', initial_mode)
    options = dict(cwd=tmp_path, bwrap_bin=bwrap, effective_env={'PATH': os.environ['PATH']})
    if consumer == 'background':
        manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                           poll_timeout_s=1, **options)
        prepare = lambda: manager.argv_builder('true')
    else:
        spec = LspServerSpec('fixture', ('true',), ('.txt',))
        manager = LspManager.sandboxed(servers=(spec,), **options)
        prepare = lambda: manager.argv_builder(spec, tmp_path)
    try:
        prepare()
        if initial_mode is None:
            monkeypatch.setenv('YUJ_CONTAINER', 'ambient')
        else:
            monkeypatch.delenv('YUJ_CONTAINER', raising=False)
        with pytest.raises(TaskEnvironmentUnavailable, match='execution selection changed'):
            prepare()
    finally:
        manager.close()
