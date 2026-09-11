"""A task's host spelling must not silently select a replacement directory."""
import os
import subprocess

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap
from scripts.llm_solver.harness.task_file_runtime import task_file_scope
from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable


@pytest.mark.parametrize('consumer', ['files', 'foreground', 'background', 'lsp', 'cell'])
def test_replacement_root_is_refused_before_new_execution(bwrap, tmp_path, monkeypatch, consumer):
    from scripts.llm_solver.harness.sandbox import (
        PersistentBashSession, get_persistent_runner, set_persistent_runner,
    )

    root = tmp_path / 'task'
    root.mkdir()
    (root / 'file').write_text('SELECTED')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(root), cfg, environment=environment) as files:
        runner = PersistentBashSession(cwd=str(root), bwrap_bin=bwrap,
                                       sandbox_required=True, effective_env=environment)
        previous = get_persistent_runner()
        set_persistent_runner(runner)
        try:
            assert files.read_bytes('file') == b'SELECTED'
            original_pid = runner._proc.pid
            root.rename(tmp_path / 'selected')
            root.mkdir()
            (root / 'file').write_text('REPLACEMENT')
            # The existing shell still holds its original mount. A new launch
            # must refuse the changed binding before executing task code.
            options = dict(cwd=str(root), bwrap_bin=bwrap, effective_env=environment)
            with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
                if consumer == 'files':
                    files.read_bytes('file')
                elif consumer == 'foreground':
                    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
                    _run_in_sandbox('printf WRONG > file', timeout=cfg.bash_timeout,
                                    sandbox=True, sandbox_required=True, use_persistent=False,
                                    **options)
                else:
                    if consumer == 'background':
                        from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
                        argv = build_background_sandbox_argv('printf WRONG > file', **options)
                    elif consumer == 'lsp':
                        from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
                        argv = build_lsp_sandbox_argv(('bash', '-c', 'printf WRONG > file'), **options)
                    else:
                        import importlib
                        cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
                        monkeypatch.setattr(cell, '_cell_command', lambda: 'printf WRONG > file')
                        argv, _, _ = cell._build_cell_process(
                            cwd=str(root), cfg=cfg, unreadable_paths=(), readable_paths=(),
                            effective_env=environment, allow_login_shell=False)
                    subprocess.run(argv, check=True, timeout=cfg.bash_timeout, capture_output=True,
                                   pass_fds=getattr(argv, 'pass_fds', ()))
            assert runner._proc.pid == original_pid
            assert (root / 'file').read_text() == 'REPLACEMENT'
            assert (tmp_path / 'selected/file').read_text() == 'SELECTED'
        finally:
            set_persistent_runner(previous)
            runner.close()


def test_retained_file_executor_refuses_a_replacement_root(bwrap, tmp_path):
    root = tmp_path / 'task'
    root.mkdir()
    (root / 'file').write_text('SELECTED')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with task_file_scope(str(root), cfg, environment={'PATH': os.environ['PATH']},
                         persistent=False) as files:
        assert files.read_bytes('file') == b'SELECTED'
    root.rename(tmp_path / 'selected')
    root.mkdir()
    (root / 'file').write_text('REPLACEMENT')
    with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
        files.write_bytes('file', b'WRONG')
    assert (root / 'file').read_text() == 'REPLACEMENT'
    assert (tmp_path / 'selected/file').read_text() == 'SELECTED'


@pytest.mark.parametrize('consumer', ['background', 'lsp', 'persistent'])
def test_retained_manager_refuses_root_replacement(bwrap, tmp_path, consumer):
    from scripts.llm_solver.harness.process_manager import ProcessManager
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
    from scripts.llm_solver.harness.sandbox import PersistentBashSession

    root = tmp_path / 'task'
    root.mkdir()
    (root / 'file').write_text('SELECTED')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(root), cfg, environment=environment, persistent=False):
        options = dict(cwd=str(root), bwrap_bin=bwrap, effective_env=environment)
        if consumer == 'background':
            manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                               poll_timeout_s=cfg.bash_timeout, **options)
            launch = lambda: manager.argv_builder('printf WRONG > file')
        elif consumer == 'lsp':
            spec = LspServerSpec('fixture', ('bash', '-c', 'printf WRONG > file'), ('.txt',))
            manager = LspManager.sandboxed(servers=(spec,), **options)
            launch = lambda: manager.argv_builder(spec, manager.cwd)
        else:
            manager = PersistentBashSession(sandbox_required=True, **options)
            # Exercise lazy restart too, without a model or background task.
            manager.start()
            manager.close()
            launch = manager.start
    try:
        root.rename(tmp_path / 'selected')
        root.mkdir()
        (root / 'file').write_text('REPLACEMENT')
        with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
            launch()
        assert (root / 'file').read_text() == 'REPLACEMENT'
        assert (tmp_path / 'selected/file').read_text() == 'SELECTED'
    finally:
        manager.close()


def test_root_binding_ignores_ordinary_task_edits_and_releases_descriptor(tmp_path):
    import gc
    from scripts.llm_solver.harness.task_path import HostTaskRoot

    gc.collect()
    before = len(os.listdir('/proc/self/fd'))
    root = HostTaskRoot(str(tmp_path))
    assert len(os.listdir('/proc/self/fd')) == before + 1
    # Directory timestamps change on ordinary edits; they are not identity.
    (tmp_path / 'new-file').write_text('EDIT')
    root.verify()
    del root
    gc.collect()
    assert len(os.listdir('/proc/self/fd')) == before


def test_activation_uses_executor_root_if_alias_changed_since_capture(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_file_runtime import make_task_files
    from scripts.llm_solver.harness.task_path import activate_task_files, active_task_host_root
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox

    selected = tmp_path / 'selected'
    selected.mkdir()
    (selected / 'file').write_text('SELECTED')
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    (replacement / 'file').write_text('REPLACEMENT')
    alias = tmp_path / 'alias'
    alias.symlink_to(selected, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    files = make_task_files(str(alias), cfg, environment=environment, persistent=False)
    alias.unlink()
    alias.symlink_to(replacement, target_is_directory=True)
    with activate_task_files(files, host_root=str(alias)):
        assert active_task_host_root(str(alias)) == str(selected)
        assert files.read_bytes('file') == b'SELECTED'
        result = _run_in_sandbox('cat file', cwd=str(alias), bwrap_bin=bwrap,
                                 timeout=cfg.bash_timeout, sandbox=True, sandbox_required=True,
                                 effective_env=environment, raw_result=True, use_persistent=False)
        assert result.returncode == 0 and result.stdout == b'SELECTED'
