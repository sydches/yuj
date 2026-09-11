"""Binary file transport preserves the configured execution boundary."""
import os
import subprocess

import pytest

from tests.test_task_files import bwrap
from tests._config_helpers import make_config
from scripts.llm_solver.harness.task_file_runtime import make_task_files
from scripts.llm_solver.harness.time_budget import run_time_budget


def test_binary_transport_does_not_normalize_or_merge_streams(tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    data = b'\x00\xff\r\n0x12345678\n'
    result = _run_in_sandbox(
        'cat; printf diagnostic >&2', cwd=str(tmp_path), timeout=None,
        sandbox=False, bwrap_bin='', raw_result=True, input_bytes=data,
    )
    assert result.returncode == 0
    assert result.stdout == data
    assert result.stderr == b'diagnostic'


def test_discovery_and_file_operations_share_the_declared_run_deadline(tmp_path):
    cfg = make_config(sandbox_bash=False)
    files = make_task_files(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']})
    files.write_bytes('literal', b'\x00\xff\r\n')
    assert files.read_bytes('literal') == b'\x00\xff\r\n'
    from scripts.llm_solver.harness.time_budget import BudgetExhausted
    with run_time_budget(0.001):
        import time
        time.sleep(0.005)
        with pytest.raises(BudgetExhausted):
            files.read_bytes('literal')


def test_binary_timeout_stops_the_local_execution(tmp_path):
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    with run_time_budget(0.05):
        with pytest.raises(subprocess.TimeoutExpired):
            _run_in_sandbox(
                'sleep 60', cwd=str(tmp_path), timeout=0.05,
                sandbox=False, bwrap_bin='', raw_result=True,
            )


def test_missing_selected_sandbox_never_falls_back_to_host_files(tmp_path):
    cfg = make_config(sandbox_bash=True, bwrap_bin=str(tmp_path / 'missing'))
    files = make_task_files(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']})
    (tmp_path / 'host').write_text('must not read')
    with pytest.raises(RuntimeError, match='Refusing to substitute another backend or run unsandboxed'):
        files.read_bytes('host')


def test_bound_files_refuse_a_later_container_selection(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    cfg = make_config(sandbox_bash=True)
    files = make_task_files(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']})
    monkeypatch.setenv('YUJ_CONTAINER', 'different-task')
    with pytest.raises(TaskEnvironmentUnavailable, match='changed after file binding'):
        files.read_bytes('file')


def test_nested_dispatch_cannot_replace_the_session_container_selection(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    cfg = make_config(sandbox_bash=True)
    with task_file_scope(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']}):
        monkeypatch.setenv('YUJ_CONTAINER', 'different-task')
        with pytest.raises(TaskEnvironmentUnavailable, match='changed within the file scope'):
            with task_file_scope(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']}):
                pytest.fail('the selected view was replaced')


@pytest.mark.parametrize('operation', ['read', 'write'])
def test_nested_scope_cannot_disable_the_selected_sandbox(bwrap, tmp_path, operation):
    from dataclasses import replace
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_path import active_task_files, resolve_task_path
    from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable

    target = tmp_path / 'file'
    target.write_text('ORIGINAL')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, unreadable_paths=(str(target),))
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(tmp_path), cfg, environment=environment, persistent=False) as files:
        # The selected namespace masks this file; the host still has its bytes.
        with pytest.raises(OSError):
            files.read_bytes('file')
        with pytest.raises(TaskEnvironmentUnavailable, match='changed within the file scope'):
            with task_file_scope(str(tmp_path), replace(cfg, sandbox_bash=False),
                                 environment=environment, persistent=False):
                path = resolve_task_path(str(tmp_path), 'file')
                if operation == 'write':
                    path.write_text('HOST WRITE')
                else:
                    path.read_text()
        assert active_task_files(str(tmp_path)) is files
        with pytest.raises(OSError):
            files.read_bytes('file')
    assert target.read_text() == 'ORIGINAL'


def test_separate_scope_may_select_local_execution(tmp_path):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_path import resolve_task_path

    cfg = make_config(sandbox_bash=False)
    with task_file_scope(str(tmp_path), cfg) as files:
        assert files is None
        resolve_task_path(str(tmp_path), 'file').write_text('LOCAL')
    assert (tmp_path / 'file').read_text() == 'LOCAL'


@pytest.mark.parametrize('change', ['environment', 'login', 'bwrap', 'backend'])
def test_nested_scope_refuses_changed_execution_selection(bwrap, tmp_path, change):
    from dataclasses import replace
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable
    from scripts.llm_solver.harness.task_path import active_task_files

    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH'], 'VIEW': 'SELECTED'}
    with task_file_scope(str(tmp_path), cfg, environment=environment,
                         allow_login_shell=False, persistent=False) as files:
        assert files.run('printf %s "$VIEW"', [], None).stdout == b'SELECTED'
        changed_cfg, changed_env, login = cfg, dict(environment), False
        if change == 'environment':
            changed_env['VIEW'] = 'REPLACEMENT'
        elif change == 'login':
            login = True
        elif change == 'bwrap':
            changed_cfg = replace(cfg, bwrap_bin=str(tmp_path / 'other-runtime'))
        else:
            changed_cfg = replace(cfg, sandbox_backend='container',
                                  sandbox_container_runtime='missing-fixture-runtime')
        with pytest.raises(TaskEnvironmentUnavailable, match='changed within the file scope'):
            with task_file_scope(str(tmp_path), changed_cfg, environment=changed_env,
                                 allow_login_shell=login, persistent=False):
                pass
        assert active_task_files(str(tmp_path)) is files
        assert files.run('printf %s "$VIEW"', [], None).stdout == b'SELECTED'


def test_nested_scope_reuses_an_equivalent_environment(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope

    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH'], 'VIEW': 'SELECTED'}
    with task_file_scope(str(tmp_path), cfg, environment=environment) as outer:
        with task_file_scope(str(tmp_path), cfg, environment=dict(reversed(environment.items()))) as inner:
            assert inner is outer


def test_nested_scope_can_update_masks_without_changing_execution(bwrap, tmp_path):
    from dataclasses import replace
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_path import active_task_files

    target = tmp_path / 'file'
    target.write_text('VISIBLE')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH'], 'VIEW': 'SELECTED'}
    with task_file_scope(str(tmp_path), cfg, environment=environment, persistent=False) as outer:
        assert outer.read_bytes('file') == b'VISIBLE'
        with task_file_scope(str(tmp_path), replace(cfg, unreadable_paths=(str(target),)),
                             environment=environment, persistent=False) as inner:
            with pytest.raises(OSError):
                inner.read_bytes('file')
            assert inner.run('printf %s "$VIEW"', [], None).stdout == b'SELECTED'
        assert active_task_files(str(tmp_path)) is outer
        assert outer.read_bytes('file') == b'VISIBLE'
    assert target.read_text() == 'VISIBLE'


def test_separate_scope_can_select_a_new_environment(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope

    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    for value in ['FIRST', 'SECOND']:
        environment = {'PATH': os.environ['PATH'], 'VIEW': value}
        with task_file_scope(str(tmp_path), cfg, environment=environment, persistent=False) as files:
            assert files.run('printf %s "$VIEW"', [], None).stdout == value.encode()


def test_changed_environment_requires_a_fresh_native_read(bwrap, tmp_path):
    import json
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.stale_guard import StaleFileGuard

    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    (tmp_path / 'file').write_text('UNCHANGED')
    guard = StaleFileGuard(cwd=tmp_path, mode='block')
    environment = {'PATH': os.environ['PATH'], 'VIEW': 'PRIVATE_FIRST_VALUE'}
    with task_file_scope(str(tmp_path), cfg, environment=environment, persistent=False) as files:
        guard.observe('file', source='read')
        first_binding = dict(files.binding)
    environment['VIEW'] = 'PRIVATE_SECOND_VALUE'
    with task_file_scope(str(tmp_path), cfg, environment=environment, persistent=False) as files:
        assert files.binding != first_binding
        assert 'PRIVATE_FIRST_VALUE' not in json.dumps(first_binding)
        assert 'PRIVATE_SECOND_VALUE' not in json.dumps(files.binding)
        decision = guard.check_edit('file')
        assert decision.blocked and decision.reason == 'view_changed'
        guard.observe('file', source='read')
        assert guard.check_edit('file').allowed


@pytest.mark.parametrize('consumer', [
    'files', 'foreground', 'background', 'lsp', 'cell', 'background_manager', 'lsp_manager',
])
def test_execution_keeps_selected_task_after_host_alias_retarget(bwrap, tmp_path, monkeypatch, consumer):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope

    original = tmp_path / 'original'
    original.mkdir()
    (original / 'file').write_text('SELECTED TASK')
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    (replacement / 'file').write_text('REPLACEMENT TASK')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(alias), cfg, environment=environment, persistent=False) as files:
        assert files.read_bytes('file') == b'SELECTED TASK'
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        if consumer == 'files':
            output = files.read_bytes('file')
            files.write_bytes('file', b'EDITED SELECTED TASK')
        elif consumer == 'foreground':
            from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
            result = _run_in_sandbox('cat file', cwd=str(alias), timeout=cfg.bash_timeout,
                                     sandbox=True, sandbox_required=True, bwrap_bin=bwrap,
                                     effective_env=environment, raw_result=True, use_persistent=False)
            assert result.returncode == 0
            output = result.stdout
        else:
            options = dict(cwd=str(alias), bwrap_bin=bwrap, effective_env=environment)
            process_cwd, process_env = None, None
            if consumer in {'background', 'background_manager'}:
                from scripts.llm_solver.harness.process_manager import (
                    build_background_sandbox_argv, ProcessManager,
                )
                if consumer == 'background':
                    argv = build_background_sandbox_argv('cat file', **options)
                else:
                    manager = ProcessManager.sandboxed(
                        run_dir=tmp_path / 'records', max_procs=1,
                        poll_timeout_s=cfg.bash_timeout, **options)
                    argv = manager.argv_builder('cat file')
                    process_cwd = str(manager.cwd)
                    manager.close()
            elif consumer in {'lsp', 'lsp_manager'}:
                from scripts.llm_solver.harness.lsp_support import (
                    build_lsp_sandbox_argv, LspManager, LspServerSpec,
                )
                if consumer == 'lsp':
                    argv = build_lsp_sandbox_argv(('cat', 'file'), **options)
                else:
                    spec = LspServerSpec('fixture', ('cat', 'file'), ('.txt',))
                    manager = LspManager.sandboxed(servers=(spec,), **options)
                    argv = manager.argv_builder(spec, manager.cwd)
                    process_cwd = str(manager.cwd)
                    manager.close()
            else:
                import importlib
                cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
                # Check the cell transport using a literal read, not a model
                # request or the interactive cell protocol.
                monkeypatch.setattr(cell, '_cell_command', lambda: 'cat file')
                argv, process_cwd, process_env = cell._build_cell_process(
                    cwd=str(alias), cfg=cfg, unreadable_paths=(), readable_paths=(),
                    effective_env=environment, allow_login_shell=False)
            result = subprocess.run(argv, cwd=process_cwd, env=process_env,
                                    capture_output=True, timeout=cfg.bash_timeout,
                                    pass_fds=getattr(argv, 'pass_fds', ()))
            assert result.returncode == 0, result.stderr
            output = result.stdout
        assert output == b'SELECTED TASK'
    assert (replacement / 'file').read_text() == 'REPLACEMENT TASK'
    assert (original / 'file').read_text() == (
        'EDITED SELECTED TASK' if consumer == 'files' else 'SELECTED TASK')


def test_persistent_execution_retains_task_and_process_after_alias_retarget(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.sandbox import (
        PersistentBashSession, get_persistent_runner, set_persistent_runner,
    )
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox

    original = tmp_path / 'original'
    original.mkdir()
    (original / 'file').write_text('SELECTED TASK')
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    (replacement / 'file').write_text('REPLACEMENT TASK')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(alias), cfg, environment=environment) as files:
        runner = PersistentBashSession(cwd=str(alias), bwrap_bin=bwrap,
                                       sandbox_required=True, effective_env=environment)
        previous = get_persistent_runner()
        set_persistent_runner(runner)
        try:
            assert files.read_bytes('file') == b'SELECTED TASK'
            pid = runner._proc.pid
            state = files.run('printf RETAINED > /tmp/alias-state', [], None)
            assert state.returncode == 0
            alias.unlink()
            alias.symlink_to(replacement, target_is_directory=True)
            assert files.read_bytes('file') == b'SELECTED TASK'
            result = _run_in_sandbox('cat file /tmp/alias-state', cwd=str(alias), timeout=cfg.bash_timeout,
                                     sandbox=True, sandbox_required=True, bwrap_bin=bwrap,
                                     effective_env=environment, raw_result=True)
            assert result.returncode == 0 and result.stdout == b'SELECTED TASKRETAINED'
            assert runner._proc.pid == pid
        finally:
            set_persistent_runner(previous)
            runner.close()
    assert (replacement / 'file').read_text() == 'REPLACEMENT TASK'


@pytest.mark.parametrize('pattern', ['private', 'optional:private', 'priv*', 'priv*/'])
@pytest.mark.parametrize('persistent', [False, True])
def test_task_masks_retain_captured_alias_after_retarget(bwrap, tmp_path, pattern, persistent):
    from contextlib import ExitStack
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
    from scripts.llm_solver.harness.sandbox import (
        PersistentBashSession, get_persistent_runner, set_persistent_runner,
    )

    original = tmp_path / 'original'
    (original / 'private').mkdir(parents=True)
    (original / 'private/secret').write_text('SELECTED SECRET')
    (original / 'private_file').write_text('VISIBLE FILE')
    replacement = tmp_path / 'replacement'
    (replacement / 'private').mkdir(parents=True)
    (replacement / 'private/secret').write_text('REPLACEMENT SECRET')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    optional = 'optional:' if pattern.startswith('optional:') else ''
    mask = optional + str(alias) + '/' + pattern.removeprefix('optional:')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, unreadable_paths=(mask,))
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(alias), cfg, environment=environment, persistent=persistent) as files, ExitStack() as cleanup:
        if persistent:
            runner = PersistentBashSession(cwd=str(alias), bwrap_bin=bwrap,
                                           sandbox_required=True, effective_env=environment)
            cleanup.callback(runner.close)
            cleanup.callback(set_persistent_runner, get_persistent_runner())
            set_persistent_runner(runner)
        with pytest.raises(OSError):
            files.read_bytes('private/secret')
        pid = runner._proc.pid if persistent else None
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        with pytest.raises(OSError):
            files.read_bytes('private/secret')
        result = _run_in_sandbox('cat private/secret', cwd=str(alias), timeout=cfg.bash_timeout,
                                 sandbox=True, sandbox_required=True, bwrap_bin=bwrap,
                                 effective_env=environment, unreadable_paths=(mask,),
                                 raw_result=True, use_persistent=persistent)
        assert result.returncode != 0
        assert b'SELECTED SECRET' not in result.stdout
        assert b'REPLACEMENT SECRET' not in result.stdout
        if pattern.startswith('priv*'):
            (original / 'private_later').mkdir()
            (original / 'private_later/secret').write_text('LATER SECRET')
            with pytest.raises(OSError):
                files.read_bytes('private_later/secret')
        if pattern == 'priv*/':
            assert files.read_bytes('private_file') == b'VISIBLE FILE'
        if persistent:
            assert runner._proc.pid == pid
    with pytest.raises(OSError):
        files.read_bytes('private/secret')
    if pattern == 'priv*/':
        assert files.read_bytes('private_file') == b'VISIBLE FILE'
    assert (original / 'private/secret').read_text() == 'SELECTED SECRET'
    assert (replacement / 'private/secret').read_text() == 'REPLACEMENT SECRET'


@pytest.mark.parametrize('consumer', ['background', 'lsp', 'persistent'])
def test_retained_manager_masks_keep_the_captured_task_alias(bwrap, tmp_path, consumer):
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope

    original = tmp_path / 'original'
    (original / 'private').mkdir(parents=True)
    (original / 'private/secret').write_text('SELECTED SECRET')
    replacement = tmp_path / 'replacement'
    (replacement / 'private').mkdir(parents=True)
    (replacement / 'private/secret').write_text('REPLACEMENT SECRET')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    options = dict(cwd=str(alias), bwrap_bin=bwrap, effective_env=environment,
                   unreadable_paths=(str(alias / 'private'),))
    with task_file_scope(str(alias), cfg, environment=environment):
        if consumer == 'background':
            from scripts.llm_solver.harness.process_manager import ProcessManager
            manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                               poll_timeout_s=cfg.bash_timeout, **options)
        elif consumer == 'lsp':
            from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
            spec = LspServerSpec('fixture', ('cat', 'private/secret'), ('.txt',))
            manager = LspManager.sandboxed(servers=(spec,), **options)
        else:
            from scripts.llm_solver.harness.sandbox import PersistentBashSession
            manager = PersistentBashSession(sandbox_required=True, **options)
    try:
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        if consumer == 'persistent':
            result = manager.run_binary('cat private/secret', cwd=str(original), timeout=cfg.bash_timeout)
        else:
            argv = (manager.argv_builder('cat private/secret') if consumer == 'background'
                    else manager.argv_builder(spec, manager.cwd))
            result = subprocess.run(argv, cwd=manager.cwd, capture_output=True,
                                    timeout=cfg.bash_timeout, pass_fds=getattr(argv, 'pass_fds', ()))
        assert result.returncode != 0
        assert b'SELECTED SECRET' not in result.stdout
        assert b'REPLACEMENT SECRET' not in result.stdout
    finally:
        manager.close()
