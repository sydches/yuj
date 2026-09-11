"""Force a host-name change after argv construction, before bwrap mounts it."""
import importlib
import os
import subprocess

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap
from scripts.llm_solver.harness.task_file_runtime import task_file_scope


def task_fixture(tmp_path):
    root = tmp_path / 'task'
    root.mkdir()
    (root / 'file').write_text('SELECTED')

    def replace():
        root.rename(tmp_path / 'selected')
        root.mkdir()
        (root / 'file').write_text('REPLACEMENT')

    return root, replace


def assert_selected_write(tmp_path):
    assert (tmp_path / 'selected/file').read_text() == 'BOUND'
    assert (tmp_path / 'task/file').read_text() == 'REPLACEMENT'


def test_standalone_bwrap_builder_binds_before_launch(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox import _build_bwrap_argv

    root, replace = task_fixture(tmp_path)
    argv = _build_bwrap_argv('printf BOUND > file', str(root), bwrap_bin=bwrap,
                             effective_env={'PATH': os.environ['PATH']}, sandbox_required=True)
    replace()
    result = subprocess.run(argv, pass_fds=getattr(argv, 'pass_fds', ()),
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert_selected_write(tmp_path)


def test_frozen_startup_view_refuses_replacement_before_file_scope(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
    from scripts.llm_solver.harness.task_environment import task_environment_scope, TaskEnvironmentUnavailable

    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}

    @task_environment_scope
    def session():
        freeze_filesystem_view(root, environment)
        replace()
        with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
            with task_file_scope(str(root), cfg, environment=environment, persistent=False):
                pytest.fail('startup task binding was silently replaced')

    session()


def test_early_task_reads_bind_the_root_before_startup_discovery(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.task_file_runtime import make_task_files
    from scripts.llm_solver.harness.sandbox import _filesystem as filesystem
    from scripts.llm_solver.harness.task_environment import task_environment_scope, TaskEnvironmentUnavailable

    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}

    @task_environment_scope
    def session():
        executor = make_task_files(str(root), cfg, environment=environment, persistent=False)
        assert executor.read_bytes('file') == b'SELECTED'
        replace()
        monkeypatch.setattr(filesystem, 'discover_filesystem_view',
                            lambda *_args, **_kwargs: pytest.fail('replacement task was inspected'))
        with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
            filesystem.freeze_filesystem_view(root, environment)

    session()


def test_file_scope_keeps_the_task_alias_captured_at_startup(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
    from scripts.llm_solver.harness.task_environment import task_environment_scope

    root, _ = task_fixture(tmp_path)
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    (replacement / 'file').write_text('REPLACEMENT')
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}

    @task_environment_scope
    def session():
        freeze_filesystem_view(alias, environment)
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        with task_file_scope(str(alias), cfg, environment=environment, persistent=False) as files:
            assert files.read_bytes('file') == b'SELECTED'
            files.write_bytes('file', b'BOUND')

    session()
    assert (root / 'file').read_text() == 'BOUND'
    assert (replacement / 'file').read_text() == 'REPLACEMENT'


@pytest.mark.parametrize('binary', [False, True])
def test_foreground_mount_uses_descriptor_after_root_name_changes(bwrap, tmp_path, monkeypatch, binary):
    runtime = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    execute = runtime._execute

    def launch(argv, **kwargs):
        replace()
        return execute(argv, **kwargs)

    with task_file_scope(str(root), cfg, environment=environment, persistent=False):
        monkeypatch.setattr(runtime, '_execute', launch)
        result = runtime._run_in_sandbox(
            'printf BOUND > file', cwd=str(root), timeout=cfg.bash_timeout,
            sandbox=True, sandbox_required=True, bwrap_bin=bwrap,
            effective_env=environment, raw_result=binary, use_persistent=False)
    assert (result.returncode if binary else result[1]) == 0
    assert_selected_write(tmp_path)


def test_retained_file_executor_mount_uses_its_original_descriptor(bwrap, tmp_path, monkeypatch):
    runtime = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with task_file_scope(str(root), cfg, environment={'PATH': os.environ['PATH']},
                         persistent=False) as files:
        assert files.read_bytes('file') == b'SELECTED'
    execute = runtime._execute

    def launch(argv, **kwargs):
        replace()
        return execute(argv, **kwargs)

    monkeypatch.setattr(runtime, '_execute', launch)
    result = files.run('printf BOUND > file', [], None)
    assert result.returncode == 0
    assert_selected_write(tmp_path)


@pytest.mark.parametrize('consumer', ['background', 'lsp', 'cell'])
def test_prepared_launcher_keeps_root_after_scope_ends(bwrap, tmp_path, monkeypatch, consumer):
    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    with task_file_scope(str(root), cfg, environment=environment, persistent=False):
        options = dict(cwd=str(root), bwrap_bin=bwrap, effective_env=environment)
        if consumer == 'background':
            from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
            argv = build_background_sandbox_argv('printf BOUND > file', **options)
        elif consumer == 'lsp':
            from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
            argv = build_lsp_sandbox_argv(('bash', '-c', 'printf BOUND > file'), **options)
        else:
            cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
            monkeypatch.setattr(cell, '_cell_command', lambda: 'printf BOUND > file')
            argv, _, _ = cell._build_cell_process(
                cwd=str(root), cfg=cfg, unreadable_paths=(), readable_paths=(),
                effective_env=environment, allow_login_shell=False)
    replace()
    result = subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True,
                            timeout=cfg.bash_timeout)
    assert result.returncode == 0, result.stderr
    assert_selected_write(tmp_path)


def test_bwrap_closes_the_inherited_task_descriptor_before_task_code(bwrap, tmp_path):
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with task_file_scope(str(tmp_path), cfg, environment={'PATH': os.environ['PATH']},
                         persistent=False) as files:
        # The mount source handle must not offer a second route around masks.
        result = files.run('for fd in /proc/self/fd/*; do readlink "$fd"; done; printf EXECUTED', [], None)
    assert result.returncode == 0 and b'EXECUTED' in result.stdout.splitlines()
    assert os.fsencode(str(tmp_path)) not in result.stdout.splitlines()


@pytest.mark.parametrize('consumer', ['background', 'lsp'])
def test_manager_passes_root_descriptor_to_its_process(bwrap, tmp_path, consumer):
    from scripts.llm_solver.harness.process_manager import ProcessManager
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec

    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    environment = {'PATH': os.environ['PATH']}
    processes = []

    def launch(argv, **kwargs):
        replace()
        process = subprocess.Popen(argv, **kwargs)
        processes.append(process)
        # This fixture tests launch transport only. The literal LSP command
        # exits without speaking the protocol, which the manager must report.
        if consumer == 'lsp':
            process.wait(timeout=cfg.bash_timeout)
        return process

    options = dict(cwd=str(root), bwrap_bin=bwrap, effective_env=environment,
                   popen_factory=launch)
    with task_file_scope(str(root), cfg, environment=environment, persistent=False):
        if consumer == 'background':
            manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                               poll_timeout_s=cfg.bash_timeout, **options)
        else:
            spec = LspServerSpec('fixture', ('bash', '-c', 'printf BOUND > file'), ('.txt',))
            manager = LspManager.sandboxed(servers=(spec,), **options)
    try:
        if consumer == 'background':
            manager.start('printf BOUND > file')
        else:
            assert manager._start(spec, root) is None
        assert len(processes) == 1
        assert processes[0].wait(timeout=cfg.bash_timeout) == 0
        assert_selected_write(tmp_path)
    finally:
        manager.close()


def test_persistent_start_mounts_the_bound_directory(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.sandbox import PersistentBashSession

    root, replace = task_fixture(tmp_path)
    runner = PersistentBashSession(cwd=str(root), bwrap_bin=bwrap, sandbox_required=True,
                                   effective_env={'PATH': os.environ['PATH']})
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        replace()
        return popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', launch)
    try:
        runner.start()
        # The pinned native root lets this check observe the launched mount
        # without running a later command whose host binding has now changed.
        mounted_file = f'/proc/self/fd/{runner._namespace.root}{root}/file'
        with open(mounted_file) as stream:
            assert stream.read() == 'SELECTED'
    finally:
        runner.close()


def test_cell_launch_inherits_the_task_descriptor(bwrap, tmp_path, monkeypatch):
    cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
    root, replace = task_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, tools_exec_cell_enabled=True)
    environment = {'PATH': os.environ['PATH']}
    popen = subprocess.Popen

    def launch(argv, **kwargs):
        replace()
        return popen(argv, **kwargs)

    # The shell consumes the cell request, then writes the marker. This checks
    # the production process launch without invoking a model or Python cell.
    monkeypatch.setattr(cell, '_cell_command', lambda: 'read -r request; printf BOUND > file')
    with task_file_scope(str(root), cfg, environment=environment, persistent=False):
        monkeypatch.setattr(subprocess, 'Popen', launch)
        cell.execute_cell('fixture', cwd=str(root), cfg=cfg, inner_dispatch=lambda *_: None,
                          unreadable_paths=(), readable_paths=(), effective_env=environment,
                          allow_login_shell=False)
    assert_selected_write(tmp_path)
