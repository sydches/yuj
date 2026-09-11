"""Admitted read-only resources retain their source during bwrap launch."""
import os
import shlex
import subprocess
from contextlib import ExitStack

import pytest

from tests.test_task_files import bwrap
from scripts.llm_solver.harness.sandbox import _build_bwrap_argv
from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
from scripts.llm_solver.harness.task_environment import task_environment_scope


def resource_fixture(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    resource = tmp_path / 'resource'
    resource.mkdir()
    (resource / 'guide').write_text('SELECTED')
    environment = {'PATH': os.environ['PATH']}

    def replace_resource():
        resource.rename(tmp_path / 'selected-resource')
        resource.mkdir()
        (resource / 'guide').write_text('REPLACEMENT')

    return task, resource, environment, replace_resource


def test_readable_resource_replacement_after_argv_keeps_original_mount(bwrap, tmp_path):
    task, resource, environment, replace_resource = resource_fixture(tmp_path)
    argv = _build_bwrap_argv(
        'cat ' + shlex.quote(str(resource / 'guide')), str(task), bwrap_bin=bwrap,
        effective_env=environment, readable_paths=(str(resource),), sandbox_required=True)
    replace_resource()
    result = subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b'SELECTED'


def test_frozen_resource_refuses_replacement_before_argv(bwrap, tmp_path):
    task, resource, environment, replace_resource = resource_fixture(tmp_path)

    @task_environment_scope
    def session():
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        replace_resource()
        with pytest.raises(RuntimeError, match='runtime resource changed'):
            _build_bwrap_argv('true', str(task), bwrap_bin=bwrap, effective_env=environment,
                              readable_paths=(str(resource),), sandbox_required=True)

    session()


def test_resource_mount_stays_read_only_and_drops_source_handles(bwrap, tmp_path):
    task, resource, environment, replace_resource = resource_fixture(tmp_path)
    guide = shlex.quote(str(resource / 'guide'))
    command = (f'cat {guide}; if printf WRONG > {guide}; then exit 99; fi; '
               'for fd in /proc/self/fd/*; do readlink "$fd"; done; printf EXECUTED')
    argv = _build_bwrap_argv(command, str(task), bwrap_bin=bwrap, effective_env=environment,
                             readable_paths=(str(resource),), sandbox_required=True)
    replace_resource()
    result = subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, timeout=20)
    assert result.returncode == 0 and result.stdout.startswith(b'SELECTED')
    assert result.stdout.endswith(b'EXECUTED')
    assert os.fsencode(str(resource)) not in result.stdout
    assert b'selected-resource' not in result.stdout
    assert (tmp_path / 'selected-resource/guide').read_text() == 'SELECTED'
    assert (resource / 'guide').read_text() == 'REPLACEMENT'


def test_prepared_resource_handles_are_released_with_the_arguments(bwrap, tmp_path):
    import gc

    task, resource, environment, _ = resource_fixture(tmp_path)
    gc.collect()
    before = len(os.listdir('/proc/self/fd'))
    argv = _build_bwrap_argv('true', str(task), bwrap_bin=bwrap, effective_env=environment,
                             readable_paths=(str(resource),), sandbox_required=True)
    assert len(os.listdir('/proc/self/fd')) == before + len(argv.pass_fds)
    del argv
    gc.collect()
    assert len(os.listdir('/proc/self/fd')) == before


def test_persistent_restart_refuses_replacement_runtime_resource(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox import PersistentBashSession

    task, resource, environment, replace_resource = resource_fixture(tmp_path)
    runner = PersistentBashSession(cwd=str(task), bwrap_bin=bwrap, sandbox_required=True,
                                   effective_env=environment, readable_paths=(str(resource),))
    try:
        result = runner.run_binary('cat ' + shlex.quote(str(resource / 'guide')),
                                    cwd=str(task), timeout=20)
        assert result.returncode == 0 and result.stdout == b'SELECTED'
        runner.close()
        replace_resource()
        with pytest.raises(RuntimeError, match='runtime.*changed'):
            runner.start()
    finally:
        runner.close()


def test_persistent_resource_identity_ignores_descriptor_numbers(bwrap, tmp_path):
    from scripts.llm_solver.harness.sandbox import PersistentBashSession

    task, resource, environment, _ = resource_fixture(tmp_path)
    environment['VALUE'] = '--ro-bind-fd'
    options = dict(bwrap_bin=bwrap, sandbox_required=True, effective_env=environment,
                   readable_paths=(str(resource),), unreadable_paths=(), allow_login_shell=False)
    runner = PersistentBashSession(cwd=str(task), **options)
    pid = None
    try:
        for _ in range(2):
            result = runner.run_binary('printf %s "$VALUE"; cat ' + shlex.quote(str(resource / 'guide')),
                                        cwd=str(task), timeout=20, execution_options=options)
            assert result.returncode == 0 and result.stdout == b'--ro-bind-fdSELECTED'
            if pid is not None:
                assert runner._proc.pid == pid
            pid = runner._proc.pid
    finally:
        runner.close()


def test_standalone_runtime_file_is_pinned_before_launch(bwrap, tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    binaries = tmp_path / 'runtime/bin'
    binaries.mkdir(parents=True)
    executable = binaries / 'python'
    executable.write_text('#!/bin/sh\nprintf SELECTED\n')
    executable.chmod(0o755)
    environment = {'PATH': str(binaries) + ':/usr/bin:/bin'}
    argv = _build_bwrap_argv('python', str(task), bwrap_bin=bwrap,
                             effective_env=environment, sandbox_required=True)
    executable.rename(binaries / 'selected-python')
    executable.write_text('#!/bin/sh\nprintf REPLACEMENT\n')
    executable.chmod(0o755)
    result = subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, timeout=20)
    assert result.returncode == 0 and result.stdout == b'SELECTED', result.stderr


def test_runtime_record_keeps_process_handles_private(tmp_path):
    import json

    task, resource, environment, _ = resource_fixture(tmp_path)
    view = freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
    record = view.record()
    assert json.loads(json.dumps(record)) == record
    assert all('source_binding' not in mount for mount in record['mounts'])


@pytest.mark.parametrize('consumer', ['files', 'background', 'lsp', 'persistent'])
@pytest.mark.parametrize('replace_source', [False, True])
@pytest.mark.parametrize('create_before_freeze', [False, True])
def test_retained_executor_keeps_frozen_resource_binding(
        bwrap, tmp_path, consumer, replace_source, create_before_freeze):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import make_task_files
    from scripts.llm_solver.harness.process_manager import ProcessManager
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
    from scripts.llm_solver.harness.sandbox import PersistentBashSession

    task, resource, environment, replace_resource = resource_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    command = 'cat ' + shlex.quote(str(resource / 'guide'))

    def build():
        options = dict(cwd=str(task), bwrap_bin=bwrap, effective_env=environment,
                       readable_paths=(str(resource),))
        if consumer == 'files':
            executor = make_task_files(str(task), cfg, environment=environment,
                                        readable_paths=(str(resource),), persistent=False)
            return executor, lambda: executor.run(command, [], None)
        if consumer == 'background':
            executor = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                                poll_timeout_s=20, **options)
            return executor, lambda: executor.argv_builder(command)
        if consumer == 'lsp':
            spec = LspServerSpec('fixture', ('bash', '-c', command), ('.txt',))
            executor = LspManager.sandboxed(servers=(spec,), **options)
            return executor, lambda: executor.argv_builder(spec, executor.cwd)
        executor = PersistentBashSession(sandbox_required=True, **options)
        return executor, executor.start

    @task_environment_scope
    def construct():
        if create_before_freeze:
            result = build()
            freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
            return result
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        return build()

    executor, launch = construct()
    try:
        if replace_source:
            replace_resource()
            with pytest.raises(RuntimeError, match='runtime resource changed'):
                launch()
        else:
            result = launch()
            if consumer == 'persistent':
                result = executor.run_binary(command, cwd=str(task), timeout=20)
            elif consumer != 'files':
                result = subprocess.run(result, pass_fds=result.pass_fds, capture_output=True, timeout=20)
            assert result.returncode == 0 and result.stdout == b'SELECTED'
    finally:
        if consumer != 'files':
            executor.close()


@pytest.mark.parametrize('create_before_freeze', [False, True])
@pytest.mark.parametrize('persistent', [False, True])
def test_retained_file_executor_does_not_adopt_another_scopes_resources(
        bwrap, tmp_path, create_before_freeze, persistent):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import make_task_files

    task, resource, environment, _ = resource_fixture(tmp_path)
    other = tmp_path / 'other-resource'
    other.mkdir()
    (other / 'guide').write_text('OTHER SCOPE')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)

    @task_environment_scope
    def construct():
        if create_before_freeze:
            executor = make_task_files(str(task), cfg, environment=environment, persistent=persistent)
            freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
            return executor
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        return make_task_files(str(task), cfg, environment=environment, persistent=persistent)

    executor = construct()

    @task_environment_scope
    def later():
        freeze_filesystem_view(task, environment, readable_paths=(str(other),))
        with ExitStack() as cleanup:
            if persistent:
                from scripts.llm_solver.harness.sandbox import (
                    PersistentBashSession, get_persistent_runner, set_persistent_runner,
                )
                runner = PersistentBashSession(cwd=str(task), bwrap_bin=bwrap,
                                               effective_env=environment, sandbox_required=True)
                cleanup.callback(runner.close)
                cleanup.callback(set_persistent_runner, get_persistent_runner())
                set_persistent_runner(runner)
                runner.start()
                pid = runner._proc.pid
            allowed = executor.run('cat ' + shlex.quote(str(resource / 'guide')), [], None)
            assert allowed.returncode == 0 and allowed.stdout == b'SELECTED'
            withheld = executor.run('cat ' + shlex.quote(str(other / 'guide')), [], None)
            assert withheld.returncode != 0 and not withheld.stdout
            if persistent:
                assert runner._proc.pid == pid
                result = runner.run_binary('cat ' + shlex.quote(str(other / 'guide')),
                                           cwd=str(task), timeout=20)
                assert result.returncode == 0 and result.stdout == b'OTHER SCOPE'

    later()


@pytest.mark.parametrize('reader_before_freeze', [False, True])
def test_reader_and_persistent_shell_share_completed_startup_admission(
        bwrap, tmp_path, reader_before_freeze):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import make_task_files
    from scripts.llm_solver.harness.sandbox import (
        PersistentBashSession, get_persistent_runner, set_persistent_runner,
    )

    task, resource, environment, _ = resource_fixture(tmp_path)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)

    def make_reader():
        return make_task_files(str(task), cfg, environment=environment)

    def make_runner():
        return PersistentBashSession(cwd=str(task), bwrap_bin=bwrap,
                                     effective_env=environment, sandbox_required=True)

    @task_environment_scope
    def session():
        early = make_reader() if reader_before_freeze else make_runner()
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        files, runner = ((early, make_runner()) if reader_before_freeze
                         else (make_reader(), early))
        previous = get_persistent_runner()
        set_persistent_runner(runner)
        try:
            result = files.run('printf SHARED > /tmp/admission-state', [], None)
            assert result.returncode == 0
            pid = runner._proc.pid
            result = runner.run_binary('cat /tmp/admission-state', cwd=str(task), timeout=20)
            assert result.returncode == 0 and result.stdout == b'SHARED'
            assert runner._proc.pid == pid
        finally:
            set_persistent_runner(previous)
            runner.close()

    session()


def test_refreshed_reader_keeps_its_parents_frozen_resource_view(bwrap, tmp_path):
    from dataclasses import replace
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    from scripts.llm_solver.harness.task_path import activate_task_files

    task, resource, environment, _ = resource_fixture(tmp_path)
    (task / 'private').write_text('do not return')
    other = tmp_path / 'other-resource'
    other.mkdir()
    (other / 'guide').write_text('OTHER SCOPE')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)

    @task_environment_scope
    def construct():
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        with task_file_scope(str(task), cfg, environment=environment, persistent=False) as files:
            return files

    retained = construct()

    @task_environment_scope
    def later():
        freeze_filesystem_view(task, environment, readable_paths=(str(other),))
        with activate_task_files(retained, host_root=task):
            with task_file_scope(str(task), replace(cfg, unreadable_paths=(str(task / 'private'),)),
                                 environment=environment, persistent=False) as files:
                allowed = files.run('cat ' + shlex.quote(str(resource / 'guide')), [], None)
                assert allowed.returncode == 0 and allowed.stdout == b'SELECTED'
                withheld = files.run('cat ' + shlex.quote(str(other / 'guide')), [], None)
                assert withheld.returncode != 0 and not withheld.stdout
                with pytest.raises(OSError):
                    files.read_bytes('private')

    later()


def test_executor_without_completed_startup_cannot_borrow_later_admission(bwrap, tmp_path):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import make_task_files

    task, resource, environment, _ = resource_fixture(tmp_path)
    (task / 'file').write_text('TASK')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)

    @task_environment_scope
    def construct():
        executor = make_task_files(str(task), cfg, environment=environment, persistent=False)
        # Mechanical task reads remain possible while its own startup runs.
        assert executor.read_bytes('file') == b'TASK'
        return executor

    executor = construct()
    with pytest.raises(RuntimeError, match='was not admitted'):
        executor.read_bytes('file')

    @task_environment_scope
    def later():
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        with pytest.raises(RuntimeError, match='was not admitted'):
            executor.run('cat ' + shlex.quote(str(resource / 'guide')), [], None)

    later()


def test_admitted_view_cannot_be_replaced_by_another_discovery(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.sandbox import _filesystem as filesystem

    task, resource, environment, _ = resource_fixture(tmp_path)

    @task_environment_scope
    def session():
        view = freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        monkeypatch.setattr(filesystem, 'discover_filesystem_view',
                            lambda *_args, **_kwargs: pytest.fail('second discovery was executed'))
        with pytest.raises(RuntimeError, match='already admitted'):
            freeze_filesystem_view(task, environment)
        assert filesystem._ACTIVE.get() is view

    session()
