"""Task aliases must not admit a replacement host resource tree."""
import os
import shlex
import subprocess

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap
from scripts.llm_solver.harness.task_file_runtime import task_file_scope


@pytest.mark.parametrize('spelling', ['relative', 'host_alias', 'host_root', 'native'])
def test_task_alias_takes_precedence_over_external_resource_grant(bwrap, tmp_path, spelling):
    from tests.test_task_files import namespace_files
    from scripts.llm_solver.harness.task_path import activate_task_files
    from scripts.llm_solver.harness._tools._common import _resolve_read, _is_external_readonly_path

    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    (source / 'guide').write_text('NATIVE TASK GUIDE')
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'guide').write_text('HIDDEN HOST GUIDE')
    alias = tmp_path / 'alias'
    alias.symlink_to(host, target_is_directory=True)
    resource = tmp_path / 'operator-guide'
    resource.write_text('PERMITTED OPERATOR GUIDE')
    requested = {'relative': 'guide', 'host_alias': str(alias / 'guide'),
                 'host_root': str(host / 'guide'), 'native': str(files.root / 'guide')}[spelling]
    grants = (str(tmp_path),)
    with activate_task_files(files, host_root=alias):
        assert _resolve_read(str(alias), requested, readonly_roots=grants).read_text() == 'NATIVE TASK GUIDE'
        assert not _is_external_readonly_path(str(alias), requested, readonly_roots=grants)
        assert _resolve_read(str(alias), str(resource), readonly_roots=grants).read_text() == 'PERMITTED OPERATOR GUIDE'
        assert _is_external_readonly_path(str(alias), str(resource), readonly_roots=grants)


@pytest.mark.parametrize('retained', [False, True])
def test_readable_task_alias_does_not_admit_replacement(bwrap, tmp_path, retained):
    original = tmp_path / 'original'
    (original / 'resources').mkdir(parents=True)
    (original / 'resources/guide').write_text('SELECTED RESOURCE')
    replacement = tmp_path / 'replacement'
    (replacement / 'resources').mkdir(parents=True)
    (replacement / 'resources/guide').write_text('REPLACEMENT RESOURCE')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    environment = {'PATH': os.environ['PATH']}
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
                      skills_readable_dirs=(str(alias / 'resources'),))
    with task_file_scope(str(alias), cfg, environment=environment, persistent=False) as files:
        assert files.read_bytes('resources/guide') == b'SELECTED RESOURCE'
        if not retained:
            alias.unlink()
            alias.symlink_to(replacement, target_is_directory=True)
            _check_resource(files, alias, replacement)
    if retained:
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        _check_resource(files, alias, replacement)


def _check_resource(files, alias, replacement):
    assert files.read_bytes('resources/guide') == b'SELECTED RESOURCE'
    # The task's alias cannot grant access to the replacement host bytes.
    for path in (alias / 'resources/guide', replacement / 'resources/guide'):
        result = files.run('cat -- "$1"', [str(path)], None)
        assert b'REPLACEMENT RESOURCE' not in result.stdout


@pytest.mark.parametrize('consumer', ['foreground', 'background', 'lsp', 'cell'])
def test_live_launcher_readable_paths_keep_selected_task(bwrap, tmp_path, monkeypatch, consumer):
    original = tmp_path / 'original'
    (original / 'resources').mkdir(parents=True)
    (original / 'resources/guide').write_text('SELECTED RESOURCE')
    replacement = tmp_path / 'replacement'
    (replacement / 'resources').mkdir(parents=True)
    (replacement / 'resources/guide').write_text('REPLACEMENT RESOURCE')
    external = tmp_path / 'operator-resource'
    external.mkdir()
    (external / 'guide').write_text('OPERATOR RESOURCE')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    environment = {'PATH': os.environ['PATH']}
    readable = (str(alias / 'resources'), str(external))
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, skills_readable_dirs=readable)
    with task_file_scope(str(alias), cfg, environment=environment, persistent=False):
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        command = ('cat resources/guide ' + shlex.quote(str(external / 'guide'))
                   + '; cat ' + shlex.quote(str(replacement / 'resources/guide')))
        options = dict(cwd=str(alias), bwrap_bin=bwrap, effective_env=environment,
                       readable_paths=readable)
        process_cwd, process_env = None, None
        if consumer == 'foreground':
            from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
            result = _run_in_sandbox(command, sandbox=True, sandbox_required=True,
                                     timeout=cfg.bash_timeout, raw_result=True,
                                     use_persistent=False, **options)
        else:
            if consumer == 'background':
                from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
                argv = build_background_sandbox_argv(command, **options)
            elif consumer == 'lsp':
                from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
                argv = build_lsp_sandbox_argv(('bash', '--noprofile', '--norc', '-c', command), **options)
            else:
                import importlib
                cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
                monkeypatch.setattr(cell, '_cell_command', lambda: command)
                argv, process_cwd, process_env = cell._build_cell_process(
                    cwd=str(alias), cfg=cfg, unreadable_paths=(), readable_paths=readable,
                    effective_env=environment, allow_login_shell=False)
            result = subprocess.run(argv, cwd=process_cwd, env=process_env,
                                    capture_output=True, timeout=cfg.bash_timeout,
                                    pass_fds=getattr(argv, 'pass_fds', ()))
        assert result.stdout == b'SELECTED RESOURCEOPERATOR RESOURCE'
        assert result.returncode != 0  # The replacement host read is refused.


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
def test_container_mount_arguments_retain_task_resource_binding(bwrap, tmp_path, runtime):
    from scripts.llm_solver.harness.sandbox.container_backend import _build_container_argv

    original = tmp_path / 'original'
    (original / 'resources').mkdir(parents=True)
    replacement = tmp_path / 'replacement'
    (replacement / 'resources').mkdir(parents=True)
    external = tmp_path / 'operator-resource'
    external.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with task_file_scope(str(alias), cfg, environment={'PATH': os.environ['PATH']}):
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        argv = _build_container_argv('true', str(original), runtime=runtime,
                                     runtime_bin='/fixture/runtime', image='fixture:local',
                                     readable_paths=(str(alias / 'resources'), str(external)))
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == '--mount']
    assert any(f'source={original},' in value for value in mounts)
    assert any(f'source={external},' in value and ',readonly,' in value for value in mounts)
    assert not any(str(replacement) in value for value in mounts)


@pytest.mark.parametrize('consumer', ['background', 'lsp', 'persistent'])
def test_retained_manager_resource_paths_keep_selected_task(bwrap, tmp_path, consumer):
    original = tmp_path / 'original'
    (original / 'resources').mkdir(parents=True)
    (original / 'resources/guide').write_text('SELECTED RESOURCE')
    replacement = tmp_path / 'replacement'
    (replacement / 'resources').mkdir(parents=True)
    (replacement / 'resources/guide').write_text('REPLACEMENT RESOURCE')
    external = tmp_path / 'operator-resource'
    external.mkdir()
    (external / 'guide').write_text('OPERATOR RESOURCE')
    alias = tmp_path / 'alias'
    alias.symlink_to(original, target_is_directory=True)
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    command = ('cat resources/guide ' + shlex.quote(str(external / 'guide'))
               + '; cat ' + shlex.quote(str(replacement / 'resources/guide')))
    options = dict(cwd=str(alias), bwrap_bin=bwrap, effective_env={'PATH': os.environ['PATH']},
                   readable_paths=(str(alias / 'resources'), str(external)))
    with task_file_scope(str(alias), cfg, environment=options['effective_env']):
        if consumer == 'background':
            from scripts.llm_solver.harness.process_manager import ProcessManager
            manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                               poll_timeout_s=cfg.bash_timeout, **options)
        elif consumer == 'lsp':
            from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec
            spec = LspServerSpec('fixture', ('bash', '--noprofile', '--norc', '-c', command), ('.txt',))
            manager = LspManager.sandboxed(servers=(spec,), **options)
        else:
            from scripts.llm_solver.harness.sandbox import PersistentBashSession
            manager = PersistentBashSession(sandbox_required=True, **options)
    try:
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        if consumer == 'persistent':
            result = manager.run_binary(command, cwd=str(original), timeout=cfg.bash_timeout)
        else:
            argv = (manager.argv_builder(command) if consumer == 'background'
                    else manager.argv_builder(spec, manager.cwd))
            result = subprocess.run(argv, cwd=manager.cwd, capture_output=True,
                                    timeout=cfg.bash_timeout, pass_fds=getattr(argv, 'pass_fds', ()))
        assert result.stdout == b'SELECTED RESOURCEOPERATOR RESOURCE'
        assert result.returncode != 0
    finally:
        manager.close()
