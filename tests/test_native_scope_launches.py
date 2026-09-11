"""Command consumers must use the selected file reader's admitted mounts."""
import importlib
import shlex
import subprocess

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap
from tests.test_native_resource_mounts import resource_fixture
from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
from scripts.llm_solver.harness.task_environment import task_environment_scope
from scripts.llm_solver.harness.task_file_runtime import make_task_files
from scripts.llm_solver.harness.task_path import activate_task_files


@pytest.mark.parametrize('consumer', ['foreground', 'background', 'lsp', 'cell'])
@pytest.mark.parametrize('before_freeze', [False, True])
def test_command_launch_uses_retained_reader_admission(
        bwrap, tmp_path, monkeypatch, consumer, before_freeze):
    task, resource, environment, _ = resource_fixture(tmp_path)
    other = tmp_path / 'other-resource'
    other.mkdir()
    (other / 'guide').write_text('OTHER SCOPE')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)

    def reader():
        return make_task_files(str(task), cfg, environment=environment, persistent=False)

    @task_environment_scope
    def construct():
        files = reader() if before_freeze else None
        freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        return files if files is not None else reader()

    files = construct()

    def execute(command):
        options = dict(cwd=str(task), bwrap_bin=bwrap, effective_env=environment,
                       sandbox_required=True)
        if consumer == 'foreground':
            from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
            return _run_in_sandbox(command, timeout=20, sandbox=True,
                                   raw_result=True, use_persistent=False, **options)
        if consumer == 'background':
            from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
            argv = build_background_sandbox_argv(command, **options)
        elif consumer == 'lsp':
            from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
            argv = build_lsp_sandbox_argv(('bash', '-c', command), **options)
        else:
            cell = importlib.import_module('scripts.llm_solver.harness._tools.exec_cell')
            monkeypatch.setattr(cell, '_cell_command', lambda: command)
            argv, _, _ = cell._build_cell_process(
                cwd=str(task), cfg=cfg, unreadable_paths=(), readable_paths=(),
                effective_env=environment, allow_login_shell=False)
        return subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, timeout=20)

    @task_environment_scope
    def later():
        freeze_filesystem_view(task, environment, readable_paths=(str(other),))
        with activate_task_files(files, host_root=task):
            allowed = execute('cat ' + shlex.quote(str(resource / 'guide')))
            assert allowed.returncode == 0 and allowed.stdout == b'SELECTED'
            withheld = execute('cat ' + shlex.quote(str(other / 'guide')))
            assert withheld.returncode != 0 and not withheld.stdout

    later()
