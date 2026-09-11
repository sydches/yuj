"""A constructed session retains the task view used by its startup."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.task_file_runtime import file_scoped_session
from scripts.llm_solver.harness.task_path import activate_task_files, resolve_task_path


@pytest.mark.parametrize('later_scope', ['none', 'different', 'retargeted_alias'])
def test_session_execution_retains_its_startup_file_view(
        bwrap, tmp_path, monkeypatch, later_scope):
    from scripts.llm_solver.harness._loop import run_step
    monkeypatch.setenv('YUJ_PERSISTENT_BASH', '0')
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'app.py').write_text('original native document')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested' / 'app.py').write_text('hidden host document')
    alias = tmp_path / 'alias'
    alias.symlink_to(source, target_is_directory=True)
    client = MagicMock()
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with activate_task_files(files, host_root=alias):
        session = Session(cfg, client, 'system', 'fixture', str(alias),
                          effective_env={'PATH': '/usr/bin:/bin'})
    other = tmp_path / 'other'
    other.mkdir()
    other_source, other_files = namespace_files(bwrap, other)
    (other_source / 'nested').mkdir()
    (other_source / 'nested' / 'app.py').write_text('other native document')
    if later_scope == 'retargeted_alias':
        alias.unlink()
        alias.symlink_to(other_source, target_is_directory=True)

    def read_during_execution(selected_session):
        for path in ('nested/app.py', str(alias / 'nested' / 'app.py'),
                     str(source / 'nested' / 'app.py'),
                     str(files.root / 'nested' / 'app.py')):
            assert resolve_task_path(selected_session.cwd, path).read_text() == 'original native document'
        return SimpleNamespace(total_prompt_tokens=0, total_completion_tokens=0,
                               finish_reason='fixture')

    monkeypatch.setattr(run_step, 'run_session_loop', read_during_execution)

    later = (activate_task_files(other_files, host_root=alias)
             if later_scope == 'different' else nullcontext())
    with later:
        file_scoped_session(read_during_execution)(session)
        assert session.run().finish_reason == 'fixture'
        if later_scope == 'different':
            assert resolve_task_path(alias, 'nested/app.py').read_text() == 'other native document'
    client.chat.assert_not_called()


@pytest.mark.parametrize('before_freeze', [False, True])
@pytest.mark.parametrize('refresh_mask', [False, True])
def test_session_retains_production_reader_resource_admission(
        bwrap, tmp_path, before_freeze, refresh_mask):
    from dataclasses import replace
    from tests.test_native_resource_mounts import resource_fixture
    from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
    from scripts.llm_solver.harness.task_environment import task_environment_scope
    from scripts.llm_solver.harness.task_path import active_task_files

    task, resource, environment, _ = resource_fixture(tmp_path)
    (task / 'secret').write_text('task bytes')
    other = tmp_path / 'other-resource'
    other.mkdir()
    (other / 'guide').write_text('OTHER SCOPE')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    client = MagicMock()

    @task_environment_scope
    def construct():
        if not before_freeze:
            freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        session = Session(cfg, client, 'system', 'fixture', str(task),
                          effective_env=environment, allow_login_shell=False)
        if before_freeze:
            freeze_filesystem_view(task, environment, readable_paths=(str(resource),))
        return session

    session = construct()
    if refresh_mask:
        session.cfg = replace(cfg, unreadable_paths=(str(task / 'secret'),))

    @file_scoped_session
    def inspect_resources(selected_session):
        files = active_task_files(selected_session.cwd)
        assert files.readonly_view(str(resource)).read_bytes('guide') == b'SELECTED'
        with pytest.raises(OSError):
            files.readonly_view(str(other)).read_bytes('guide')
        if refresh_mask:
            with pytest.raises(OSError):
                files.read_bytes('secret')
        else:
            assert files.read_bytes('secret') == b'task bytes'

    @task_environment_scope
    def later():
        freeze_filesystem_view(task, environment, readable_paths=(str(other),))
        inspect_resources(session)

    later()
    client.chat.assert_not_called()
