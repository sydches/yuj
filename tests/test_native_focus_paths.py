"""Focus location labels use the selected task view, not host symlinks."""
from types import SimpleNamespace

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness._loop.focus_dedup import _focus_signature, _path_within_cwd


@pytest.mark.parametrize('tool', ['read', 'bash'])
@pytest.mark.parametrize('spelling', ['native', 'host'])
def test_native_focus_does_not_claim_task_path_is_outside(bwrap, tmp_path, tool, spelling):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'source.txt').write_text('NATIVE')
    host = tmp_path / 'host'
    host.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('HOST')
    (host / 'source.txt').symlink_to(outside)
    path = str((files.root if spelling == 'native' else host) / 'source.txt')
    args = {'path': path} if tool == 'read' else {'cmd': 'cat ' + path}
    with activate_task_files(files, host_root=host):
        key, display = _focus_signature(SimpleNamespace(name=tool, arguments=args), '', str(host))
    assert key.startswith('file:')
    assert display == path


def test_native_focus_marks_a_known_outside_name(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    with activate_task_files(files, host_root=source):
        key, _ = _focus_signature(SimpleNamespace(name='read', arguments={'path': '/outside/file'}),
                                  '', str(source))
    assert key.startswith('outside:')


def test_unavailable_native_resolution_is_not_outside_evidence(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.task_files import TaskFileError

    source, files = namespace_files(bwrap, tmp_path)
    def unavailable(path):
        raise TaskFileError('native view unavailable')
    monkeypatch.setattr(files, 'resolve', unavailable)
    with activate_task_files(files, host_root=source):
        assert _path_within_cwd(str(source / 'file'), str(source)) is None
        key, _ = _focus_signature(SimpleNamespace(name='read', arguments={'path': str(source / 'file')}),
                                  '', str(source))
        assert not key.startswith('outside:')


def test_native_search_root_is_not_reported_outside(bwrap, tmp_path):
    from scripts.llm_solver.harness._loop.focus_dedup import _encode_focus_target

    _, files = namespace_files(bwrap, tmp_path)
    host = tmp_path / 'host'
    host.mkdir()
    with activate_task_files(files, host_root=host):
        key, display = _encode_focus_target(
            str(files.root) + '::*.txt', f'*.txt under {files.root}',
            root_path=str(files.root), cwd=str(host))
    assert key.startswith('bash:')
    assert str(files.root) in display


def test_focus_resolution_preserves_budget_exhaustion(bwrap, tmp_path, monkeypatch):
    from scripts.llm_solver.harness.time_budget import BudgetExhausted

    source, files = namespace_files(bwrap, tmp_path)
    def exhausted(path):
        raise BudgetExhausted('fixture allowance exhausted')
    monkeypatch.setattr(files, 'resolve', exhausted)
    with activate_task_files(files, host_root=source):
        with pytest.raises(BudgetExhausted):
            _focus_signature(SimpleNamespace(name='read', arguments={'path': str(source / 'file')}),
                             '', str(source))
