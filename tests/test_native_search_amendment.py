"""Public search compatibility in a synthetic selected task namespace."""
import shutil

import pytest
from _config_helpers import make_config
from test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_path import TaskPath, activate_task_files
from scripts.llm_solver.harness.tools import dispatch


def outputs(files, source, tool, arguments):
    local = dispatch(tool, arguments, cwd=str(source), cfg=make_config(sandbox_bash=False))
    with activate_task_files(files, host_root=source):
        native = dispatch(tool, arguments, cwd=str(source), cfg=make_config(sandbox_bash=True))
    return str(local), str(native)


@pytest.mark.parametrize('pattern,path', [('../*.txt', 'nested'), ('*/', '.')])
def test_public_glob_preserves_parent_and_directory_patterns(bwrap, tmp_path, pattern, path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'nested').mkdir()
    (source / 'local.txt').write_text('ordinary task file\n')
    local, native = outputs(files, source, 'glob', {'pattern': pattern, 'path': path})
    assert native == local
    if pattern == '*/':
        assert list(TaskPath(files, files.root).glob(pattern)) == [
            TaskPath(files, files.root / 'nested')]


@pytest.mark.parametrize('pattern', ['[^a].txt', '[!a].txt', r'back\*.txt', '[[]*.txt'])
def test_native_glob_preserves_literal_pattern_characters(bwrap, tmp_path, pattern):
    source, files = namespace_files(bwrap, tmp_path)
    for name in ('a.txt', 'b.txt', '^.txt', r'back\slash.txt', '[literal].txt'):
        (source / name).write_text('')
    local, native = outputs(files, source, 'glob', {'pattern': pattern})
    assert native == local


@pytest.mark.parametrize('path', ['.', 'nested'])
@pytest.mark.parametrize('glob_filter', ['', '!*.py', '*.py', '*.txt'])
def test_public_grep_uses_native_rg_selection(bwrap, tmp_path, glob_filter, path):
    if shutil.which('rg') is None:
        pytest.skip('selection contrast requires ripgrep')
    source, files = namespace_files(bwrap, tmp_path)
    (source / '.git').mkdir()
    (source / '.gitignore').write_text('generated.txt\n')
    folder = source / path
    folder.mkdir(exist_ok=True)
    for name in ('source.py', 'generated.txt', '.hidden.txt', 'notes.txt'):
        (folder / name).write_text('needle\n')
    local, native = outputs(files, source, 'grep', {
        'pattern': 'needle', 'glob': glob_filter, 'path': path})
    assert native == local


def test_native_glob_never_enumerates_outside_parent(bwrap, tmp_path, monkeypatch):
    source, files = namespace_files(bwrap, tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'back').symlink_to(source, target_is_directory=True)
    enumerated = []
    original = files.iterdir
    def listing(path='.'):
        enumerated.append(str(path))
        return original(path)
    monkeypatch.setattr(files, 'iterdir', listing)
    assert list(TaskPath(files, files.root).glob('../outside/back/*')) == []
    assert enumerated == []
