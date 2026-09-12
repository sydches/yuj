"""Local typed reads and discovery keep the initially checked task boundary."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.llm_solver.harness._tools.glob import glob_files
from scripts.llm_solver.harness._tools.read import read
from scripts.llm_solver.harness._tools.grep import grep_files, _sorted_matches
from scripts.llm_solver.harness._tool_filters import _strip_cwd_absolute


def test_read_refuses_final_file_swapped_to_outside_link(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'file').write_text('TASK')
    outside = tmp_path / 'outside'
    outside.write_text('FORBIDDEN')
    original = os.open
    swapped = False

    def opening(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == 'file' and kwargs.get('dir_fd') is not None and not swapped:
            swapped = True
            (task / 'file').rename(task / 'saved')
            (task / 'file').symlink_to(outside)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', opening)
    result = read('file', cwd=str(task))
    assert swapped
    assert result.startswith('ERROR:')
    assert 'FORBIDDEN' not in result


def test_glob_never_lists_entries_from_swapped_outside_directory(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    folder = task / 'folder'
    folder.mkdir(parents=True)
    (folder / 'visible.py').write_text('TASK')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'leaked.py').symlink_to(task / 'saved' / 'visible.py')
    inode = folder.stat().st_ino
    original = os.scandir
    swapped = False

    def scanning(path):
        nonlocal swapped
        if isinstance(path, int) and os.fstat(path).st_ino == inode and not swapped:
            swapped = True
            folder.rename(task / 'saved')
            folder.symlink_to(outside)
        return original(path)

    monkeypatch.setattr(os, 'scandir', scanning)
    result = glob_files('folder/*.py', cwd=str(task))
    assert swapped
    assert 'leaked.py' not in result


@pytest.mark.parametrize('pattern', ['*.py', '*/*.py', '**/*.py', '**/*', '**', 'folder/../*.py', '*/'])
def test_local_glob_keeps_pathlib_matching_and_contained_aliases(tmp_path, pattern):
    (tmp_path / 'folder').mkdir()
    for name in ('top.py', '.hidden.py', 'folder/code.py', 'folder/text.txt'):
        (tmp_path / name).write_text('body')
    (tmp_path / 'alias').symlink_to('folder')
    (tmp_path / 'link.py').symlink_to('folder/code.py')
    expected = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.glob(pattern) if path.is_file())
    result = glob_files(pattern, cwd=str(tmp_path))
    assert result == ('\n'.join(expected) if expected else 'No files found.')
    assert read('link.py', cwd=str(tmp_path)) == '1: body'


def test_glob_lists_regular_files_without_read_permission(tmp_path):
    source = tmp_path / 'file.py'
    source.write_text('body')
    source.chmod(0)
    try:
        assert glob_files('*.py', cwd=str(tmp_path)) == 'file.py'
    finally:
        source.chmod(0o600)


@pytest.mark.parametrize('use_rg', [False, True])
@pytest.mark.parametrize('scope,glob_filter', [('.', ''), ('.', '*.py'), ('code.py', ''),
                                             ('ignored.py', ''), ('.hidden.py', '')])
def test_local_grep_preserves_engine_selection_and_output(tmp_path, monkeypatch, use_rg, scope, glob_filter):
    rg = shutil.which('rg')
    if use_rg and not rg:
        pytest.skip('rg unavailable')
    (tmp_path / '.git').mkdir()
    (tmp_path / '.gitignore').write_text('ignored.py\n')
    for name in ('code.py', 'ignored.py', '.hidden.py', 'text.txt', 'literal\nname.py'):
        (tmp_path / name).write_text('needle /dev/fd/10\nother\n')
    command = [rg, '-n', '--no-heading'] if use_rg else ['grep', '-rn']
    if glob_filter:
        command += ['--glob' if use_rg else '--include', glob_filter]
    command += ['needle', str((tmp_path / scope).resolve())]
    expected = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    expected = _sorted_matches(_strip_cwd_absolute(expected, str(tmp_path)))
    if not use_rg:
        monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert grep_files('needle', scope, glob_filter, cwd=str(tmp_path)) == expected


def test_grep_refuses_final_file_swap_after_selection(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    task.mkdir()
    source = task / 'file.py'
    source.write_text('needle TASK\n')
    outside = tmp_path / 'outside'
    outside.write_text('needle FORBIDDEN\n')
    original = os.open
    swapped = False

    def opening(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == 'file.py' and kwargs.get('dir_fd') is not None and not swapped:
            swapped = True
            source.rename(task / 'saved')
            source.symlink_to(outside)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', opening)
    result = grep_files('needle', cwd=str(task))
    assert swapped
    assert result.startswith('ERROR:')
    assert 'FORBIDDEN' not in result


@pytest.mark.parametrize('use_rg', [False, True])
def test_grep_preserves_binary_file_notices(tmp_path, monkeypatch, use_rg):
    rg = shutil.which('rg')
    if use_rg and not rg:
        pytest.skip('rg unavailable')
    (tmp_path / 'binary').write_bytes(b'needle\x00bytes\n')
    command = [rg, '-n', '--no-heading'] if use_rg else ['grep', '-rn']
    result = subprocess.run([*command, 'needle', str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode in (0, 1)
    expected = _sorted_matches(_strip_cwd_absolute(result.stdout, str(tmp_path)))
    if not use_rg:
        monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert grep_files('needle', cwd=str(tmp_path)) == (expected or 'No matches found.')


def test_grep_reuses_checked_parent_once_per_file_batch(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools import _local_grep
    for number in range(70):
        (tmp_path / f'file{number:03}.py').write_text('needle TASK\n')
    original = _local_grep.checked_local_parent
    opened = []

    def parent(cwd, target):
        opened.append(target.parent)
        return original(cwd, target)

    monkeypatch.setattr(_local_grep, 'checked_local_parent', parent)
    result = grep_files('needle', cwd=str(tmp_path))
    assert len(result.splitlines()) == 70
    assert opened == [tmp_path, tmp_path]


def test_grep_retains_checked_parent_across_directory_swap(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    folder = task / 'folder'
    folder.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    for name in ('a.py', 'b.py'):
        (folder / name).write_text('needle TASK\n')
        (outside / name).write_text('needle FORBIDDEN\n')
    original = os.open
    swapped = False

    def opening(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original(path, flags, *args, **kwargs)
        if path in ('a.py', 'b.py') and kwargs.get('dir_fd') is not None and not swapped:
            swapped = True
            folder.rename(task / 'saved')
            folder.symlink_to(outside)
        return descriptor

    monkeypatch.setattr(os, 'open', opening)
    result = grep_files('needle', 'folder', cwd=str(task))
    assert swapped
    assert result.count('TASK') == 2
    assert 'FORBIDDEN' not in result
