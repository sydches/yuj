"""Metadata and directory reads use the checked native directory."""
import shlex

import pytest

from scripts.llm_solver.harness.task_files import TaskFileError
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('operation', ['metadata', 'lstat', 'list', 'readlink', 'symlink', 'kind'])
@pytest.mark.parametrize('change_at', ['resolution', 'entered_directory'])
def test_metadata_keeps_the_checked_directory(bwrap, tmp_path, operation, change_at):
    source, files = namespace_files(bwrap, tmp_path)
    folder = source / 'folder'
    folder.mkdir()
    (folder / 'file').write_bytes(b'SELECTED')
    (folder / 'link').symlink_to('selected-target')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'file').mkdir()
    (outside / 'link').write_bytes(b'OUTSIDE')
    (outside / 'foreign').touch()
    root = str(files.root)
    native_folder = root + '/folder'

    def observe():
        if operation == 'metadata':
            value = files.metadata('folder/file')
            return value.inode, value.size
        if operation == 'lstat':
            value = files.metadata('folder/link', follow_symlinks=False)
            return value.inode, value.size
        if operation == 'list':
            return sorted(path.name for path in files.iterdir('folder'))
        if operation == 'readlink':
            return files.readlink('folder/link')
        if operation == 'symlink':
            return files.is_symlink('folder/link')
        return files.kind('folder/file')

    expected = observe()
    swap = (f'mv -- {shlex.quote(native_folder)} {shlex.quote(root + "/saved")}; '
            f'ln -s -- {shlex.quote(str(outside))} {shlex.quote(native_folder)}')
    wrapper = source / 'controlled-observer'
    if change_at == 'resolution':
        observer = files._utility('realpath')
        observed_path = native_folder + '/file' if operation in ('metadata', 'kind') else native_folder
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(observer)} "$@"; code=$?; printf .; exit "$code") || exit\n'
            'value=${value%.}\n'
            f'if [[ "${{@: -1}}" == {shlex.quote(observed_path)} && ! -L {shlex.quote(native_folder)} ]]; then {swap}; fi\n'
            'printf %s "$value"\n')
        files._utilities['realpath'] = root + '/controlled-observer'
    else:
        observer = files._utility('readlink')
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(observer)} -- "${{@: -1}}"; code=$?; printf .; exit "$code") || exit\n'
            "value=${value%.}; value=${value%$'\\n'}\n"
            f'if [[ "${{@: -1}}" == /proc/self/cwd ]]; then {swap}; fi\n'
            "printf '%s\\0' \"$value\"\n")
        files._utilities['readlink'] = root + '/controlled-observer'
    wrapper.chmod(0o755)
    if change_at == 'resolution':
        with pytest.raises(PermissionError, match='escapes task root'):
            observe()
    else:
        assert observe() == expected
        assert folder.is_symlink(), 'the controlled directory change must have run'


def test_stat_refuses_a_final_symlink_installed_at_stat(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'file').write_bytes(b'SELECTED')
    outside = tmp_path / 'outside'
    outside.write_bytes(b'OUTSIDE')
    observer = files._utility('stat')
    wrapper = source / 'controlled-stat'
    wrapper.write_text(
        '#!/bin/bash\n'
        'rm -- ./file\n'
        f'ln -s -- {shlex.quote(str(outside))} ./file\n'
        f'exec {shlex.quote(observer)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['stat'] = str(files.root / 'controlled-stat')
    with pytest.raises(TaskFileError, match='symlink changed'):
        files.metadata('file')
    assert (source / 'file').is_symlink()


def test_metadata_needs_search_permission_not_file_read_permission(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    folder = source / 'folder'
    folder.mkdir(mode=0o300)
    entry = folder / 'file'
    entry.write_bytes(b'SELECTED')
    entry.chmod(0)
    try:
        assert files.metadata('.').is_dir
        assert files.metadata('folder/file').size == len(b'SELECTED')
        assert files.kind('folder/file') == 'file'
    finally:
        entry.chmod(0o600)
        folder.chmod(0o700)


def test_rooted_readonly_view_can_inspect_a_top_level_directory(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    view = files.readonly_view('/')
    first_directory = files.root.parents[-2]
    assert view.metadata('/').is_dir
    assert view.metadata(str(first_directory)).is_dir
    assert view.kind(str(first_directory)) == 'directory'


def test_directory_metadata_does_not_require_traversing_its_children(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import TaskPath

    source, files = namespace_files(bwrap, tmp_path)
    sealed = source / 'sealed'
    sealed.mkdir()
    (sealed / 'hidden.txt').write_bytes(b'not readable')
    visible = source / 'visible'
    visible.mkdir()
    (visible / 'file.txt').write_bytes(b'visible')
    sealed.chmod(0)
    try:
        assert files.metadata('sealed').is_dir
        assert files.kind('sealed') == 'directory'
        with pytest.raises(PermissionError):
            files.iterdir('sealed')
        root = TaskPath(files, files.root)
        assert [str(path.relative_to(root)) for path in root.glob('**/*.txt')] == ['visible/file.txt']
    finally:
        sealed.chmod(0o700)
