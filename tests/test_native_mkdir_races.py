"""Directory creation checks every parent before creating its children."""
import shlex

import pytest

from scripts.llm_solver.harness.task_files import TaskFileError
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('parents', [False, True])
@pytest.mark.parametrize('change_at', ['resolution', 'entered_parent'])
def test_mkdir_keeps_the_checked_parent(bwrap, tmp_path, parents, change_at):
    outside = tmp_path / 'outside'
    outside.mkdir()
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    (source / 'folder').mkdir()
    root = str(files.root)
    folder = root + '/folder'
    suffix = 'branch/leaf' if parents else 'leaf'
    path = 'folder/' + suffix
    swap = (f'mv -- {shlex.quote(folder)} {shlex.quote(root + "/saved")}; '
            f'ln -s -- {shlex.quote(str(outside))} {shlex.quote(folder)}')
    wrapper = source / 'controlled-observer'
    if change_at == 'resolution':
        observer = files._utility('realpath')
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(observer)} "$@"; code=$?; printf .; exit "$code") || exit\n'
            'value=${value%.}\n'
            f'if [[ "${{@: -1}}" == {shlex.quote(root + "/" + path)} ]]; then {swap}; fi\n'
            'printf %s "$value"\n')
        files._utilities['realpath'] = root + '/controlled-observer'
    else:
        observer = files._utility('readlink')
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(observer)} -- "${{@: -1}}"; code=$?; printf .; exit "$code") || exit\n'
            "value=${value%.}; value=${value%$'\\n'}\n"
            f'if [[ "$value" == {shlex.quote(folder)} && ! -L {shlex.quote(folder)} ]]; then {swap}; fi\n'
            "printf '%s\\0' \"$value\"\n")
        files._utilities['readlink'] = root + '/controlled-observer'
    wrapper.chmod(0o755)
    if change_at == 'resolution':
        with pytest.raises(PermissionError, match='escapes task root'):
            files.mkdir(path, parents=parents)
    else:
        files.mkdir(path, parents=parents)
        assert (source / 'folder').is_symlink(), 'the controlled parent change must have run'
        assert (source / 'saved' / suffix).is_dir()
    assert list(outside.iterdir()) == []


def test_recursive_mkdir_refuses_a_replaced_intermediate_directory(bwrap, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    (source / 'folder').mkdir()
    branch = str(files.root / 'folder/branch')
    saved = str(files.root / 'folder/saved')
    native_mkdir = files._utility('mkdir')
    wrapper = source / 'controlled-mkdir'
    wrapper.write_text(
        '#!/bin/bash\n'
        f'{shlex.quote(native_mkdir)} "$@" || exit\n'
        f'if [[ -d {shlex.quote(branch)} && ! -L {shlex.quote(branch)} ]]; then\n'
        f'  mv -- {shlex.quote(branch)} {shlex.quote(saved)}\n'
        f'  ln -s -- {shlex.quote(str(outside))} {shlex.quote(branch)}\n'
        'fi\n')
    wrapper.chmod(0o755)
    files._utilities['mkdir'] = str(files.root / 'controlled-mkdir')
    with pytest.raises(PermissionError, match='escapes task root'):
        files.mkdir('folder/branch/leaf', parents=True)
    assert (source / 'folder/branch').is_symlink()
    assert list(outside.iterdir()) == []


def test_recursive_mkdir_does_not_accept_a_late_final_symlink(bwrap, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    native_mkdir = files._utility('mkdir')
    wrapper = source / 'controlled-mkdir'
    wrapper.write_text(
        '#!/bin/bash\n'
        f'ln -s -- {shlex.quote(str(outside))} ./leaf\n'
        f'exec {shlex.quote(native_mkdir)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['mkdir'] = str(files.root / 'controlled-mkdir')
    with pytest.raises(TaskFileError):
        files.mkdir('leaf', parents=True)
    assert (source / 'leaf').is_symlink()
    assert list(outside.iterdir()) == []


def test_mkdir_keeps_contained_aliases_and_literal_names(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'selected').mkdir()
    (source / 'alias').symlink_to('selected')
    name = "literal ' $(touch escaped)\nfolder\n"
    files.mkdir('alias/' + name + '/leaf', parents=True)
    assert (source / 'selected' / name / 'leaf').is_dir()
    assert (source / 'alias').is_symlink()
    assert not (source / 'escaped').exists()
    assert not (source / 'selected/escaped').exists()


@pytest.mark.parametrize('mask', ['0022', '0077', '0777'])
def test_recursive_mkdir_preserves_native_umask_behavior(bwrap, tmp_path, mask):
    source, files = namespace_files(bwrap, tmp_path)
    original_run = files.run
    files.run = lambda script, args, data: original_run(f'umask {mask}\n' + script, args, data)
    native_mkdir = files._utility('mkdir')
    baseline = files.run('exec "$@"', [native_mkdir, '-p', 'reference/leaf'], None)
    assert baseline.returncode == 0, baseline.stderr
    try:
        files.mkdir('selected/leaf', parents=True)
        assert (source / 'selected').stat().st_mode == (source / 'reference').stat().st_mode
        assert (source / 'selected/leaf').stat().st_mode == (source / 'reference/leaf').stat().st_mode
        files.mkdir('selected/leaf', parents=True)
        with pytest.raises(TaskFileError):
            files.mkdir('selected/leaf')
    finally:
        for name in ('reference', 'selected'):
            folder = source / name
            if folder.exists():
                folder.chmod(0o700)
                if (folder / 'leaf').exists():
                    (folder / 'leaf').chmod(0o700)
