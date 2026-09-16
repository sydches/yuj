"""Directory discovery must retain the directory whose access was checked."""
import shlex

import pytest

from scripts.llm_solver.harness._tools.glob import glob_files
from scripts.llm_solver.harness.task_path import activate_task_files
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('swap_after_check', [False, True])
@pytest.mark.parametrize('pattern', ['folder/*.txt', '**/*.txt', '**/**/*.txt'])
def test_glob_does_not_enumerate_a_swapped_outside_directory(bwrap, tmp_path, swap_after_check, pattern):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'folder').mkdir()
    (source / 'folder' / 'visible.txt').write_text('task bytes')
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = str(files.root)
    (outside / 'leaked_name.txt').symlink_to(root + '/saved/visible.txt')
    observer = files._utility('readlink')
    wrapper = source / 'controlled-readlink'
    wrapper.write_text(
        '#!/bin/bash\n'
        f'value=$({shlex.quote(observer)} "$@") || exit\n'
        f'if [[ "$value" == {shlex.quote(root + "/folder")} '
        f'&& ! -L {shlex.quote(root + "/folder")} ]]; then\n'
        f'mv -- {shlex.quote(root + "/folder")} {shlex.quote(root + "/saved")}\n'
        f'ln -s -- {shlex.quote(str(outside))} {shlex.quote(root + "/folder")}\n'
        'fi\n'
        + ('printf "%s\\n" "$value"\n' if swap_after_check else
           f'exec {shlex.quote(observer)} "$@"\n'))
    wrapper.chmod(0o755)
    files._utilities['readlink'] = root + '/controlled-readlink'
    with activate_task_files(files, host_root=source):
        result = glob_files(pattern, cwd=str(source))
    assert (source / 'folder').is_symlink(), 'controlled swap did not run'
    assert 'leaked_name' not in result


def test_glob_keeps_inaccessible_directory_handling(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'folder').mkdir()
    (source / 'folder' / 'file.py').write_text('task bytes')
    (source / 'folder').chmod(0)
    try:
        with activate_task_files(files, host_root=source):
            assert glob_files('folder/*.py', cwd=str(source)) == 'No files found.'
    finally:
        (source / 'folder').chmod(0o700)


def test_recursive_glob_checks_each_retained_directory_once(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'folder').mkdir()
    (source / 'folder/code.py').write_text('body')
    observer = files._utility('readlink')
    log = str(files.root / 'directory-checks')
    wrapper = source / 'observed-readlink'
    wrapper.write_text('#!/bin/bash\n'
        f'if [[ "$2" == /proc/self/fd/* ]]; then printf "checked\\n" >> {shlex.quote(log)}; fi\n'
        f'exec {shlex.quote(observer)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['readlink'] = str(files.root / wrapper.name)
    with activate_task_files(files, host_root=source):
        assert glob_files('**/*.py', cwd=str(source)) == 'folder/code.py'
    assert (source / 'directory-checks').read_text().splitlines() == ['checked', 'checked']
