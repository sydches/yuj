"""Directory discovery must retain the directory whose access was checked."""
import shlex

import pytest

from scripts.llm_solver.harness._tools.glob import glob_files
from scripts.llm_solver.harness.task_path import activate_task_files
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('swap_after_check', [False, True])
def test_glob_does_not_enumerate_a_swapped_outside_directory(bwrap, tmp_path, swap_after_check):
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
        result = glob_files('folder/*.txt', cwd=str(source))
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
