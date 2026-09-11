"""Deterministic native path changes around the file helper's read boundary."""
import shlex

import pytest

from scripts.llm_solver.harness.task_files import TaskFileError, TaskUtilityUnavailable
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('operation', ['read', 'range', 'search_rg', 'search_grep'])
@pytest.mark.parametrize('change_at', ['resolution', 'opened_file'])
def test_reads_do_not_follow_a_retargeted_parent(bwrap, tmp_path, monkeypatch, operation, change_at):
    source, files = namespace_files(bwrap, tmp_path)
    if operation.startswith('search_'):
        select_search_utility(files, operation.removeprefix('search_'), monkeypatch)
    folder = source / 'folder'
    folder.mkdir()
    selected = b'SELECTED\n' if operation.startswith('search_') else b'SELECTED\x00\xff'
    (folder / 'file').write_bytes(selected)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'file').write_bytes(b'OUTSIDE\x00\xff')
    root = str(files.root)
    native_folder = root + '/folder'
    swap = (f'mv -- {shlex.quote(native_folder)} {shlex.quote(root + "/saved")}; '
            f'ln -s -- {shlex.quote(str(outside))} {shlex.quote(native_folder)}')
    wrapper = source / 'controlled-observer'
    if change_at == 'resolution':
        realpath = files._utility('realpath')
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(realpath)} "$@"; code=$?; printf .; exit "$code") || exit\n'
            'value=${value%.}\n'
            f'if [[ "${{@: -1}}" == {shlex.quote(native_folder + "/file")} ]]; then {swap}; fi\n'
            'printf %s "$value"\n')
        files._utilities['realpath'] = root + '/controlled-observer'
    else:
        readlink = files._utility('readlink')
        wrapper.write_text(
            '#!/bin/bash\n'
            f'value=$({shlex.quote(readlink)} -- "${{@: -1}}"; code=$?; printf .; exit "$code") || exit\n'
            "value=${value%.}; value=${value%$'\\n'}\n"
            f'{swap}\n'
            "printf '%s\\0' \"$value\"\n")
        files._utilities['readlink'] = root + '/controlled-observer'
    wrapper.chmod(0o755)

    def read():
        if operation.startswith('search_'):
            return files.search('folder/file', 'SELECTED')
        return (files.read_bytes('folder/file') if operation == 'read'
                else files.read_range('folder/file', 1, 5))

    if change_at == 'resolution':
        with pytest.raises(PermissionError, match='escapes task root'):
            read()
    else:
        expected = (root + '/folder/file:1:SELECTED\n').encode() if operation.startswith('search_') else (
            selected if operation == 'read' else b'ELECT')
        assert read() == expected
        assert folder.is_symlink(), 'the controlled post-open change must have run'
    assert (outside / 'file').read_bytes() == b'OUTSIDE\x00\xff'


def select_search_utility(files, utility, monkeypatch):
    try:
        files._utility(utility)
    except TaskUtilityUnavailable:
        pytest.skip(f'{utility} is unavailable in the fixture namespace')
    discover = files._utility

    def selected(name):
        if utility == 'grep' and name == 'rg':
            raise TaskUtilityUnavailable('exercise the native grep fallback')
        return discover(name)

    monkeypatch.setattr(files, '_utility', selected)


@pytest.mark.parametrize('utility', ['rg', 'grep'])
def test_search_preserves_alias_labels_and_no_match(bwrap, tmp_path, monkeypatch, utility):
    source, files = namespace_files(bwrap, tmp_path)
    select_search_utility(files, utility, monkeypatch)
    (source / 'file').write_bytes(b'first\nmatch:colon\nlast match')
    (source / 'alias').symlink_to('file')
    label = str(files.root / 'alias').encode()
    assert files.search('alias', 'match') == (
        label + b':2:match:colon\n' + label + b':3:last match\n')
    assert files.search('alias', 'absent') == b''
    with pytest.raises(TaskFileError, match='search failed'):
        files.search('alias', '[unclosed')
