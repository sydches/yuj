"""Native directory-entry changes stay attached to their checked parent."""
import shlex
import re
from pathlib import PurePosixPath

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_files import TaskFileError, TaskUtilityUnavailable


@pytest.mark.parametrize('operation', ['write', 'write_new', 'create', 'replace', 'unlink', 'link', 'rmdir', 'chmod', 'chmod_dir'])
@pytest.mark.parametrize('change_at', ['resolution', 'entered_parent'])
def test_entry_mutations_keep_the_checked_parent(bwrap, tmp_path, operation, change_at):
    outside = tmp_path / 'outside'
    outside.mkdir()
    # Allow fixture writes outside the task so path redirection is observable;
    # a read-only outer mount must not hide a missing task containment check.
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    folder = source / 'folder'
    folder.mkdir()
    if operation in ('write', 'replace', 'unlink', 'chmod'):
        (folder / 'entry').write_bytes(b'SELECTED')
        (outside / 'entry').write_bytes(b'OUTSIDE')
    if operation in ('rmdir', 'chmod_dir'):
        (folder / 'entry').mkdir()
        (outside / 'entry').mkdir()
    if operation == 'chmod':
        (outside / 'entry').chmod(0o600)
    elif operation == 'chmod_dir':
        (outside / 'entry').chmod(0o755)
    root = str(files.root)
    native_folder = root + '/folder'
    swap = (f'mv -- {shlex.quote(native_folder)} {shlex.quote(root + "/saved")}; '
            f'ln -s -- {shlex.quote(str(outside))} {shlex.quote(native_folder)}')
    wrapper = source / 'controlled-observer'
    if change_at == 'resolution':
        observer = files._utility('realpath')
        observed_path = native_folder + '/entry' if operation.startswith(('write', 'chmod')) else native_folder
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
            f'{swap}\n'
            "printf '%s\\0' \"$value\"\n")
        files._utilities['readlink'] = root + '/controlled-observer'
    wrapper.chmod(0o755)

    def mutate():
        path = 'folder/entry'
        if operation.startswith('write'):
            files.write_bytes(path, b'NEW\x00\xff')
        elif operation == 'create':
            files.create_bytes(path, b'NEW\x00\xff')
        elif operation == 'replace':
            files.replace_bytes(path, b'NEW\x00\xff', mode=0o640)
        elif operation == 'unlink':
            files.unlink(path)
        elif operation == 'link':
            files.symlink_to(path, 'literal-target')
        elif operation.startswith('chmod'):
            files.chmod(path, 0o700)
        else:
            files.rmdir(path)

    if change_at == 'resolution':
        with pytest.raises(PermissionError, match='escapes task root'):
            mutate()
    else:
        mutate()
        assert folder.is_symlink(), 'the controlled parent change must have run'
        entry = source / 'saved' / 'entry'
        if operation in ('write', 'write_new', 'create', 'replace'):
            assert entry.read_bytes() == b'NEW\x00\xff'
            if operation == 'replace':
                assert entry.stat().st_mode & 0o777 == 0o640
        elif operation == 'link':
            assert str(entry.readlink()) == 'literal-target'
        elif operation.startswith('chmod'):
            assert entry.stat().st_mode & 0o777 == 0o700
        else:
            assert not entry.exists()
    if operation in ('write', 'replace', 'unlink', 'chmod'):
        assert (outside / 'entry').read_bytes() == b'OUTSIDE'
        if operation == 'chmod':
            assert (outside / 'entry').stat().st_mode & 0o777 != 0o700
    elif operation in ('rmdir', 'chmod_dir'):
        assert (outside / 'entry').is_dir()
        if operation == 'chmod_dir':
            assert (outside / 'entry').stat().st_mode & 0o777 != 0o700
    else:
        assert list(outside.iterdir()) == []


def test_removal_does_not_follow_the_final_directory_symlink(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'directory').mkdir()
    (source / 'alias').symlink_to('directory')
    with pytest.raises(TaskFileError):
        files.rmdir('alias')
    assert (source / 'directory').is_dir()
    assert (source / 'alias').is_symlink()
    files.unlink('alias')
    assert not (source / 'alias').is_symlink()
    assert (source / 'directory').is_dir()


def test_entry_creation_needs_parent_search_permission_not_listing(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    folder = source / 'folder'
    folder.mkdir(mode=0o300)
    try:
        files.create_bytes('folder/entry', b'created')
        assert (folder / 'entry').read_bytes() == b'created'
        files.replace_bytes('folder/entry', b'replaced', mode=0o600)
        assert (folder / 'entry').read_bytes() == b'replaced'
        files.unlink('folder/entry')
        assert not (folder / 'entry').exists()
    finally:
        folder.chmod(0o700)


@pytest.mark.parametrize('exists', [False, True])
def test_write_refuses_final_symlink_installed_at_open(bwrap, tmp_path, exists):
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker'
    marker.write_bytes(b'OUTSIDE')
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    if exists:
        (source / 'entry').write_bytes(b'SELECTED')
    native_dd = files._utility('dd')
    wrapper = source / 'controlled-writer'
    wrapper.write_text(
        '#!/bin/bash\n'
        'rm -f -- ./entry\n'
        f'ln -s -- {shlex.quote(str(marker))} ./entry\n'
        f'exec {shlex.quote(native_dd)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['dd'] = str(files.root / 'controlled-writer')
    with pytest.raises(TaskFileError):
        files.write_bytes('entry', b'CHANGED')
    assert (source / 'entry').is_symlink(), 'the controlled pre-open change must have run'
    assert marker.read_bytes() == b'OUTSIDE'


def test_write_preserves_writeonly_inode_and_resolves_contained_alias(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    entry = source / 'entry'
    entry.write_bytes(b'old longer content')
    (source / 'alias').symlink_to('entry')
    inode = entry.stat().st_ino
    entry.chmod(0o200)
    try:
        assert files.write_bytes('alias', b'new\x00\xff') == 5
        assert entry.stat().st_ino == inode
        assert entry.stat().st_mode & 0o777 == 0o200
        assert (source / 'alias').is_symlink()
    finally:
        entry.chmod(0o600)
    assert entry.read_bytes() == b'new\x00\xff'


@pytest.mark.parametrize('exists', [False, True])
def test_write_does_not_retry_a_changed_creation_state(bwrap, tmp_path, exists):
    source, files = namespace_files(bwrap, tmp_path)
    entry = source / 'entry'
    if exists:
        entry.write_bytes(b'SELECTED')
    native_dd = files._utility('dd')
    wrapper = source / 'controlled-writer'
    change = 'rm -- ./entry' if exists else "printf 'CONCURRENT' > ./entry"
    wrapper.write_text(
        '#!/bin/bash\n'
        f'{change}\n'
        f'exec {shlex.quote(native_dd)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['dd'] = str(files.root / 'controlled-writer')
    with pytest.raises(TaskFileError):
        files.write_bytes('entry', b'CHANGED')
    if exists:
        assert not entry.exists()
    else:
        assert entry.read_bytes() == b'CONCURRENT'


def test_empty_writes_create_and_truncate(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    assert files.write_bytes('new', b'') == 0
    assert (source / 'new').read_bytes() == b''
    (source / 'existing').write_bytes(b'long content')
    assert files.write_bytes('existing', b'') == 0
    assert (source / 'existing').read_bytes() == b''


def test_chmod_does_not_follow_a_final_symlink_at_execution(bwrap, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker'
    marker.write_bytes(b'OUTSIDE')
    marker.chmod(0o600)
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    (source / 'entry').write_bytes(b'SELECTED')
    native_chmod = files._utility('chmod')
    wrapper = source / 'controlled-chmod'
    wrapper.write_text(
        '#!/bin/bash\n'
        'rm -- ./entry\n'
        f'ln -s -- {shlex.quote(str(marker))} ./entry\n'
        f'exec {shlex.quote(native_chmod)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['chmod'] = str(files.root / 'controlled-chmod')
    with pytest.raises(TaskFileError):
        files.chmod('entry', 0o700)
    assert (source / 'entry').is_symlink(), 'the controlled pre-chmod change must have run'
    assert marker.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('is_directory', [False, True])
def test_chmod_can_restore_permissions_through_a_contained_alias(bwrap, tmp_path, is_directory):
    source, files = namespace_files(bwrap, tmp_path)
    entry = source / 'entry'
    if is_directory:
        entry.mkdir()
    else:
        entry.write_bytes(b'SELECTED')
    (source / 'alias').symlink_to('entry')
    entry.chmod(0)
    try:
        files.chmod('alias', 0o700)
        assert entry.stat().st_mode & 0o777 == 0o700
        assert (source / 'alias').is_symlink()
    finally:
        entry.chmod(0o700)


def test_replacement_does_not_chmod_a_retargeted_temporary_entry(bwrap, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker'
    marker.write_bytes(b'OUTSIDE')
    marker.chmod(0o600)
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    (source / 'entry').write_bytes(b'SELECTED')
    native_chmod = files._utility('chmod')
    wrapper = source / 'controlled-chmod'
    wrapper.write_text(
        '#!/bin/bash\n'
        'temporary=${@: -1}\n'
        'rm -- "$temporary"\n'
        f'ln -s -- {shlex.quote(str(marker))} "$temporary"\n'
        ': > ./chmod-called\n'
        f'exec {shlex.quote(native_chmod)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['chmod'] = str(files.root / 'controlled-chmod')
    with pytest.raises(TaskFileError):
        files.replace_bytes('entry', b'REPLACEMENT', mode=0o700)
    assert (source / 'chmod-called').exists()
    assert (source / 'entry').read_bytes() == b'SELECTED'
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.read_bytes() == b'OUTSIDE'


@pytest.mark.parametrize('operation', ['create', 'replace'])
@pytest.mark.parametrize('change', ['symlink', 'removed'])
def test_temporary_content_write_refuses_a_changed_entry(bwrap, tmp_path, operation, change):
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker'
    marker.write_bytes(b'OUTSIDE')
    source, files = namespace_files(bwrap, tmp_path, writable=(outside,))
    if operation == 'replace':
        (source / 'entry').write_bytes(b'SELECTED')
    native_mktemp = files._utility('mktemp')
    wrapper = source / 'controlled-mktemp'
    replacement = f'ln -s -- {shlex.quote(str(marker))} "$temporary"\n' if change == 'symlink' else ''
    wrapper.write_text(
        '#!/bin/bash\n'
        f'temporary=$({shlex.quote(native_mktemp)} "$@") || exit\n'
        'rm -- "$temporary"\n'
        + replacement +
        ': > ./temporary-changed\n'
        "printf '%s\\n' \"$temporary\"\n")
    wrapper.chmod(0o755)
    files._utilities['mktemp'] = str(files.root / 'controlled-mktemp')
    with pytest.raises(TaskFileError):
        if operation == 'create':
            files.create_bytes('entry', b'NEW\x00\xff')
        else:
            files.replace_bytes('entry', b'NEW\x00\xff', mode=0o600)
    assert (source / 'temporary-changed').exists()
    assert marker.read_bytes() == b'OUTSIDE'
    if operation == 'replace':
        assert (source / 'entry').read_bytes() == b'SELECTED'
    else:
        assert not (source / 'entry').exists()
    assert not list(source.glob('.yuj-*'))


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('mode', [0o000, 0o100, 0o200])
def test_replacement_can_publish_without_read_permission(bwrap, tmp_path, existing, mode):
    source, files = namespace_files(bwrap, tmp_path)
    entry = source / 'entry'
    if existing:
        entry.write_bytes(b'SELECTED')
    try:
        files.replace_bytes('entry', b'NEW\x00\xff', mode=mode)
        assert entry.stat().st_mode & 0o777 == mode
        entry.chmod(0o600)
        assert entry.read_bytes() == b'NEW\x00\xff'
        assert not list(source.glob('.yuj-*'))
    finally:
        if entry.exists():
            entry.chmod(0o600)


@pytest.mark.parametrize('flush_error', [False, True])
def test_replacement_writer_flushes_its_open_temporary_file(bwrap, tmp_path, flush_error):
    source, files = namespace_files(bwrap, tmp_path)
    try:
        tracer = files._utility('strace')
    except TaskUtilityUnavailable:
        pytest.skip('native syscall tracing is unavailable')
    native_dd = files._utility('dd')
    wrapper = source / 'traced-writer'
    trace = str(files.root / 'flush.trace')
    injection = '-e inject=fdatasync:error=EIO:when=1 ' if flush_error else ''
    wrapper.write_text(
        '#!/bin/bash\n'
        f'exec {shlex.quote(tracer)} -f -yy -e trace=fdatasync -o {shlex.quote(trace)} '
        + injection +
        f'{shlex.quote(native_dd)} "$@"\n')
    wrapper.chmod(0o755)
    files._utilities['dd'] = str(files.root / 'traced-writer')
    entry = source / 'entry'
    entry.write_bytes(b'SELECTED')
    entry.chmod(0o640)
    try:
        if flush_error:
            with pytest.raises(TaskFileError):
                files.replace_bytes('entry', b'FLUSHED\x00\xff', mode=0)
            assert entry.read_bytes() == b'SELECTED'
            assert entry.stat().st_mode & 0o777 == 0o640
            result = r'-1 EIO .*\(INJECTED\)'
        else:
            files.replace_bytes('entry', b'FLUSHED\x00\xff', mode=0)
            assert entry.stat().st_mode & 0o777 == 0
            entry.chmod(0o600)
            assert entry.read_bytes() == b'FLUSHED\x00\xff'
            result = '0'
        flushed = re.findall(r'fdatasync\(\d+<([^>]+)>\)\s*=\s*' + result,
                             (source / 'flush.trace').read_text())
        assert any(PurePosixPath(path).parent == files.root and
                   PurePosixPath(path).name.startswith('.yuj-restore-')
                   for path in flushed), flushed
        assert not list(source.glob('.yuj-*'))
    finally:
        if entry.exists():
            entry.chmod(0o600)
