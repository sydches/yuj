"""Reuse checked read evidence without losing bytes, freshness or containment."""
import hashlib
import os
import shlex
import subprocess

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness import local_file_access
from scripts.llm_solver.harness.stale_guard import StaleFileGuard
from scripts.llm_solver.harness.task_files import NamespaceFiles, TaskFileError
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.tools import dispatch


def native_files(root):
    calls = []

    def run(script, args, data):
        calls.append(tuple(args))
        return subprocess.run(['bash', '--noprofile', '--norc', '-p', '-c',
                               script, 'read-test', *args], input=data, capture_output=True)

    return NamespaceFiles(str(root), run, binding={'test': 'checked-read'}), calls


@pytest.mark.parametrize('rules, directory, hidden', [
    ('', False, False), ('', True, False),
    ('target/\n', False, False), ('target/\n', True, True),
    ('target\n!target/\n', False, True), ('target\n!target/\n', True, False),
])
def test_read_keeps_type_dependent_visibility_before_access(tmp_path, monkeypatch, rules, directory, hidden):
    from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy

    (tmp_path / '.yujignore').write_text(rules)
    target = tmp_path / 'target'
    target.mkdir() if directory else target.write_text('visible bytes')
    policy = load_ignore_policy(tmp_path)
    if hidden:
        def forbidden(*args, **kwargs):
            raise AssertionError('hidden content was accessed')
        monkeypatch.setattr(local_file_access, 'read_bytes', forbidden)
        monkeypatch.setattr(local_file_access, 'read_observation', forbidden)
    result = dispatch('read', {'path': 'target'}, cwd=str(tmp_path),
                      cfg=make_config(), ignore_policy=policy)
    assert ('file not found' in result) is hidden
    if not hidden:
        assert ('is a directory' in result) is directory
        if not directory:
            assert 'visible bytes' in result


def test_native_read_preserves_binary_framing_and_uses_one_warm_operation(tmp_path):
    name = "literal\nfile\n"
    data = b'\0\xff\n81a4\0header-like\0\0'
    (tmp_path / name).write_bytes(data)
    (tmp_path / 'alias').symlink_to(name)
    files, calls = native_files(tmp_path)
    files.read_observation(name)  # Discover utilities once.
    calls.clear()
    actual, metadata = files.read_observation('alias')
    assert actual == data
    assert metadata.size == len(data)
    assert metadata.mtime_ns == (tmp_path / name).stat().st_mtime_ns
    assert len(calls) == 1
    assert calls[0][3] == 'read_observation'
    with pytest.raises(IsADirectoryError):
        files.read_observation('.')
    with pytest.raises(FileNotFoundError):
        files.read_observation('missing')


@pytest.mark.parametrize('continuous', [False, True])
def test_native_read_retries_changed_bytes_and_refuses_unstable_file(tmp_path, continuous):
    target = tmp_path / 'file'
    target.write_bytes(b'original')
    files, calls = native_files(tmp_path)
    cat = tmp_path / 'racing-cat'
    mutation = 'printf x >> ' + shlex.quote(str(target))
    # Change the bytes once, or on every attempt, without relying on timing.
    if not continuous:
        mutation = ('if [[ ! -e ' + shlex.quote(str(tmp_path / 'once')) + ' ]]; then\n'
                    'printf changed > ' + shlex.quote(str(target)) + '\n'
                    'touch ' + shlex.quote(str(tmp_path / 'once')) + '\nfi')
    cat.write_text('#!/bin/bash\n/bin/cat "$@"\n' + mutation + '\n')
    cat.chmod(0o700)
    files._utilities['cat'] = str(cat)
    if continuous:
        with pytest.raises(TaskFileError, match='changed while being read'):
            files.read_observation('file')
    else:
        data, metadata = files.read_observation('file')
        assert data == b'changed' and metadata.size == 7
    assert sum(len(call) > 3 and call[3] == 'read_observation' for call in calls) == (3 if continuous else 2)


def test_native_read_refuses_a_link_retargeted_outside_before_open(tmp_path):
    root = tmp_path / 'task'
    root.mkdir()
    target = root / 'file'
    target.write_text('inside')
    outside = tmp_path / 'outside'
    outside.write_text('outside')
    files, _calls = native_files(root)
    realpath = root / 'racing-realpath'
    realpath.write_text('#!/bin/bash\n/usr/bin/realpath "$@"\n'
                        'if [[ "${!#}" == ' + shlex.quote(str(target)) + ' ]]; then\n'
                        'rm -- ' + shlex.quote(str(target)) + '\n'
                        'ln -s -- ' + shlex.quote(str(outside)) + ' ' + shlex.quote(str(target)) + '\nfi\n')
    realpath.chmod(0o700)
    files._utilities['realpath'] = str(realpath)
    with pytest.raises(PermissionError):
        files.read_observation('file')


def test_local_read_retries_on_descriptor_change(tmp_path, monkeypatch):
    target = tmp_path / 'file'
    target.write_bytes(b'original')
    fstat = os.fstat
    calls = 0

    def changing_stat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_bytes(b'changed')
        return fstat(fd)

    monkeypatch.setattr(local_file_access.os, 'fstat', changing_stat)
    data, mtime_ns = local_file_access.read_observation(tmp_path, target)
    assert data == b'changed'
    assert mtime_ns == target.stat().st_mtime_ns
    assert calls == 4


@pytest.mark.parametrize('native', [False, True])
def test_dispatch_records_bytes_read_without_rereading_later_change(tmp_path, monkeypatch, native):
    target = tmp_path / 'file'
    target.write_text('before\nsecond line\n')
    cfg = make_config(sandbox_bash=False, tools_stale_guard_mode='block')
    guard = StaleFileGuard(cwd=tmp_path, mode='block')
    observe = guard.observe_inspection

    def changed_after_read(*args):
        target.write_text('external change\n')
        return observe(*args)

    monkeypatch.setattr(guard, 'observe_inspection', changed_after_read)
    from contextlib import nullcontext
    files, _calls = native_files(tmp_path)
    with activate_task_files(files, host_root=tmp_path) if native else nullcontext():
        result = dispatch('read', {'path': 'file', 'limit': 1}, cwd=str(tmp_path),
                          cfg=cfg, stale_guard=guard, effective_env={})
        assert 'before' in result and 'external change' not in result
        assert guard.ledger_snapshot()['file'].sha256 == hashlib.sha256(b'before\nsecond line\n').hexdigest()
        assert guard.check_edit('file').reason == 'modified'


def test_typed_read_does_not_call_guard_fingerprint(tmp_path, monkeypatch):
    (tmp_path / 'file').write_text('source')
    guard = StaleFileGuard(cwd=tmp_path)

    def unexpected(*args):
        pytest.fail('typed read repeated its fingerprint')

    monkeypatch.setattr(guard, '_fingerprint', unexpected)
    result = dispatch('read', {'path': 'file'}, cwd=str(tmp_path), cfg=make_config(),
                      stale_guard=guard, effective_env={})
    assert 'source' in result
    assert 'file' in guard.ledger_snapshot()


def test_read_without_guard_does_not_collect_unused_metadata(tmp_path, monkeypatch):
    (tmp_path / 'file').write_text('source')

    def unexpected(*args):
        pytest.fail('read without stale guard collected metadata')

    monkeypatch.setattr(local_file_access, 'read_observation', unexpected)
    result = dispatch('read', {'path': 'file'}, cwd=str(tmp_path), cfg=make_config(),
                      effective_env={})
    assert 'source' in result
