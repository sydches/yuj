"""Local typed tools retain checked directories without requiring a sandbox."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness import local_file_access as access
from scripts.llm_solver.harness._tools.edit import edit
from scripts.llm_solver.harness._tools.write import write
from scripts.llm_solver.harness.apply_patch import parse_patch, verify_and_apply, PatchVerifyError
from scripts.llm_solver.harness.udiff import (
    parse_unified_diff, verify_and_apply_unified_diff, UnifiedDiffApplyError,
)


def fixture_paths(tmp_path):
    task = tmp_path / 'task'
    parent = task / 'nested'
    parent.mkdir(parents=True)
    target = parent / 'file.txt'
    target.write_bytes(b'before\n')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / target.name).write_bytes(b'outside must remain intact\n')
    return task, parent, target, outside


@pytest.mark.parametrize('race', ['parent_before_open', 'final_link', 'parent_after_open'])
def test_checked_open_refuses_new_links_or_keeps_its_open_parent(tmp_path, monkeypatch, race):
    task, parent, target, outside = fixture_paths(tmp_path)
    opened = os.open
    fired = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal fired
        selected = parent.name if race == 'parent_before_open' else target.name
        if not fired and path == selected and kwargs.get('dir_fd') is not None:
            fired = True
            if race == 'final_link':
                target.unlink()
                target.symlink_to(outside / target.name)
            else:
                parent.rename(task / 'retained')
                parent.symlink_to(outside, target_is_directory=True)
        return opened(path, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, 'open', racing_open)
    if race == 'parent_after_open':
        access.write_bytes(task, target, b'after\n')
        assert (task / 'retained' / target.name).read_bytes() == b'after\n'
    else:
        with pytest.raises(OSError):
            access.write_bytes(task, target, b'after\n')
    assert fired
    assert (outside / target.name).read_bytes() == b'outside must remain intact\n'


def test_local_aliases_directory_roots_nested_creation_and_binary_bytes(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    alias = task / 'alias'
    real = task / 'real'
    real.mkdir()
    alias.symlink_to(real, target_is_directory=True)
    assert write('alias/new/deep/file', 'text', cwd=str(task)).startswith('OK')
    target = real / 'new/deep/file'
    access.write_bytes(task, target, b'\0\xff\r\n')
    assert access.read_bytes(task, target) == b'\0\xff\r\n'
    assert access.read_text(task, target, encoding='utf-8', errors='replace') == '\0\ufffd\n'
    with access.open_local_file(task, task, os.O_RDONLY | os.O_DIRECTORY) as descriptor:
        assert set(os.listdir(descriptor)) == {'alias', 'real'}


def test_existing_files_in_search_only_directories_keep_permissions(tmp_path):
    task, parent, target, _outside = fixture_paths(tmp_path)
    parent.chmod(0o111)
    try:
        assert access.read_bytes(task, target) == b'before\n'
        access.write_bytes(task, target, b'after\n')
        assert access.read_bytes(task, target) == b'after\n'
    finally:
        parent.chmod(0o755)


def test_write_only_file_does_not_gain_a_read_permission_requirement(tmp_path):
    target = tmp_path / 'file.txt'
    target.write_text('before')
    target.chmod(0o200)
    try:
        assert write('file.txt', 'after', cwd=str(tmp_path)).startswith('OK')
        assert target.stat().st_mode & 0o777 == 0o200
    finally:
        target.chmod(0o600)
    assert target.read_text() == 'after'


def test_checked_root_cannot_be_replaced_by_a_symlink_during_traversal(tmp_path, monkeypatch):
    task, _parent, target, outside = fixture_paths(tmp_path)
    opened = os.open
    fired = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal fired
        if not fired and path == task.name and kwargs.get('dir_fd') is not None:
            fired = True
            task.rename(tmp_path / 'retained-task')
            task.symlink_to(outside, target_is_directory=True)
        return opened(path, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, 'open', racing_open)
    with pytest.raises(OSError):
        access.write_bytes(task, target, b'after\n')
    assert fired
    assert (outside / target.name).read_bytes() == b'outside must remain intact\n'


@pytest.mark.parametrize('tool', ['write', 'edit', 'apply_patch', 'udiff'])
def test_mutation_tools_refuse_a_final_symlink_before_truncation(tmp_path, monkeypatch, tool):
    task, _parent, target, outside = fixture_paths(tmp_path)
    opened = os.open
    fired = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal fired
        if not fired and path == target.name and flags & os.O_WRONLY:
            fired = True
            target.unlink()
            target.symlink_to(outside / target.name)
        return opened(path, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, 'open', racing_open)
    if tool == 'write':
        assert write('nested/file.txt', 'after\n', cwd=str(task)).startswith('ERROR:')
    elif tool == 'edit':
        assert edit('nested/file.txt', 'before', 'after', cwd=str(task)).startswith('ERROR:')
    elif tool == 'apply_patch':
        patch = '*** Begin Patch\n*** Update File: nested/file.txt\n@@\n-before\n+after\n*** End Patch'
        with pytest.raises(PatchVerifyError):
            verify_and_apply(parse_patch(patch), str(task))
    else:
        patch = '--- a/nested/file.txt\n+++ b/nested/file.txt\n@@ -1 +1 @@\n-before\n+after\n'
        with pytest.raises(UnifiedDiffApplyError):
            verify_and_apply_unified_diff(parse_unified_diff(patch), str(task))
    assert fired
    assert (outside / target.name).read_bytes() == b'outside must remain intact\n'


@pytest.mark.parametrize('tool', ['write', 'edit'])
def test_postcheck_rollback_cannot_follow_a_replaced_parent(tmp_path, monkeypatch, tool):
    task, parent, target, outside = fixture_paths(tmp_path)

    def block(*args, **kwargs):
        parent.rename(task / 'retained')
        parent.symlink_to(outside, target_is_directory=True)
        return SimpleNamespace(action='block', check_name='test-check', output='')

    monkeypatch.setattr('scripts.llm_solver.harness.post_edit.run_post_edit_checks', block)
    result = (write('nested/file.txt', 'after\n', cwd=str(task)) if tool == 'write' else
              edit('nested/file.txt', 'before', 'after', cwd=str(task)))
    assert result.startswith('ERROR:')
    assert (outside / target.name).read_bytes() == b'outside must remain intact\n'


def test_delete_keeps_the_parent_it_checked(tmp_path, monkeypatch):
    task, parent, target, outside = fixture_paths(tmp_path)
    unlink = os.unlink

    def racing_unlink(path, *args, **kwargs):
        if path == target.name and kwargs.get('dir_fd') is not None:
            parent.rename(task / 'retained')
            parent.symlink_to(outside, target_is_directory=True)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(access.os, 'unlink', racing_unlink)
    access.unlink(task, target)
    assert not (task / 'retained' / target.name).exists()
    assert (outside / target.name).read_bytes() == b'outside must remain intact\n'
