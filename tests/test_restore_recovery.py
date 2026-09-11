"""Failed restores undo earlier changes or retain explicit recovery evidence."""
import fcntl
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.workspace_checkpoints import (
    RestoreRecoveryError, WorkspaceCheckpointError, WorkspaceCheckpointStore,
)


def test_native_later_readonly_failure_recovers_bytes_links_and_directory_modes(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'later').write_bytes(b'captured readonly')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'first').write_bytes(b'captured first')
    (source / 'a-link').symlink_to('first')
    store = WorkspaceCheckpointStore(tmp_path / 'view', shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=tmp_path / 'view'):
        store.capture(1)
        (source / 'first').write_bytes(b'current\0\xff\n')
        (source / 'first').chmod(0o600)
        (source / 'a-link').unlink()
        (source / 'a-link').symlink_to('missing target')
        extra = source / 'removed/nested/extra'
        extra.parent.mkdir(parents=True)
        extra.parent.parent.chmod(0o700)
        extra.write_bytes(b'current extra')
        (overlay / 'later').write_bytes(b'current readonly')
        with pytest.raises(OSError):
            store.restore_checkpoint(1)
    assert (source / 'first').read_bytes() == b'current\0\xff\n'
    assert stat.S_IMODE((source / 'first').stat().st_mode) == 0o600
    assert os.readlink(source / 'a-link') == 'missing target'
    assert extra.read_bytes() == b'current extra'
    assert stat.S_IMODE(extra.parent.parent.stat().st_mode) == 0o700
    assert (overlay / 'later').read_bytes() == b'current readonly'
    assert not (store.shadow_dir / '.restore_recovery').exists()


def local_store(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'a').write_text('captured a')
    (task / 'z').write_text('captured z')
    store = WorkspaceCheckpointStore(task, shadow_dir=tmp_path / 'shadow')
    store.capture(1)
    (task / 'a').write_text('current a')
    (task / 'z').write_text('current z')
    return task, store


def test_failed_restore_removes_created_parent_directories(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    (task / 'created/nested').mkdir(parents=True)
    (task / 'created/nested/file').write_text('captured new file')
    (task / 'z').write_text('captured z')
    store = WorkspaceCheckpointStore(task, shadow_dir=tmp_path / 'shadow')
    store.capture(1)
    shutil.rmtree(task / 'created')
    (task / 'z').write_text('current z')
    write = store._write_regular_file

    def fail_later(target, data, mode):
        if target.name == 'z':
            raise OSError('injected later write failure')
        return write(target, data, mode)

    monkeypatch.setattr(store, '_write_regular_file', fail_later)
    with pytest.raises(OSError, match='injected later'):
        store.restore_checkpoint(1)
    assert not (task / 'created').exists()
    assert (task / 'z').read_text() == 'current z'


def test_unexpected_file_change_retains_journal_and_reopened_store_can_recover(tmp_path, monkeypatch):
    task, store = local_store(tmp_path)
    (task / 'extra').write_text('current extra')
    write = store._write_regular_file

    def fail_with_external_change(target, data, mode):
        write(target, data, mode)
        if target.name == 'a':
            target.write_text('external change')
            raise OSError('injected interrupted write')

    monkeypatch.setattr(store, '_write_regular_file', fail_with_external_change)
    with pytest.raises(RestoreRecoveryError) as caught:
        store.restore_checkpoint(1)
    directory = caught.value.recovery_path
    record = json.loads((directory / 'state.json').read_text())
    assert record['phase'] == 'incomplete'
    assert (directory / record['before']['a']['blob']).read_text() == 'current a'
    assert (task / 'a').read_text() == 'external change'
    with pytest.raises(RestoreRecoveryError):
        store.capture(2)
    # The owner resolves the unexpected change to a recorded restore state.
    (task / 'a').write_text('captured a')
    reopened = WorkspaceCheckpointStore(task, shadow_dir=store.shadow_dir)
    assert reopened.recover_incomplete_restore() is True
    assert (task / 'a').read_text() == 'current a'
    assert (task / 'extra').read_text() == 'current extra'
    assert not directory.exists()
    assert reopened.recover_incomplete_restore() is False


def test_exhausted_allowance_keeps_undo_bytes_for_later_recovery(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import time_budget
    task, store = local_store(tmp_path)
    write = store._write_regular_file
    clock = [100.0]
    monkeypatch.setattr(time_budget.time, 'monotonic', lambda: clock[0])

    def exhaust_after_write(target, data, mode):
        write(target, data, mode)
        clock[0] += 11

    monkeypatch.setattr(store, '_write_regular_file', exhaust_after_write)
    with time_budget.run_time_budget(10):
        with pytest.raises(RestoreRecoveryError) as caught:
            store.restore_checkpoint(1)
    assert isinstance(caught.value.__cause__, time_budget.BudgetExhausted)
    assert (task / 'a').read_text() == 'captured a'
    assert (task / 'z').read_text() == 'current z'
    reopened = WorkspaceCheckpointStore(task, shadow_dir=store.shadow_dir)
    assert reopened.recover_incomplete_restore()
    assert (task / 'a').read_text() == 'current a'


def test_restore_does_not_recover_a_live_operation(tmp_path):
    task, store = local_store(tmp_path)
    with (store.shadow_dir / '.restore.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(WorkspaceCheckpointError, match='another workspace restore'):
            store.restore_checkpoint(1)
    assert (task / 'a').read_text() == 'current a'


@pytest.mark.parametrize('replacement', [False, True])
def test_process_exit_leaves_recovery_for_a_fresh_store(tmp_path, replacement):
    task, store = local_store(tmp_path)
    program = '''
import os
from pathlib import Path
import sys
from scripts.llm_solver.harness.workspace_checkpoints import WorkspaceCheckpointStore
store = WorkspaceCheckpointStore(Path(sys.argv[1]), shadow_dir=Path(sys.argv[2]))
write = store._write_regular_file
def interrupt(target, data, mode):
    write(target, data, mode)
    os._exit(86)
store._write_regular_file = interrupt
store.restore_checkpoint(1)
'''
    result = subprocess.run([sys.executable, '-c', program, str(task), str(store.shadow_dir)],
                            capture_output=True, timeout=30)
    assert result.returncode == 86, result.stderr.decode()
    assert (task / 'a').read_text() == 'captured a'
    if replacement:
        task.rename(tmp_path / 'original-task')
        task.mkdir()
        (task / 'a').write_text('replacement task')
        reopened = WorkspaceCheckpointStore(task, shadow_dir=store.shadow_dir)
        with pytest.raises(RestoreRecoveryError):
            reopened.recover_incomplete_restore()
        assert (task / 'a').read_text() == 'replacement task'
        assert (store.shadow_dir / '.restore_recovery/state.json').is_file()
        return
    reopened = WorkspaceCheckpointStore(task, shadow_dir=store.shadow_dir)
    assert reopened.recover_incomplete_restore()
    assert (task / 'a').read_text() == 'current a'
    assert (task / 'z').read_text() == 'current z'


def test_failure_after_index_update_restores_the_private_index(tmp_path, monkeypatch):
    task, store = local_store(tmp_path)
    store._git(['read-tree', '--empty'])
    previous_index = (store.shadow_dir / 'index').read_bytes()
    commit = store.checkpoint_for_turn(1)
    git = store._git

    def fail_after_update(args, **kwargs):
        result = git(args, **kwargs)
        if args == ['read-tree', commit]:
            raise OSError('injected failure after index publication')
        return result

    monkeypatch.setattr(store, '_git', fail_after_update)
    with pytest.raises(OSError, match='after index publication'):
        store.restore_checkpoint(1)
    assert (task / 'a').read_text() == 'current a'
    assert (store.shadow_dir / 'index').read_bytes() == previous_index


def test_corrupt_backup_refuses_recovery_before_other_files_change(tmp_path, monkeypatch):
    task, store = local_store(tmp_path)
    (task / 'extra').write_text('current extra')
    write = store._write_regular_file

    def corrupt_and_fail(target, data, mode):
        write(target, data, mode)
        if target.name == 'a':
            directory = store.shadow_dir / '.restore_recovery'
            record = json.loads((directory / 'state.json').read_text())
            (directory / record['before']['a']['blob']).write_text('damaged backup')
            raise OSError('injected backup damage')

    monkeypatch.setattr(store, '_write_regular_file', corrupt_and_fail)
    with pytest.raises(RestoreRecoveryError):
        store.restore_checkpoint(1)
    assert (task / 'a').read_text() == 'captured a'
    assert not (task / 'extra').exists()
    assert (store.shadow_dir / '.restore_recovery/state.json').is_file()
