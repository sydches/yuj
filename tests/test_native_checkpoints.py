"""Checkpoint capture and restore use native files and private host storage."""
import os
import stat

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.workspace_checkpoints import WorkspaceCheckpointStore


def test_capture_restore_uses_native_bytes_modes_links_and_ignore_rules(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside data')
    (view / 'selected.bin').write_bytes(b'hidden host bytes')
    (view / '.gitignore').write_text('selected.bin\n')
    (source / 'selected.bin').write_bytes(b'native\x00\xff\r\n')
    (source / 'selected.bin').chmod(0o755)
    (source / '.gitignore').write_text('ignored/\n')
    (source / 'ignored').mkdir()
    (source / 'ignored' / 'secret').write_text('ignored')
    (source / 'link').symlink_to('./selected.bin')
    (source / 'outside-link').symlink_to(outside)
    store = WorkspaceCheckpointStore(view, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=view):
        first = store.capture(1)
        entries = store._tree_entries(first.commit)
        assert set(entries) == {'.gitignore', 'selected.bin', 'link', 'outside-link'}
        assert store._git(['cat-file', 'blob', entries['selected.bin'].object_id]).stdout == b'native\x00\xff\r\n'
        assert store._git(['cat-file', 'blob', entries['link'].object_id]).stdout == b'./selected.bin'
        (source / 'selected.bin').write_bytes(b'changed')
        (source / 'selected.bin').chmod(0o644)
        (source / 'new').write_bytes(b'new')
        (source / 'link').unlink()
        restored = store.restore_checkpoint(1)
    assert restored.commit == first.commit
    assert (source / 'selected.bin').read_bytes() == b'native\x00\xff\r\n'
    assert stat.S_IMODE((source / 'selected.bin').stat().st_mode) == 0o755
    assert os.readlink(source / 'link') == './selected.bin'
    assert not (source / 'new').exists()
    assert (source / 'ignored' / 'secret').read_text() == 'ignored'
    assert outside.read_bytes() == b'outside data'
    assert (view / 'selected.bin').read_bytes() == b'hidden host bytes'
    assert not list(store.shadow_dir.glob('.view-*'))


def test_readonly_task_can_be_captured_but_restore_cannot_write_host(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    view = tmp_path / 'view'
    (source / 'file').write_bytes(b'native original')
    (view / 'file').write_bytes(b'hidden host original')
    store = WorkspaceCheckpointStore(view, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=view):
        checkpoint = store.capture(1)
        assert checkpoint.file_count == 1
        assert store.restore_checkpoint(1).commit == checkpoint.commit
        (source / 'file').write_bytes(b'native later')
        with pytest.raises(OSError):
            store.restore_checkpoint(1)
    assert (source / 'file').read_bytes() == b'native later'
    assert (view / 'file').read_bytes() == b'hidden host original'
    assert not list(source.glob('.yuj-restore-*'))


@pytest.mark.parametrize('later_scope', ['none', 'different'])
def test_bound_checkpoint_store_retains_native_reader_after_scope_exit(bwrap, tmp_path, later_scope):
    from contextlib import nullcontext
    from tests._config_helpers import make_config

    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_bytes(b'native original')
    (workspace / 'file').write_bytes(b'hidden host original')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    with activate_task_files(files, host_root=workspace):
        store.bind_task_access(cfg, environment={'PATH': os.environ['PATH']},
                               allow_login_shell=False, ignore_policy=None)

    other_dir = tmp_path / 'other'
    other_dir.mkdir()
    other_source, other_files = namespace_files(bwrap, other_dir)
    (other_source / 'file').write_bytes(b'other task original')
    with (activate_task_files(other_files, host_root=workspace)
          if later_scope == 'different' else nullcontext()):
        checkpoint = store.capture(1)
        entry = store._tree_entries(checkpoint.commit)['file']
        assert store._git(['cat-file', 'blob', entry.object_id]).stdout == b'native original'
        (source / 'file').write_bytes(b'native changed')
        store.restore_checkpoint(1)
    assert (source / 'file').read_bytes() == b'native original'
    assert (workspace / 'file').read_bytes() == b'hidden host original'
    assert (other_source / 'file').read_bytes() == b'other task original'


def test_restore_preserves_unchanged_readonly_nested_mount(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'mounted').write_bytes(b'readonly mounted data')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    view = tmp_path / 'view'
    (source / 'nested' / 'hidden').write_bytes(b'hidden host subtree')
    (source / 'writable').write_bytes(b'original')
    store = WorkspaceCheckpointStore(view, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=view):
        checkpoint = store.capture(1)
        assert set(store._tree_entries(checkpoint.commit)) == {'nested/mounted', 'writable'}
        (source / 'writable').write_bytes(b'changed')
        store.restore_checkpoint(1)
    assert (source / 'writable').read_bytes() == b'original'
    assert (source / 'nested' / 'hidden').read_bytes() == b'hidden host subtree'
    assert (overlay / 'mounted').read_bytes() == b'readonly mounted data'


def test_captured_file_remains_tracked_when_native_ignore_rules_change(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    view = tmp_path / 'view'
    (source / 'kept').write_text('first')
    store = WorkspaceCheckpointStore(view, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=view):
        store.capture(1)
        (source / '.gitignore').write_text('kept\nnew-ignored\n')
        (source / 'kept').write_text('second')
        (source / 'new-ignored').write_text('ignored')
        second = store.capture(2)
    entries = store._tree_entries(second.commit)
    assert set(entries) == {'.gitignore', 'kept'}
    assert store._git(['cat-file', 'blob', entries['kept'].object_id]).stdout == b'second'


def test_configured_checkpoint_access_keeps_private_storage_masked(bwrap, tmp_path):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'file').write_bytes(b'permitted native data')
    store = WorkspaceCheckpointStore(task, shadow_dir=tmp_path / 'shadow')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
                      unreadable_paths=store.sandbox_unreadable_paths)
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    store.bind_task_access(cfg, environment=environment, allow_login_shell=False,
                           ignore_policy=None)
    checkpoint = store.capture(1)
    assert checkpoint.file_count == 1
    assert (store.shadow_dir / 'HEAD').is_file()
    with task_file_scope(str(task), cfg, environment=environment) as files:
        result = files.run('cat -- "$1"', [str(store.shadow_dir / 'HEAD')], None)
    assert result.returncode != 0
    assert b'ref: refs/heads/checkpoints' not in result.stdout
    (task / 'file').write_bytes(b'changed')
    store.restore_checkpoint(1)
    assert (task / 'file').read_bytes() == b'permitted native data'


def test_checkpoint_preserves_task_with_declared_hidden_directory(bwrap, tmp_path):
    from dataclasses import replace
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.workspace_checkpoints import WorkspaceCheckpointError
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'visible').write_text('captured')
    hidden = task / 'private'
    hidden.mkdir()
    (hidden / 'secret').write_text('must remain hidden and unchanged')
    store = WorkspaceCheckpointStore(task, shadow_dir=tmp_path / 'shadow')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap,
                      unreadable_paths=(*store.sandbox_unreadable_paths, str(hidden)))
    store.bind_task_access(cfg, environment={'PATH': '/usr/bin:/bin'},
                           allow_login_shell=False, ignore_policy=None)
    checkpoint = store.capture(1)
    assert set(store._tree_entries(checkpoint.commit)) == {'visible'}
    (task / 'visible').write_text('changed')
    store.restore_checkpoint(1)
    assert (task / 'visible').read_text() == 'captured'
    assert (hidden / 'secret').read_text() == 'must remain hidden and unchanged'
    store.bind_task_access(replace(cfg, unreadable_paths=store.sandbox_unreadable_paths),
                           environment={'PATH': '/usr/bin:/bin'},
                           allow_login_shell=False, ignore_policy=None)
    with pytest.raises(WorkspaceCheckpointError, match='task binding'):
        store.restore_checkpoint(1)
    assert (hidden / 'secret').read_text() == 'must remain hidden and unchanged'
