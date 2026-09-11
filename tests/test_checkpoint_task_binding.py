"""A durable checkpoint must belong to the observed task view before restore."""
from contextlib import nullcontext

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.workspace_checkpoints import (
    WorkspaceCheckpointError, WorkspaceCheckpointStore,
)


@pytest.mark.parametrize('native', [False, True])
def test_reopened_checkpoint_refuses_replacement_root_before_any_write(bwrap, tmp_path, native):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view' if native else source
    (source / 'file').write_text('captured')
    shadow = tmp_path / 'shadow'
    store = WorkspaceCheckpointStore(workspace, shadow_dir=shadow)
    scope = lambda: activate_task_files(files, host_root=workspace) if native else nullcontext()
    with scope():
        original = store.capture(1)
    source.rename(tmp_path / 'original-root')
    source.mkdir()
    (source / 'file').write_text('replacement')
    (source / 'extra').write_text('must not be deleted')
    reopened = WorkspaceCheckpointStore(workspace, shadow_dir=shadow)
    with scope(), pytest.raises(WorkspaceCheckpointError, match='task binding'):
        reopened.restore_checkpoint(1)
    assert (source / 'file').read_text() == 'replacement'
    assert (source / 'extra').read_text() == 'must not be deleted'
    assert reopened.checkpoint_for_turn(1) == original.commit


def test_reopened_native_checkpoint_restores_same_view(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_text('captured')
    shadow = tmp_path / 'shadow'
    with activate_task_files(files, host_root=workspace):
        original = WorkspaceCheckpointStore(workspace, shadow_dir=shadow).capture(1)
    (source / 'file').write_text('changed')
    with activate_task_files(files, host_root=workspace):
        result = WorkspaceCheckpointStore(workspace, shadow_dir=shadow).restore_checkpoint(1)
    assert result.commit == original.commit
    assert (source / 'file').read_text() == 'captured'


def test_legacy_checkpoint_without_binding_is_inspectable_but_not_restored(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_text('captured')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=workspace):
        checkpoint = store.capture(1)
        tree = store._git(['rev-parse', checkpoint.commit + '^{tree}']).stdout.decode().strip()
        legacy = store._git(['commit-tree', tree], input_bytes=b'legacy checkpoint\n').stdout.decode().strip()
        store._git(['update-ref', store._turn_ref(1), legacy])
        (source / 'file').write_text('current')
        assert store.checkpoint_for_turn(1) == legacy
        assert 'file' in store._tree_entries(legacy)
        with pytest.raises(WorkspaceCheckpointError, match='task binding'):
            store.restore_checkpoint(1)
    assert (source / 'file').read_text() == 'current'


def test_capture_refuses_changed_view_without_publishing_ref(bwrap, tmp_path, monkeypatch):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_text('first')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    build = store._build_tree

    def replace_after_read(*args):
        result = build(*args)
        source.rename(tmp_path / 'previous-root')
        source.mkdir()
        (source / 'file').write_text('second')
        return result

    monkeypatch.setattr(store, '_build_tree', replace_after_read)
    with activate_task_files(files, host_root=workspace):
        with pytest.raises(WorkspaceCheckpointError, match='task binding'):
            store.capture(1)
    assert store._current_commit() is None
    assert not store._git(['show-ref'], check=False).stdout
    assert not list(store.shadow_dir.glob('.index-*'))


def test_replacement_nested_mount_refuses_restore_before_other_mutations(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'file').write_text('unchanged bytes')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    workspace = tmp_path / 'view'
    (source / 'writable').write_text('captured')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=workspace):
        store.capture(1)
    overlay.rename(tmp_path / 'previous-overlay')
    overlay.mkdir()
    (overlay / 'file').write_text('unchanged bytes')
    (source / 'writable').write_text('current')
    (source / 'extra').write_text('retain')
    with activate_task_files(files, host_root=workspace):
        with pytest.raises(WorkspaceCheckpointError, match='task binding'):
            store.restore_checkpoint(1)
    assert (source / 'writable').read_text() == 'current'
    assert (source / 'extra').read_text() == 'retain'


def test_native_checkpoint_cannot_be_restored_through_host_access(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_text('native bytes')
    (workspace / 'file').write_text('hidden host bytes')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=workspace):
        store.capture(1)
    with pytest.raises(WorkspaceCheckpointError, match='task binding'):
        store.restore_checkpoint(1)
    assert (workspace / 'file').read_text() == 'hidden host bytes'


def test_changed_executor_descriptor_refuses_restore(bwrap, tmp_path):
    source, files = namespace_files(bwrap, tmp_path)
    workspace = tmp_path / 'view'
    (source / 'file').write_text('captured')
    store = WorkspaceCheckpointStore(workspace, shadow_dir=tmp_path / 'shadow')
    with activate_task_files(files, host_root=workspace):
        store.capture(1)
        (source / 'file').write_text('current')
        files.binding['container_id'] = 'replacement-executor-fixture'
        with pytest.raises(WorkspaceCheckpointError, match='task binding'):
            store.restore_checkpoint(1)
    assert (source / 'file').read_text() == 'current'
