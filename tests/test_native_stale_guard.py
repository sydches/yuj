"""The read-before-edit ledger follows the native task view and aliases."""
import hashlib
from pathlib import Path

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.stale_guard import StaleFileGuard, StaleGuardError
from scripts.llm_solver.harness.task_file_runtime import task_file_scope
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize('spelling', ['relative', 'host', 'native'])
def test_dispatched_read_and_edit_share_native_ledger(bwrap, tmp_path, spelling):
    source, files = namespace_files(bwrap, tmp_path)
    native = source / 'source.txt'
    native.write_text('NATIVE_SOURCE\n')
    host_copy = Path(str(files.root)) / 'source.txt'
    host_copy.write_text('HIDDEN_HOST_SOURCE\n')
    # The task's host spelling may differ from its command workdir.
    host_root = tmp_path / 'host_alias'
    host_root.mkdir()
    (host_root / 'source.txt').write_text('ANOTHER_HOST_COPY')
    path = {'relative': 'source.txt', 'host': str(host_root / 'source.txt'),
            'native': str(files.root / 'source.txt')}[spelling]
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, tools_stale_guard_mode='block')
    events = []
    guard = StaleFileGuard(cwd=host_root, mode='block', event_sink=events.append)
    with activate_task_files(files, host_root=host_root), task_file_scope(host_root, cfg):
        result = dispatch('read', {'path': path}, cwd=str(host_root), cfg=cfg, stale_guard=guard)
        assert 'NATIVE_SOURCE' in result and 'HOST' not in result
        assert guard.ledger_snapshot()['source.txt'].sha256 == hashlib.sha256(b'NATIVE_SOURCE\n').hexdigest()
        assert events[-1]['task_view'] == files.binding
        result = dispatch('edit', {'path': path, 'old_str': 'NATIVE_SOURCE', 'new_str': 'EDITED'},
                          cwd=str(host_root), cfg=cfg, stale_guard=guard)
        assert 'stale_file' not in result and native.read_text() == 'EDITED\n'
        assert guard.check_edit(path).reason == 'fresh'
        native.write_text('EXTERNAL_CHANGE\n')
        result = dispatch('edit', {'path': path, 'old_str': 'EXTERNAL_CHANGE', 'new_str': 'WRONG'},
                          cwd=str(host_root), cfg=cfg, stale_guard=guard)
        assert 'stale_file' in result and native.read_text() == 'EXTERNAL_CHANGE\n'
    assert host_copy.read_text() == 'HIDDEN_HOST_SOURCE\n'
    assert (host_root / 'source.txt').read_text() == 'ANOTHER_HOST_COPY'


def test_native_read_only_overlay_is_observed_but_not_made_writable(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'source.txt').write_text('native overlay\n')
    root, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (root / 'nested/source.txt').write_text('hidden writable file\n')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap)
    guard = StaleFileGuard(cwd=root, mode='block')
    with activate_task_files(files, host_root=root), task_file_scope(root, cfg):
        dispatch('read', {'path': 'nested/source.txt'}, cwd=str(root), cfg=cfg, stale_guard=guard)
        assert guard.check_edit('nested/source.txt').allowed
        result = dispatch('edit', {'path': 'nested/source.txt', 'old_str': 'native', 'new_str': 'changed'},
                          cwd=str(root), cfg=cfg, stale_guard=guard)
        assert 'ERROR' in result
    assert (root / 'nested/source.txt').read_text() == 'hidden writable file\n'
    assert (overlay / 'source.txt').read_text() == 'native overlay\n'


def test_resumed_native_ledger_rejects_different_view_and_host_access(bwrap, tmp_path):
    root, files = namespace_files(bwrap, tmp_path)
    (root / 'source.txt').write_text('same bytes')
    events = []
    guard = StaleFileGuard(cwd=root, mode='block', event_sink=events.append)
    with activate_task_files(files, host_root=root):
        guard.observe_read('source.txt')
        resumed = StaleFileGuard.from_trace(cwd=root, mode='block', events=events)
        assert resumed.check_edit('source.txt').reason == 'fresh'
        files.binding = {**files.binding, 'fixture_view': 'replacement'}
        assert resumed.check_edit('source.txt').reason == 'view_changed'
    assert resumed.check_edit('source.txt').reason == 'view_changed'


def test_legacy_host_ledger_cannot_authorize_native_edit(bwrap, tmp_path):
    root, files = namespace_files(bwrap, tmp_path)
    (root / 'source.txt').write_text('same bytes')
    events = []
    guard = StaleFileGuard(cwd=root, mode='block', event_sink=events.append)
    guard.observe_read('source.txt')
    resumed = StaleFileGuard.from_trace(cwd=root, mode='block', events=events)
    with activate_task_files(files, host_root=root):
        assert resumed.check_edit('source.txt').reason == 'view_changed'
        resumed.observe_read('source.txt')
        assert resumed.check_edit('source.txt').reason == 'fresh'


def test_outside_spelling_is_not_rerooted_to_a_task_collision(tmp_path):
    (tmp_path / 'outside').mkdir()
    (tmp_path / 'outside/source.txt').write_text('collision')
    guard = StaleFileGuard(cwd=tmp_path, mode='block')
    with pytest.raises(StaleGuardError):
        guard.observe_read('/outside/source.txt')
    assert not guard.ledger_snapshot()
