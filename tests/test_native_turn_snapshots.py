"""Turn snapshot objects describe the selected task and stay outside it."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver._shared.telemetry_paths import telemetry_dir
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.snapshot_files import RECORDS_NAME
from scripts.llm_solver.harness.turn_snapshots import (
    ensure_snapshot_setup, snapshot, snapshot_object_store, read_map,
)


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          check=True).stdout


def initialize(root, object_format):
    git(root, 'init', '-q', f'--object-format={object_format}')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'initial')


@pytest.mark.parametrize('object_format', ['sha1', 'sha256'])
def test_readonly_native_snapshot_preserves_objects_parent_and_task_git(bwrap, tmp_path, object_format):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    view = tmp_path / 'view'
    (source / 'file').write_bytes(b'native initial')
    initialize(source, object_format)
    (source / 'file').write_bytes(b'native\x00later\xff')
    (source / 'new').write_bytes(b'new native file')
    (view / 'file').write_bytes(b'hidden host file')
    before = {name: (source / '.git' / name).read_bytes() for name in ('index', 'config', 'HEAD')}
    before_log = git(source, 'log', '--all', '--format=%H')
    before_status = git(source, 'status', '--porcelain')
    with activate_task_files(files, host_root=view):
        ensure_snapshot_setup(view)
        sha = snapshot(view, 3, session=SimpleNamespace())
    assert sha
    store = snapshot_object_store(view, sha)
    assert store != view and not store.is_relative_to(view)
    assert git(store, 'rev-parse', f'{sha}^') == git(source, 'rev-parse', 'HEAD')
    assert git(store, 'show', f'{sha}:file') == b'native\x00later\xff'
    assert git(store, 'show', f'{sha}:new') == b'new native file'
    assert git(store, 'fsck', '--no-reflogs') is not None
    assert read_map(view) == [(3, sha)]
    record = json.loads((telemetry_dir(view) / RECORDS_NAME).read_text())
    assert record['storage'] == 'private_git_v1'
    assert record['task_binding'] == files.binding
    assert {name: (source / '.git' / name).read_bytes() for name in before} == before
    assert git(source, 'log', '--all', '--format=%H') == before_log
    assert git(source, 'status', '--porcelain') == before_status
    assert (view / 'file').read_bytes() == b'hidden host file'
    destination = tmp_path / 'restored'
    destination.mkdir()
    subprocess.run(['git', f'--git-dir={store}', f'--work-tree={destination}',
                    'checkout', '-f', sha, '--', '.'], check=True, capture_output=True)
    assert (destination / 'file').read_bytes() == b'native\x00later\xff'


def test_native_snapshot_observes_nested_mount_and_native_git_exclusions(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'visible').write_bytes(b'native mounted file')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    view = tmp_path / 'view'
    (source / 'nested' / 'hidden').write_bytes(b'hidden host subtree')
    (source / 'file').write_bytes(b'ordinary')
    initialize(source, 'sha1')
    (source / '.git' / 'info' / 'exclude').write_text('native-ignored\n')
    (source / 'native-ignored').write_bytes(b'excluded by native Git')
    (source / 'artifacts [one]').mkdir()
    (source / 'artifacts [one]' / 'metrics.json').write_text('ordinary native metrics')
    (view / 'artifacts [one]').mkdir()
    session = SimpleNamespace(_artifact_dir=view / 'artifacts [one]')
    with activate_task_files(files, host_root=view):
        sha = snapshot(view, 1, session=session)
    assert sha
    assert set(git(snapshot_object_store(view, sha), 'ls-tree', '-r', '--name-only', sha).splitlines()) == {
        b'file', b'nested/visible', b'artifacts [one]/metrics.json',
    }


def test_session_snapshot_binds_native_access_and_masks_private_store(bwrap, tmp_path):
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness.task_file_runtime import task_file_scope
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'file').write_bytes(b'permitted file')
    initialize(task, 'sha1')
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, unreadable_paths=())
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    session = SimpleNamespace(cfg=cfg, _effective_env=environment,
                              _allow_login_shell=False, _ignore_policy=None)
    sha = snapshot(task, 1, session=session)
    assert sha
    store = snapshot_object_store(task, sha)
    assert store != task
    assert git(store, 'show', f'{sha}:file') == b'permitted file'
    with task_file_scope(str(task), cfg, environment=environment) as files:
        result = files.run('cat -- "$1"', [str(store / 'HEAD')], None)
    assert result.returncode != 0
    assert b'ref: refs/heads/checkpoints' not in result.stdout


def test_snapshot_keeps_an_overlay_copy_even_when_artifact_bytes_match(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'metrics.json').write_text('same bytes')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested' / 'metrics.json').write_text('same bytes')
    initialize(source, 'sha1')
    session = SimpleNamespace(_artifact_dir=source / 'nested')
    with activate_task_files(files, host_root=source):
        sha = snapshot(source, 1, session=session)
    assert sha
    store = snapshot_object_store(source, sha)
    assert git(store, 'show', sha + ':nested/metrics.json') == b'same bytes'
    record = json.loads((telemetry_dir(source) / RECORDS_NAME).read_text())
    decision = record['artifact_decisions']['nested/metrics.json']
    assert decision['relation'] == 'different_entry' and decision['excluded'] is False
    assert decision['basis'] == 'shared_kernel_entry_metadata'
    assert decision['host'] != decision['native']


@pytest.mark.parametrize('owner_name', ['', 'artifacts [one]'])
@pytest.mark.parametrize('entry_kind', ['file', 'symlink'])
def test_snapshot_excludes_registered_artifact_entries_with_shared_identity(bwrap, tmp_path, owner_name, entry_kind):
    source, files = namespace_files(bwrap, tmp_path)
    (source / 'ordinary').write_text('ordinary file')
    initialize(source, 'sha1')
    owner = source / owner_name
    owner.mkdir(exist_ok=True)
    if entry_kind == 'symlink':
        (owner / 'metrics.json').symlink_to(source / 'ordinary')
    else:
        (owner / 'metrics.json').write_text('owned artifact')
    session = SimpleNamespace(_artifact_dir=owner)
    with activate_task_files(files, host_root=source):
        sha = snapshot(source, 1, session=session)
    assert sha
    store = snapshot_object_store(source, sha)
    assert git(store, 'ls-tree', '-r', '--name-only', sha).splitlines() == [b'ordinary']
    record = json.loads((telemetry_dir(source) / RECORDS_NAME).read_text())
    relative = (owner / 'metrics.json').relative_to(source).as_posix()
    decision = record['artifact_decisions'][relative]
    assert decision['relation'] == 'same_entry' and decision['excluded'] is True
    assert decision['host'] == decision['native']


def test_explicit_local_session_does_not_follow_an_ambient_container_selector(tmp_path, monkeypatch):
    from tests._config_helpers import make_config
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'file').write_bytes(b'local file')
    initialize(task, 'sha1')
    monkeypatch.setenv('YUJ_CONTAINER', 'unselected-container')
    session = SimpleNamespace(cfg=make_config(sandbox_bash=False))
    sha = snapshot(task, 1, session=session)
    assert sha
    assert snapshot_object_store(task, sha) == task
    assert git(task, 'show', f'{sha}:file') == b'local file'


def test_scripted_session_records_the_snapshot_object_store(bwrap, tmp_path):
    from unittest.mock import MagicMock, patch
    from tests._config_helpers import make_config
    from scripts.llm_solver.harness import loop
    from scripts.llm_solver.server.types import TurnResult, ToolCall, Usage
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'file').write_text('before')
    initialize(task, 'sha1')
    artifacts = tmp_path / 'artifacts'
    cfg = make_config(max_turns=2, max_sessions=1, sandbox_bash=True,
                      bwrap_bin=bwrap, turn_snapshots_enabled=True,
                      state_writer_enabled=False, unreadable_paths=())
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult('write', [ToolCall('write-1', 'write', {'path': 'file', 'content': 'after'})],
                   'tool_calls', Usage(10, 2)),
        TurnResult('done', [], 'stop', Usage(12, 2)),
    ]
    client.build_assistant_message.side_effect = lambda content, tool_calls, replay=None: {
        'role': 'assistant', 'content': content,
    }
    with patch.object(loop, '_auto_commit'), patch.object(loop.Session, '_get_server_ctx', return_value=cfg.context_size):
        assert loop.solve_task(task, cfg, client, initial_prompt='Synthetic fixture.', artifacts_dir=artifacts)
    events = [json.loads(line) for line in (artifacts / '.trace.jsonl').read_text().splitlines()]
    event = next(row for row in events if row.get('snapshot_sha'))
    store = snapshot_object_store(task, event['snapshot_sha'])
    assert event['snapshot_object_store'] == str(store)
    assert git(store, 'show', event['snapshot_sha'] + ':file') == b'after'
