"""Offline rewind must use the task executor, including its write permissions."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_assist import runner
from scripts.llm_solver.harness import task_file_runtime
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.turn_snapshots import (
    load_pending_rewind, rewind_snapshot_dir, write_conversation_snapshot,
)
from scripts.llm_solver.harness.workspace_checkpoints import WorkspaceCheckpointStore


@pytest.mark.parametrize('readonly,changed', [(False, True), (True, False), (True, True)])
def test_offline_rewind_obeys_the_selected_view(bwrap, tmp_path, monkeypatch, readonly, changed):
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    source, files = namespace_files(bwrap, tmp_path, readonly=readonly)
    workspace = Path(str(files.root))
    (source / 'file').write_text('native original')
    (workspace / 'file').write_text('hidden host original')
    artifacts = tmp_path / 'records'
    artifacts.mkdir()
    cfg = make_config(sandbox_bash=True, bwrap_bin=bwrap, rewind_enabled=True,
                      tools_file_checkpoints_enabled=True)
    checkpoints = WorkspaceCheckpointStore(
        workspace, shadow_dir=artifacts / '.shadow_git',
        excludes=cfg.tools_file_checkpoints_exclude,
    )
    with activate_task_files(files, host_root=workspace):
        checkpoint = checkpoints.capture(1)
    snapshot_root = rewind_snapshot_dir(workspace, artifacts)
    write_conversation_snapshot(
        snapshot_root, session_number=1, turn_number=1,
        checkpoint_commit=checkpoint.commit, original_prompt='Synthetic task',
        history_messages=[{'role': 'user', 'content': 'Synthetic task'}],
        model_messages=[{'role': 'user', 'content': 'Synthetic task'}],
    )
    trace = artifacts / '.trace.jsonl'
    trace_text = ''.join(json.dumps(event) + '\n' for event in (
        {'event': 'session_start', 'session_number': 1},
        {'event': 'turn_end', 'session_number': 1, 'turn_number': 2},
    ))
    trace.write_text(trace_text)
    record = SimpleNamespace(
        config_paths=(), model='fixture', cwd=str(workspace), artifact_path=artifacts,
        worktree_path=None, worktree_branch=None, worktree_base_commit=None,
        session_id='fixture-session',
    )
    monkeypatch.setattr(runner, 'load_config', lambda **kwargs: cfg)
    calls = []

    def bind(cwd, config, **options):
        from scripts.llm_solver.harness.time_budget import execution_deadline, remaining_before
        assert str(cwd) == str(workspace)
        deadline = execution_deadline()
        assert deadline is not None and 0 < remaining_before(deadline) <= cfg.bash_timeout
        calls.append(options)
        return files

    monkeypatch.setattr(task_file_runtime, 'make_task_files', bind)
    if changed:
        (source / 'file').write_text('native later')
    sessions = Mock()
    if readonly and changed:
        with pytest.raises(OSError):
            runner.rewind_session(sessions, record, turn=1)
        assert (source / 'file').read_text() == 'native later'
        assert trace.read_text() == trace_text
        assert load_pending_rewind(snapshot_root) is None
        sessions.update_session.assert_not_called()
    else:
        event = runner.rewind_session(sessions, record, turn=1)
        assert event['commit'] == checkpoint.commit
        assert (source / 'file').read_text() == 'native original'
        assert load_pending_rewind(snapshot_root)['commit'] == checkpoint.commit
        sessions.update_session.assert_called_once()
    assert calls
    assert (workspace / 'file').read_text() == 'hidden host original'
