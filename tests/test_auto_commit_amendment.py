"""Optional commits must use selected bytes and the declared time allowance."""
import subprocess

import pytest
from test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness._loop import session_io
from scripts.llm_solver.harness.task_path import activate_task_files
from scripts.llm_solver.harness.time_budget import run_time_budget


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def initialize(root):
    git(root, 'init')
    git(root, 'add', '-A')
    git(root, '-c', 'user.name=fixture', '-c', 'user.email=fixture@local',
        'commit', '--allow-empty', '-m', 'base')


def test_optional_commit_uses_native_overlay_not_host_dirt(bwrap, tmp_path):
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'file').write_text('base\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/file').write_text('base\n')
    initialize(source)
    (overlay / 'file').write_text('selected work\n')
    with activate_task_files(files, host_root=source):
        session_io._auto_commit(source, 1, 'stop', enabled=True)
        assert git(source, 'show', 'HEAD:nested/file') == b'selected work\n'
        head = git(source, 'rev-parse', 'HEAD')
        (source / 'nested/file').write_text('hidden work\n')
        session_io._auto_commit(source, 2, 'stop', enabled=True)
        assert git(source, 'rev-parse', 'HEAD') == head
    assert (overlay / 'file').read_text() == 'selected work\n'
    assert (source / 'nested/file').read_text() == 'hidden work\n'


@pytest.mark.parametrize('expire_after', [0, 1, 2])
def test_each_git_launch_obeys_remaining_allowance(tmp_path, monkeypatch, caplog, expire_after):
    from scripts.llm_solver.harness import time_budget
    (tmp_path / '.git').mkdir()
    now = [100.0]
    monkeypatch.setattr(time_budget.time, 'monotonic', lambda: now[0])
    calls = []
    def command(args, **kwargs):
        assert kwargs['timeout'] == 10 - len(calls)
        calls.append(args)
        now[0] = 110.0 if len(calls) == expire_after else now[0] + 1
        return subprocess.CompletedProcess(args, 0, ' M file\n', '')
    monkeypatch.setattr(session_io.subprocess, 'run', command)
    with run_time_budget(10):
        if expire_after == 0:
            now[0] = 110.0
        session_io._auto_commit(tmp_path, 1, 'stop', enabled=True)
    assert len(calls) == expire_after
    assert 'auto_commit_failed' in caplog.text


def test_failed_hook_preserves_work_and_reports_nonatomic_staging(tmp_path, caplog):
    initialize(tmp_path)
    head = git(tmp_path, 'rev-parse', 'HEAD')
    (tmp_path / 'work').write_text('completed work')
    hook = tmp_path / '.git/hooks/pre-commit'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o700)
    session_io._auto_commit(tmp_path, 1, 'error', enabled=True)
    assert git(tmp_path, 'rev-parse', 'HEAD') == head
    assert git(tmp_path, 'diff', '--cached', '--name-only') == b'work\n'
    assert (tmp_path / 'work').read_text() == 'completed work'
    assert 'auto_commit_failed' in caplog.text


@pytest.mark.parametrize('finish', ['done', 'error', 'hook_block', 'stop', 'signal'])
def test_driver_commits_through_retained_session_view(bwrap, tmp_path, monkeypatch, finish):
    from unittest.mock import MagicMock
    from _config_helpers import make_config
    from scripts.llm_solver.harness.loop import Session, SessionResult, solve_task
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / 'file').write_text('base\n')
    source, files = namespace_files(bwrap, tmp_path, overlay=overlay)
    (source / 'nested/file').write_text('base\n')
    initialize(source)
    def session_run(session):
        session._task_file_scope = lambda: activate_task_files(files, host_root=source)
        (overlay / 'file').write_text('session work\n')
        if finish == 'signal':
            raise SystemExit(143)
        return SessionResult(turns=1, finish_reason=finish, done=finish == 'done')
    monkeypatch.setattr(Session, 'run', session_run)
    cfg = make_config(sandbox_bash=False, auto_commit=True, max_sessions=1)
    if finish == 'signal':
        with pytest.raises(SystemExit):
            solve_task(source, cfg, MagicMock(), initial_prompt='fixture task')
    else:
        solve_task(source, cfg, MagicMock(), initial_prompt='fixture task')
    assert git(source, 'show', 'HEAD:nested/file') == b'session work\n'
    assert (source / 'nested/file').read_text() == 'base\n'


def test_native_refusal_never_commits_host_copy(bwrap, tmp_path, caplog):
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    initialize(source)
    head = git(source, 'rev-parse', 'HEAD')
    (source / 'work').write_text('preserve work')
    with activate_task_files(files, host_root=source):
        session_io._auto_commit(source, 1, 'stop', enabled=True)
    assert git(source, 'rev-parse', 'HEAD') == head
    assert (source / 'work').read_text() == 'preserve work'
    assert 'auto_commit_failed' in caplog.text


def test_timeout_is_reported_without_discarding_work(tmp_path, monkeypatch, caplog):
    (tmp_path / '.git').mkdir()
    (tmp_path / 'work').write_text('preserve work')
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(session_io.subprocess, 'run', timeout)
    with run_time_budget(10):
        session_io._auto_commit(tmp_path, 1, 'stop', enabled=True)
    assert 'auto_commit_failed' in caplog.text
    assert (tmp_path / 'work').read_text() == 'preserve work'
