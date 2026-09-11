"""A persistent executor owns the environment captured at construction."""
import os

import pytest

from scripts.llm_solver.harness.sandbox import PersistentBashSession
from tests.test_task_files import bwrap


@pytest.mark.parametrize('started', [False, True])
def test_persistent_environment_does_not_follow_caller_mutation(bwrap, tmp_path, started):
    environment = {'PATH': os.environ['PATH'], 'SELECTED_VALUE': 'original'}
    runner = PersistentBashSession(cwd=str(tmp_path), bwrap_bin=bwrap,
                                   sandbox_required=True, effective_env=environment)
    try:
        if started:
            result = runner.run_binary('printf %s "$SELECTED_VALUE"', cwd=str(tmp_path), timeout=20)
            assert result.returncode == 0 and result.stdout == b'original'
            runner.close()
        environment['SELECTED_VALUE'] = 'replacement'
        result = runner.run_binary('printf %s "$SELECTED_VALUE"', cwd=str(tmp_path), timeout=20)
        assert result.returncode == 0 and result.stdout == b'original'
    finally:
        runner.close()


@pytest.mark.parametrize('restart', [False, True])
def test_configured_environment_cannot_change_after_a_started_session(bwrap, tmp_path, restart):
    environment = {'PATH': os.environ['PATH'], 'SELECTED_VALUE': 'original'}
    options = dict(bwrap_bin=bwrap, sandbox_required=True, effective_env=environment,
                   unreadable_paths=(), readable_paths=(), allow_login_shell=False)
    runner = PersistentBashSession(cwd=str(tmp_path), **options)
    try:
        result = runner.run_binary('printf %s "$SELECTED_VALUE"', cwd=str(tmp_path),
                                    timeout=20, execution_options=options)
        assert result.returncode == 0 and result.stdout == b'original'
        if restart:
            runner.close()
        changed = {**options, 'effective_env': {**environment, 'SELECTED_VALUE': 'replacement'}}
        with pytest.raises(RuntimeError, match='runtime mounts or environment changed'):
            runner.run_binary('printf %s "$SELECTED_VALUE"', cwd=str(tmp_path),
                              timeout=20, execution_options=changed)
        result = runner.run_binary('printf %s "$SELECTED_VALUE"', cwd=str(tmp_path),
                                    timeout=20, execution_options=options)
        assert result.returncode == 0 and result.stdout == b'original'
    finally:
        runner.close()
