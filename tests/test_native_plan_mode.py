"""Plan admission and completion inspect the plan in the command filesystem."""
from pathlib import Path

import pytest

from tests._config_helpers import make_config
from tests.test_task_files import bwrap, namespace_files
from scripts.llm_solver.harness.plan_mode import PLAN_FILE, PlanModeController, is_exact_plan_path
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from scripts.llm_solver.harness.task_path import activate_task_files


@pytest.mark.parametrize('native_state', ['valid', 'empty', 'missing', 'denied', 'hidden'])
def test_exit_uses_native_plan_and_visibility(bwrap, tmp_path, monkeypatch, native_state):
    from scripts.llm_solver.harness import task_file_runtime
    source, files = namespace_files(bwrap, tmp_path, readonly=True)
    target = Path(str(files.root))
    (target / '.solver').mkdir()
    (target / PLAN_FILE).write_text('HIDDEN_HOST_PLAN')
    (source / '.solver').mkdir()
    body = 'Visible plan 文é\n'
    if native_state != 'missing':
        (source / PLAN_FILE).write_text('' if native_state == 'empty' else body)
    if native_state == 'denied':
        (source / PLAN_FILE).chmod(0)
    if native_state == 'hidden':
        (source / '.yujignore').write_text(PLAN_FILE + '\n')
    with activate_task_files(files, host_root=target):
        policy = load_ignore_policy(target)
    environments = []

    def bind(cwd, cfg, **kw):
        environments.append(kw['environment'])
        return files

    monkeypatch.setattr(task_file_runtime, 'make_task_files', bind)
    events = []
    environment = {'PATH': '/fixture/selected'}
    controller = PlanModeController(
        cwd=str(target), cfg=make_config(plan_mode='required', sandbox_bash=True),
        events=(), event_sink=events.append, effective_env=environment,
        allow_login_shell=False, ignore_policy=policy,
    )
    result = controller.exit(turn=2)
    if native_state == 'valid':
        assert not controller.active
        assert events[0]['plan_chars'] == len(body)
    else:
        assert controller.active and not events
        assert 'Cannot exit plan mode' in result
    assert environments and all(env is environment for env in environments)


@pytest.mark.parametrize('redirect', ['directory', 'file', 'host_only'])
def test_plan_write_checks_native_symlinks_and_path_aliases(bwrap, tmp_path, redirect):
    source, files = namespace_files(bwrap, tmp_path)
    target = Path(str(files.root))
    outside = tmp_path / 'outside'
    outside.mkdir()
    if redirect == 'directory':
        (source / '.solver').symlink_to(outside, target_is_directory=True)
    else:
        (source / '.solver').mkdir()
        if redirect == 'file':
            (source / PLAN_FILE).symlink_to(outside / 'other.md')
        else:
            (target / '.solver').symlink_to(outside, target_is_directory=True)
    with activate_task_files(files, host_root=source):
        for spelling in (PLAN_FILE, str(source / PLAN_FILE), str(target / PLAN_FILE)):
            assert is_exact_plan_path(source, spelling) is (redirect == 'host_only')
        assert not is_exact_plan_path(source, '../.solver/plan.md')
        assert not is_exact_plan_path(source, '.solver/other.md')
