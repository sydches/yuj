"""Task paths come from container mounts, and remain consistent across tools."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.llm_solver.harness import task_environment as env
from scripts.llm_solver.harness._tools._common import _resolve
from scripts.llm_solver.harness.sandbox import _build_bwrap_argv
from scripts.llm_solver.harness.bash_write_classification import is_workspace_path, normalize_trace_path


@pytest.fixture(autouse=True)
def task_scope(monkeypatch):
    from scripts.llm_solver.harness.docker_identity import DockerIdentity
    from scripts.llm_solver.harness.process_identity import ProcessIdentity
    monkeypatch.setattr(env, 'observe_docker_identity', lambda **kwargs: DockerIdentity('fixture-engine', 'fixture-context'))
    monkeypatch.setattr(env, 'observe_container_process', lambda *args: ProcessIdentity(
        (1001,) * 4, (1002,) * 4, (1002,), ('0',) * 5))
    token = env._ACTIVE.set(None)
    yield
    env._ACTIVE.reset(token)


def metadata(root, *destinations, workdir):
    return {"mounts": [{"Type": "bind", "Source": str(root), "Destination": d}
                       for d in destinations], "workdir": workdir,
            "id": 'a' * 64, "user": '1001:1002'}


@pytest.mark.parametrize("destination", ["/workspace/project", "/app", "/task with spaces"])
def test_discovered_root_is_shared_by_shell_files_and_trace_paths(tmp_path, monkeypatch, destination):
    monkeypatch.setenv("YUJ_CONTAINER", "task-container")
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata(tmp_path, destination, workdir=destination))))
    monkeypatch.setattr(env.subprocess, "run", inspect)
    discovered = env.discover_task_environment(tmp_path, refresh=True)
    assert discovered.working_directory == destination
    # Mount aliases describe names; without a native file executor they must
    # not turn a container path into permission to read a host copy.
    with pytest.raises(ValueError, match='escapes cwd'):
        _resolve(str(tmp_path), destination + '/src/main.go')
    argv = _build_bwrap_argv("pwd", str(tmp_path))
    assert discovered.container_id == 'a' * 64
    assert discovered.configured_user == '1001:1002'
    assert 'task-container' not in argv
    assert 'a' * 64 in argv
    # The inspected container's default user also retains its group setup.
    assert '--user' not in argv
    assert argv[argv.index("--workdir") + 1] == destination
    assert normalize_trace_path(destination + "/src/main.go") == "src/main.go"
    assert is_workspace_path(destination + "/src/main.go")
    assert not is_workspace_path("/elsewhere/main.go")
    with pytest.raises(ValueError, match="escapes cwd"):
        _resolve(str(tmp_path), destination + "/../answer.go")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes cwd"):
        _resolve(str(tmp_path), destination + "/escape/answer.go")
    assert inspect.call_count == 1


def test_reusing_startup_binding_does_not_repeat_docker_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "task-container")
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata(tmp_path, '/work', workdir='/work'))))
    engine = Mock(wraps=env.observe_docker_identity)
    process = Mock(wraps=env.observe_container_process)
    monkeypatch.setattr(env.subprocess, 'run', inspect)
    monkeypatch.setattr(env, 'observe_docker_identity', engine)
    monkeypatch.setattr(env, 'observe_container_process', process)
    first = env.discover_task_environment(tmp_path, refresh=True)
    with env.use_task_environment(first):
        for _ in range(20):
            assert env.discover_task_environment(tmp_path) is first
            assert first.container_id in _build_bwrap_argv('pwd', str(tmp_path))
    assert inspect.call_count == engine.call_count == process.call_count == 1


def test_multiple_mount_aliases_use_declared_workdir(tmp_path):
    discovered = env.from_container_metadata(tmp_path, metadata(tmp_path, "/app", "/worktree", workdir="/worktree/src"), container="task")
    assert discovered.working_directory == "/worktree"
    assert discovered.relative_path("/app/main.go") == "main.go"
    assert discovered.relative_path("/worktree/main.go") == "main.go"


def test_task_can_be_a_subdirectory_of_a_bind_mount(tmp_path):
    task = tmp_path / "project"
    task.mkdir()
    discovered = env.from_container_metadata(task, metadata(tmp_path, "/workspace", workdir="/workspace"), container="task")
    assert discovered.working_directory == "/workspace/project"


@pytest.mark.parametrize("covered", [False, True])
@pytest.mark.parametrize("alternate", [False, True])
def test_effective_task_mount_supplies_both_command_and_startup_directory(
    tmp_path, monkeypatch, covered, alternate,
):
    from _config_helpers import make_config
    from scripts.llm_solver.harness._loop import _driver_setup

    task = tmp_path / "projects" / "selected task"
    task.mkdir(parents=True)
    foreign = tmp_path / "different repository"
    foreign.mkdir()
    record = metadata(task.parent, "/projects", workdir="/projects/selected task")
    if covered:
        record["mounts"].append({"Type": "bind", "Source": str(foreign), "Destination": "/projects/selected task"})
    if alternate:
        record["mounts"].append({"Type": "bind", "Source": str(task), "Destination": "/valid copy"})
    monkeypatch.setenv("YUJ_CONTAINER", "selected-container")
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(record)))
    monkeypatch.setattr(env.subprocess, "run", inspect)
    if covered and not alternate:
        with pytest.raises(env.TaskEnvironmentUnavailable, match="no bind mount"):
            env.discover_task_environment(task, refresh=True)
        with pytest.raises(env.TaskEnvironmentUnavailable, match="no bind mount"):
            _build_bwrap_argv("pwd", str(task), sandbox_required=True)
        assert all(call.args[0][:2] == ["docker", "inspect"] for call in inspect.call_args_list)
        return
    expected = "/valid copy" if covered else "/projects/selected task"
    observed = env.discover_task_environment(task, refresh=True)
    argv = _build_bwrap_argv("pwd", str(task), sandbox_required=True)
    assert argv[argv.index("--workdir") + 1] == expected
    if covered:
        assert "/projects/selected task" not in observed.aliases
    monkeypatch.setattr(_driver_setup, "collect_provenance", lambda *a, **kw: {})
    # Exercise rendering with the actual retained mapping; native file setup
    # is separately owned by 024 and is not needed for this mapper contrast.
    render = _driver_setup.load_system_prompt_and_provenance.__wrapped__
    prompt, provenance, _, _ = render(
        make_config(analysis_task_format="generic", project_docs_enabled=False),
        SimpleNamespace(profile=SimpleNamespace(preamble="")), task,
        None, None, None, None,
        skill_catalog=SimpleNamespace(format_prompt_block=lambda: "", trace_records=lambda: ()),
    )
    assert prompt.split("Task environment (observed at startup):\n")[1] == "Working directory: " + expected
    assert provenance["task_environment"]["working_directory"] == expected
    assert inspect.call_count == 1


def test_mount_below_task_root_does_not_invalidate_task_alias(tmp_path):
    record = metadata(tmp_path, "/task", workdir="/task")
    record["mounts"].append({"Type": "volume", "Destination": "/task/data"})
    assert env.from_container_metadata(tmp_path, record, container="task").working_directory == "/task"


def test_non_bind_mount_covering_task_root_cannot_supply_host_task(tmp_path):
    task = tmp_path / "group" / "project"
    task.mkdir(parents=True)
    record = metadata(tmp_path, "/workspace", workdir="/workspace/group/project")
    record["mounts"].append({"Type": "volume", "Destination": "/workspace/group"})
    with pytest.raises(env.TaskEnvironmentUnavailable, match="no bind mount"):
        env.from_container_metadata(task, record, container="task")


def test_missing_or_ambiguous_mounts_are_not_guessed(tmp_path):
    with pytest.raises(RuntimeError, match="no bind mount"):
        env.from_container_metadata(tmp_path, {"mounts": [], "workdir": "/testbed"}, container="task")
    with pytest.raises(RuntimeError, match="multiple container mounts"):
        env.from_container_metadata(tmp_path, metadata(tmp_path, "/one", "/two", workdir="/"), container="task")


def test_task_facts_are_cleared_on_failure(tmp_path):
    @env.task_environment_scope
    def fail():
        env._ACTIVE.set(env.TaskEnvironment(str(tmp_path), "/task", ("/task",)))
        raise RuntimeError("test")
    with pytest.raises(RuntimeError):
        fail()
    assert env._ACTIVE.get() is None


def test_missing_container_identity_is_not_replaced_with_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv('YUJ_CONTAINER', 'reusable-name')
    record = metadata(tmp_path, '/work', workdir='/work')
    record.pop('id')
    monkeypatch.setattr(env.subprocess, 'run', Mock(return_value=SimpleNamespace(stdout=json.dumps(record))))
    with pytest.raises(env.TaskEnvironmentUnavailable, match='inspected identity'):
        env.discover_task_environment(tmp_path, refresh=True)


def test_live_solve_refuses_a_changed_container_selection(tmp_path, monkeypatch):
    monkeypatch.setenv('YUJ_CONTAINER', 'first')
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata(tmp_path, '/work', workdir='/work'))))
    monkeypatch.setattr(env.subprocess, 'run', inspect)

    @env.task_environment_scope
    def solve():
        first = env.discover_task_environment(tmp_path)
        monkeypatch.setenv('YUJ_CONTAINER', 'second')
        with pytest.raises(env.TaskEnvironmentUnavailable, match='changed within the solve'):
            env.discover_task_environment(tmp_path)
        return first

    assert solve().container_id == 'a' * 64
    assert inspect.call_count == 1
    assert env._ACTIVE.get() is None


@pytest.mark.parametrize('selector', ['DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG'])
def test_live_binding_refuses_changed_docker_selectors(tmp_path, monkeypatch, selector):
    monkeypatch.setenv('YUJ_CONTAINER', 'selected')
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata(tmp_path, '/work', workdir='/work'))))
    monkeypatch.setattr(env.subprocess, 'run', inspect)
    first = env.discover_task_environment(tmp_path, refresh=True)
    with env.use_task_environment(first):
        monkeypatch.setenv(selector, 'different-value')
        with pytest.raises(env.TaskEnvironmentUnavailable, match='changed within the solve'):
            env.discover_task_environment(tmp_path)
    assert inspect.call_count == 1


def test_reentering_a_binding_keeps_its_original_container_id(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setenv('YUJ_CONTAINER', 'reused-name')
    first = env.from_container_metadata(tmp_path, metadata(tmp_path, '/work', workdir='/work'),
                                        container='reused-name', docker_host=env.os.environ.get('DOCKER_HOST', ''))
    newer = replace(first, container_id='b' * 64)
    env._ACTIVE.set(newer)
    with env.use_task_environment(first):
        command = _build_bwrap_argv('pwd', str(tmp_path))
        assert 'a' * 64 in command
        assert 'b' * 64 not in command
        assert 'reused-name' not in command
    assert env._ACTIVE.get() is newer


@pytest.mark.parametrize('consumer', ['background', 'lsp'])
@pytest.mark.parametrize('later_binding', [False, True])
def test_retained_manager_keeps_discovered_container_after_scope_exit(
        tmp_path, monkeypatch, consumer, later_binding):
    from dataclasses import replace
    from scripts.llm_solver.harness.process_manager import ProcessManager
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec

    monkeypatch.setenv('YUJ_CONTAINER', 'reused-name')
    inspect = Mock(return_value=SimpleNamespace(
        stdout=json.dumps(metadata(tmp_path, '/selected/task', workdir='/selected/task'))))
    monkeypatch.setattr(env.subprocess, 'run', inspect)

    @env.task_environment_scope
    def construct():
        task = env.discover_task_environment(tmp_path)
        options = dict(cwd=tmp_path, bwrap_bin='unused')
        if consumer == 'background':
            manager = ProcessManager.sandboxed(run_dir=tmp_path / 'records', max_procs=1,
                                               poll_timeout_s=1, **options)
            prepare = lambda: manager.argv_builder('true')
        else:
            spec = LspServerSpec('fixture', ('true',), ('.txt',))
            manager = LspManager.sandboxed(servers=(spec,), **options)
            prepare = lambda: manager.argv_builder(spec, tmp_path)
        return task, manager, prepare

    task, manager, prepare = construct()
    try:
        replacement = metadata(tmp_path, '/replacement/task', workdir='/replacement/task')
        replacement['id'] = 'b' * 64
        inspect.return_value = SimpleNamespace(stdout=json.dumps(replacement))
        newer = replace(task, container_id='b' * 64, working_directory='/replacement/task')
        env._ACTIVE.set(newer if later_binding else None)
        argv = prepare()
        assert task.container_id in argv and newer.container_id not in argv
        assert argv[argv.index('--workdir') + 1] == task.working_directory
        assert env._ACTIVE.get() is (newer if later_binding else None)
        assert inspect.call_count == 1
    finally:
        manager.close()


def test_container_discovery_uses_the_remaining_declared_allowance(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.time_budget import run_time_budget
    monkeypatch.setenv('YUJ_CONTAINER', 'selected')
    inspect = Mock(return_value=SimpleNamespace(stdout=json.dumps(metadata(tmp_path, '/work', workdir='/work'))))
    monkeypatch.setattr(env.subprocess, 'run', inspect)
    with run_time_budget(0.2):
        env.discover_task_environment(tmp_path, refresh=True, timeout=3)
    assert 0 < inspect.call_args.kwargs['timeout'] <= 0.2


def test_failed_discovery_refuses_shell_execution(tmp_path, monkeypatch):
    import importlib
    import subprocess
    from scripts.llm_solver.harness._tools._run_in_sandbox import SandboxUnavailableError

    runner = importlib.import_module("scripts.llm_solver.harness._tools._run_in_sandbox")
    monkeypatch.setenv("YUJ_CONTAINER", "missing-task")
    inspect = Mock(side_effect=subprocess.CalledProcessError(1, ["docker", "inspect"]))
    monkeypatch.setattr(env.subprocess, "run", inspect)
    with pytest.raises(SandboxUnavailableError, match="cannot discover"):
        runner._run_in_sandbox("touch should-not-exist", cwd=str(tmp_path), timeout=10,
                               sandbox=True, bwrap_bin="bwrap")
    assert not (tmp_path / "should-not-exist").exists()
    assert all(call.args[0][:2] == ["docker", "inspect"] for call in inspect.call_args_list)


def test_crash_frames_keep_task_paths_and_exclude_dependencies(tmp_path):
    from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event

    env._ACTIVE.set(env.TaskEnvironment(str(tmp_path), "/workspace", ("/workspace",)))
    slot = project_tool_event({
        "tool_name": "bash", "args_summary": "python main.py", "exit_status": 1,
        "result_summary": 'Traceback:\n  File "/usr/lib/dependency.py", line 2\n  File "/workspace/main.py", line 3\n',
    })
    assert slot["traceback_paths"] == "/workspace/main.py"
