"""A mutable image tag must not change a task's discovered execution image."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import replace
import importlib
import subprocess
from types import SimpleNamespace

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness.container_binding import (
    bind_container_image, container_image_scope, container_scoped_session,
)
from scripts.llm_solver.harness.sandbox.container_backend import (
    ContainerBackend, ContainerBackendError, inspect_container_image_digest,
)
from scripts.llm_solver.harness.task_environment import task_environment_scope
from scripts.llm_solver.harness.task_file_runtime import make_task_files, task_file_scope
from scripts.llm_solver.harness.time_budget import (
    BudgetExhausted, command_time_budget, run_time_budget,
)


FIRST = 'sha256:' + 'a' * 64
SECOND = 'sha256:' + 'b' * 64
IMAGE = 'fixture/task:mutable'


@pytest.fixture
def selected_image(monkeypatch):
    from scripts.llm_solver.harness.sandbox import policy
    monkeypatch.setattr(policy, 'probe_sandbox_capabilities', lambda **kw:
        policy.SandboxCapabilities('linux', ('docker', 'podman'),
            ('docker', 'podman'), {'docker': '/fixture/docker', 'podman': '/fixture/podman'}))
    state = SimpleNamespace(digest=FIRST, inspections=[])
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    monkeypatch.setattr(ContainerBackend, 'resolve_runtime',
                        lambda self, **kw: '/fixture/' + self.runtime)

    def inspect(self, runtime_bin, **kwargs):
        state.inspections.append((runtime_bin, self.image))
        return state.digest

    monkeypatch.setattr(ContainerBackend, 'image_digest', inspect)
    return state


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
def test_all_execution_consumers_share_startup_image(tmp_path, monkeypatch, selected_image, runtime):
    from scripts.llm_solver.harness._loop._driver_setup import compute_runtime_envelope_fields
    from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
    from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
    from scripts.llm_solver.harness._tools.exec_cell import _build_cell_process
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    cfg = make_config(sandbox_bash=True, sandbox_required=True,
                      sandbox_backend='container', sandbox_container_runtime=runtime,
                      sandbox_container_image=IMAGE)
    launched = []

    def execute(argv, **kw):
        launched.append(argv)
        # Run only the trusted file helper in this explicit local fixture.
        # We inspect container argv; no container or model is launched.
        if kw.get('binary'):
            return subprocess.run(argv[argv.index('/bin/bash'):],
                                  capture_output=True, input=kw.get('input_bytes'))
        return 'command output', 0, False

    monkeypatch.setattr(execution, '_execute', execute)
    (tmp_path / 'file').write_bytes(b'literal\x00\xff')
    with container_image_scope():
        fields = compute_runtime_envelope_fields(cfg, tmp_path)
        from scripts.llm_solver.harness.sandbox.policy import bind_sandbox_envelope
        pinned_cfg = bind_sandbox_envelope(cfg, fields)
        selected_image.digest = SECOND  # Same declared tag now names another image.
        with task_file_scope(str(tmp_path), pinned_cfg, environment={}) as files:
            assert files.read_bytes('file') == b'literal\x00\xff'
            files.replace_bytes('file', b'edited', mode=0o600)
            assert files.binding['container_image_digest'] == fields['container_image_digest'] == FIRST
            execution._run_in_sandbox('true', cwd=str(tmp_path), timeout=None,
                                      sandbox=True, bwrap_bin='', sandbox_required=True,
                                      sandbox_backend='container', container_runtime=runtime,
                                      container_image=IMAGE)
            options = dict(cwd=str(tmp_path), bwrap_bin='', sandbox_required=True,
                           sandbox_backend='container', container_runtime=runtime,
                           container_image=IMAGE)
            launched.append(build_background_sandbox_argv('true', **options))
            launched.append(build_lsp_sandbox_argv(('fixture-lsp',), **options))
            launched.append(_build_cell_process(cwd=str(tmp_path), cfg=cfg,
                            unreadable_paths=(), readable_paths=(), effective_env={},
                            allow_login_shell=False)[0])
        assert compute_runtime_envelope_fields(cfg, tmp_path)['container_image_digest'] == FIRST
    assert (tmp_path / 'file').read_bytes() == b'edited'
    assert len(selected_image.inspections) == 1
    assert len(launched) >= 5
    assert all(FIRST in argv and IMAGE not in argv and SECOND not in argv for argv in launched)
    with container_image_scope():
        assert compute_runtime_envelope_fields(cfg, tmp_path)['container_image_digest'] == SECOND


def test_file_executor_retains_binding_after_scope_exit(tmp_path, monkeypatch, selected_image):
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    observed = []
    monkeypatch.setattr(execution, '_execute', lambda argv, **kw: observed.append(argv))
    cfg = make_config(sandbox_backend='container', sandbox_bash=True,
                      sandbox_container_image=IMAGE)
    files = make_task_files(str(tmp_path), cfg, environment={})
    selected_image.digest = SECOND
    files.run('true', [], None)
    assert FIRST in observed[0] and SECOND not in observed[0]
    assert len(selected_image.inspections) == 1


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
def test_lsp_reader_and_later_launcher_retain_one_image(tmp_path, selected_image, runtime):
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec

    spec = LspServerSpec('fixture', ('fixture-lsp',), ('.txt',))
    manager = LspManager.sandboxed(
        cwd=tmp_path, servers=(spec,), bwrap_bin='', sandbox_backend='container',
        container_runtime=runtime, container_image=IMAGE,
    )
    try:
        assert manager._task_files.binding['container_image_digest'] == FIRST
        selected_image.digest = SECOND
        argv = manager.argv_builder(spec, tmp_path)
        assert FIRST in argv and SECOND not in argv and IMAGE not in argv
        assert len(selected_image.inspections) == 1
    finally:
        manager.close()


def test_lsp_launcher_refuses_another_active_image_binding(tmp_path, selected_image):
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec

    spec = LspServerSpec('fixture', ('fixture-lsp',), ('.txt',))
    manager = LspManager.sandboxed(
        cwd=tmp_path, servers=(spec,), bwrap_bin='', sandbox_backend='container',
        container_image=IMAGE,
    )
    try:
        selected_image.digest = SECOND
        with container_image_scope():
            bind_container_image(ContainerBackend(image=IMAGE), '/fixture/docker')
            with pytest.raises(ContainerBackendError, match='active container image binding'):
                manager.argv_builder(spec, tmp_path)
    finally:
        manager.close()


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
@pytest.mark.parametrize('startup_bound', [False, True])
def test_background_launcher_retains_image_across_scope_exit(
    tmp_path, selected_image, runtime, startup_bound,
):
    from scripts.llm_solver.harness.process_manager import ProcessManager

    with container_image_scope() if startup_bound else nullcontext():
        if startup_bound:
            bind_container_image(ContainerBackend(runtime=runtime, image=IMAGE), '/fixture/' + runtime)
        manager = ProcessManager.sandboxed(
            cwd=tmp_path, run_dir=tmp_path / 'records', max_procs=1, poll_timeout_s=1,
            bwrap_bin='', sandbox_backend='container',
            container_runtime=runtime, container_image=IMAGE,
        )
        if not startup_bound:
            assert FIRST in manager.argv_builder('true')
    try:
        selected_image.digest = SECOND
        argv = manager.argv_builder('true')
        assert FIRST in argv and SECOND not in argv and IMAGE not in argv
        assert len(selected_image.inspections) == 1
    finally:
        manager.close()


@pytest.mark.parametrize('change', ['image', 'flags', 'runtime', 'binary', 'connection'])
def test_changed_selection_is_refused(selected_image, monkeypatch, change):
    backend = ContainerBackend(image=IMAGE)
    binary = '/fixture/docker'
    with container_image_scope():
        bind_container_image(backend, binary)
        if change == 'image':
            backend = replace(backend, image='fixture/other:tag')
        elif change == 'flags':
            backend = replace(backend, flags=('--memory', '2g'))
        elif change == 'runtime':
            backend = replace(backend, runtime='podman')
        elif change == 'binary':
            binary = '/another/docker'
        else:
            monkeypatch.setenv('DOCKER_HOST', 'changed-fixture')
        with pytest.raises(ContainerBackendError, match='selection changed'):
            bind_container_image(backend, binary)
    assert len(selected_image.inspections) == 1


def test_parallel_readers_discover_once(selected_image):
    backend = ContainerBackend(image=IMAGE)
    with container_image_scope(), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_context().run, bind_container_image,
                               backend, '/fixture/docker') for _ in range(8)]
        assert all(future.result().image == FIRST for future in futures)
    assert len(selected_image.inspections) == 1


def test_session_construction_run_and_separate_solve_lifetimes(selected_image):
    backend = ContainerBackend(image=IMAGE)

    class FixtureSession:
        @container_scoped_session
        def __init__(self):
            self.initial = bind_container_image(backend, '/fixture/docker').image

        @container_scoped_session
        def run(self):
            return bind_container_image(backend, '/fixture/docker').image

    @task_environment_scope
    def solve():
        first = FixtureSession()
        selected_image.digest = SECOND
        assert FixtureSession().initial == first.initial
        return first.run()

    assert solve() == FIRST
    assert solve() == SECOND
    direct = FixtureSession()
    selected_image.digest = FIRST
    assert direct.run() == SECOND
    assert FixtureSession().initial == FIRST


def test_image_discovery_uses_remaining_budget_and_stops_when_exhausted(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr('scripts.llm_solver.harness.time_budget.time.monotonic', lambda: clock[0])

    def inspect(argv, **kwargs):
        calls.append(kwargs['timeout'])
        return SimpleNamespace(returncode=0, stdout=FIRST, stderr='')

    monkeypatch.setattr('scripts.llm_solver.harness.sandbox.container_backend.subprocess.run', inspect)
    with run_time_budget(8), command_time_budget(5):
        clock[0] += 2
        assert inspect_container_image_digest('/fixture/docker', IMAGE, timeout=20) == FIRST
        clock[0] += 3
        with pytest.raises(BudgetExhausted):
            inspect_container_image_digest('/fixture/docker', IMAGE)
    assert calls == [3]


def test_direct_command_discovery_and_execution_share_one_allowance(tmp_path, monkeypatch):
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    clock = [100.0]
    timeouts = []
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    monkeypatch.setattr('scripts.llm_solver.harness.time_budget.time.monotonic', lambda: clock[0])
    monkeypatch.setattr(ContainerBackend, 'resolve_runtime', lambda self, **kw: '/fixture/docker')

    def inspect(argv, **kwargs):
        timeouts.append(kwargs['timeout'])
        clock[0] += 2
        return SimpleNamespace(returncode=0, stdout=FIRST, stderr='')

    def communicate(**kwargs):
        timeouts.append(kwargs['timeout'])
        return 'ok', ''

    monkeypatch.setattr(execution.subprocess, 'run', inspect)
    monkeypatch.setattr(execution.subprocess, 'Popen', lambda *a, **kw:
                        SimpleNamespace(returncode=0, communicate=communicate))
    result = execution._run_in_sandbox('true', cwd=str(tmp_path), timeout=5,
                sandbox=True, bwrap_bin='', sandbox_backend='container', container_image=IMAGE)
    assert result == ('ok', 0, False)
    assert timeouts == [5, 3]


def test_discovery_failure_is_not_cached(monkeypatch):
    backend = ContainerBackend(image=IMAGE)
    attempts = []

    def inspect(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise ContainerBackendError('fixture unavailable')
        return FIRST

    monkeypatch.setattr(ContainerBackend, 'image_digest', inspect)
    with container_image_scope():
        with pytest.raises(ContainerBackendError, match='fixture unavailable'):
            bind_container_image(backend, '/fixture/docker')
        assert bind_container_image(backend, '/fixture/docker').image == FIRST
    assert len(attempts) == 2


def test_connection_change_during_discovery_is_refused(monkeypatch):
    def inspect(*args, **kwargs):
        monkeypatch.setenv('CONTAINER_CONNECTION', 'another-fixture')
        return FIRST

    monkeypatch.delenv('CONTAINER_CONNECTION', raising=False)
    monkeypatch.setattr(ContainerBackend, 'image_digest', inspect)
    with container_image_scope(), pytest.raises(ContainerBackendError, match='during image discovery'):
        bind_container_image(ContainerBackend(runtime='podman', image=IMAGE), '/fixture/podman')


def test_bound_runtime_cannot_fall_back_to_host(monkeypatch):
    from scripts.llm_solver.harness.sandbox.container_backend import ContainerRuntimeUnavailable
    monkeypatch.setattr(ContainerBackend, 'image_digest', lambda *a, **kw: FIRST)
    monkeypatch.setattr('scripts.llm_solver.harness.sandbox.container_backend.shutil.which',
                        lambda name: None)
    backend = ContainerBackend(image=IMAGE)
    assert backend.resolve_runtime(sandbox_required=False) is None
    with container_image_scope():
        bind_container_image(backend, '/fixture/docker')
        with pytest.raises(ContainerRuntimeUnavailable):
            backend.resolve_runtime(sandbox_required=False)
