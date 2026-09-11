"""Backend responses govern container binding; unchanged names prove nothing."""
import json
import subprocess
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from scripts.llm_solver.harness import docker_identity as identity
from scripts.llm_solver.harness import task_environment as environment


class Backend:
    def __init__(self, root):
        self.root = root
        self.context = {'Name': 'selected', 'Endpoints': {'docker': {'Host': 'unix:///fixture.sock'}}}
        self.engine = 'engine-one'
        self.calls = []
        self.after_inspect = lambda: None

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[1:3] == ['context', 'inspect']:
            value = [self.context]
        elif argv[1] == 'info':
            value = self.engine
        elif argv[1] == 'inspect':
            value = {'mounts': [{'Type': 'bind', 'Source': str(self.root), 'Destination': '/selected/task'}],
                     'workdir': '/selected/task', 'id': 'a' * 64, 'user': '1000:1000'}
            self.after_inspect()
        else:
            raise AssertionError('unexpected backend action: ' + repr(argv))
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), '')


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.process_identity import ProcessIdentity
    selected = Backend(tmp_path)
    monkeypatch.setenv('YUJ_CONTAINER', 'reusable-container-name')
    monkeypatch.setattr(identity.subprocess, 'run', selected.run)
    monkeypatch.setattr(environment, 'observe_container_process', lambda *args: ProcessIdentity(
        (1000,) * 4, (1000,) * 4, (1000,), ('0',) * 5))
    token = environment._ACTIVE.set(None)
    yield selected
    environment._ACTIVE.reset(token)


def test_binding_records_engine_identity_and_a_private_context_digest(backend):
    backend.context['Metadata'] = {'private_value': 'do-not-record-this'}
    bound = environment.discover_task_environment(backend.root)
    assert bound.docker_engine_id == 'engine-one'
    assert len(bound.docker_context_fingerprint) == 64
    assert 'do-not-record-this' not in json.dumps(asdict(bound))
    assert [argv[1] for argv, _ in backend.calls] == ['context', 'info', 'inspect', 'context', 'info', 'context', 'info']


@pytest.mark.parametrize('change', ['engine', 'context_endpoint'])
def test_changed_backend_refuses_execution_with_unchanged_container_name(backend, monkeypatch, change):
    import importlib
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    environment.discover_task_environment(backend.root)
    if change == 'engine':
        backend.engine = 'engine-two'
    else:
        backend.context['Endpoints']['docker']['Host'] = 'unix:///replacement.sock'
    execute = Mock(side_effect=AssertionError('task command must not execute'))
    monkeypatch.setattr(execution, '_execute', execute)
    with pytest.raises(execution.SandboxUnavailableError, match='backend changed'):
        execution._run_in_sandbox('touch unwanted', cwd=str(backend.root), timeout=1,
                                 sandbox=True, bwrap_bin='unused')
    execute.assert_not_called()
    assert not (backend.root / 'unwanted').exists()


def test_backend_change_during_container_inspection_is_not_bound(backend):
    backend.after_inspect = lambda: setattr(backend, 'engine', 'engine-two')
    with pytest.raises(environment.TaskEnvironmentUnavailable, match='backend changed'):
        environment.discover_task_environment(backend.root)
    assert environment._ACTIVE.get() is None


@pytest.mark.parametrize('consumer', ['background', 'lsp'])
@pytest.mark.parametrize('change', ['engine', 'context_endpoint'])
def test_retained_manager_checks_its_original_backend(backend, consumer, change):
    from scripts.llm_solver.harness.process_manager import ProcessManager
    from scripts.llm_solver.harness.lsp_support import LspManager, LspServerSpec

    options = dict(cwd=backend.root, bwrap_bin='unused')
    if consumer == 'background':
        manager = ProcessManager.sandboxed(run_dir=backend.root / 'records', max_procs=1,
                                           poll_timeout_s=1, **options)
        prepare = lambda: manager.argv_builder('true')
    else:
        spec = LspServerSpec('fixture', ('true',), ('.txt',))
        manager = LspManager.sandboxed(servers=(spec,), **options)
        prepare = lambda: manager.argv_builder(spec, backend.root)
    try:
        environment._ACTIVE.set(None)
        if change == 'engine':
            backend.engine = 'engine-two'
        else:
            backend.context['Endpoints']['docker']['Host'] = 'unix:///replacement.sock'
        with pytest.raises(environment.TaskEnvironmentUnavailable, match='backend changed'):
            prepare()
        assert sum(argv[1] == 'inspect' for argv, _ in backend.calls) == 1
    finally:
        manager.close()


@pytest.mark.parametrize('value', ['', None, True, {}])
def test_missing_engine_identity_is_not_inferred_from_context_or_container(backend, value):
    backend.engine = value
    with pytest.raises(environment.TaskEnvironmentUnavailable, match='cannot discover'):
        environment.discover_task_environment(backend.root)
    assert all(argv[1] != 'inspect' for argv, _ in backend.calls)


def test_backend_queries_share_the_declared_allowance(tmp_path, monkeypatch):
    from scripts.llm_solver.harness import time_budget
    backend = Backend(tmp_path)
    now = [0.0]
    monkeypatch.setattr(time_budget.time, 'monotonic', lambda: now[0])
    def query(argv, **kwargs):
        result = backend.run(argv, **kwargs)
        now[0] += 1
        return result
    monkeypatch.setattr(identity.subprocess, 'run', query)
    with time_budget.run_time_budget(5):
        identity.observe_docker_identity()
    assert [kwargs['timeout'] for _, kwargs in backend.calls] == [5, 4]
