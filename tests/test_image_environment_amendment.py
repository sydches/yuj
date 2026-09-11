"""Image-source and delivery checks without a container daemon or model."""
import importlib
import subprocess

import pytest
from _config_helpers import make_config
from scripts.llm_solver.harness.container_binding import container_image_scope
from scripts.llm_solver.harness.sandbox.container_backend import ContainerBackend
from scripts.llm_solver.harness.sandbox.env_policy import EnvironmentPolicyError
from scripts.llm_solver.harness.tools import _effective_command_environment

DIGEST = 'sha256:' + 'a' * 64
execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')


@pytest.fixture
def image_runtime(monkeypatch):
    monkeypatch.delenv('YUJ_CONTAINER', raising=False)
    monkeypatch.setenv('HOME', '/host-only/home')
    monkeypatch.setenv('GOPATH', '/host-only/modules')
    monkeypatch.setattr(ContainerBackend, 'resolve_runtime', lambda *a, **k: '/fixture/runtime')
    inspections = []
    def inspect(backend, *args, **kwargs):
        inspections.append(backend.image)
        return DIGEST
    monkeypatch.setattr(ContainerBackend, 'image_digest', inspect)
    return inspections


@pytest.mark.parametrize('inherit', ['core', 'runtime', 'none'])
@pytest.mark.parametrize('runtime', ['docker', 'podman'])
def test_image_source_reaches_command_after_policy(tmp_path, monkeypatch, image_runtime, inherit, runtime):
    tools_dir = tmp_path / 'image-bin'
    tools_dir.mkdir()
    command = tools_dir / 'ordinary-tool'
    command.write_text('#!/bin/sh\nprintf image-tool\n')
    command.chmod(0o700)
    source = {'PATH': str(tools_dir), 'HOME': '/image/home with spaces',
              'LANG': 'image-locale',
              'GOPATH': '/image/modules\nwith newline', 'PRIVATE_TOKEN': 'private',
              'BASH_ENV': str(tmp_path / 'startup')}
    (tmp_path / 'startup').write_text('printf UNEXPECTED_STARTUP\n')
    cfg = make_config(sandbox_bash=True, sandbox_backend='container',
                      sandbox_container_image='ordinary/image:fixture',
                      sandbox_container_runtime=runtime, sandbox_env_inherit=inherit,
                      sandbox_env_set={'LANG': 'declared-locale'})
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        entrypoint = argv[argv.index('--entrypoint') + 1]
        tail = argv[argv.index(DIGEST) + 1:]
        assert '--read-only' in argv and '--pull=never' in argv
        assert argv[argv.index('--network') + 1] == 'none'
        assert argv[argv.index('--workdir') + 1] == str(tmp_path)
        # Replay only the captured entrypoint in a local stand-in environment.
        return subprocess.run([entrypoint, *tail], env=source, cwd=tmp_path,
                              capture_output=True, text=not kwargs.get('binary', False))
    monkeypatch.setattr(execution, '_execute', execute)
    with container_image_scope():
        effective, login = _effective_command_environment(cfg, cwd=tmp_path)
        assert effective['LANG'] == 'declared-locale'
        assert 'PRIVATE_TOKEN' not in effective and 'BASH_ENV' not in effective
        assert ('GOPATH' in effective) == (inherit == 'runtime')
        if inherit == 'none':
            assert calls == [] and effective == {'LANG': 'declared-locale'}
            return
        assert effective['PATH'] == source['PATH'] and effective['HOME'] == source['HOME']
        output, code, timed = execution._run_in_sandbox(
            'ordinary-tool', cwd=str(tmp_path), sandbox=True, bwrap_bin=cfg.bwrap_bin,
            timeout=cfg.bash_timeout, sandbox_required=True, sandbox_backend='container',
            container_runtime=runtime, container_image=cfg.sandbox_container_image,
            effective_env=effective, allow_login_shell=login,
            normalize_output=False, normalize_addresses=False)
    assert output == 'image-tool' and code == 0 and not timed
    assert len(calls) == 2 and len(image_runtime) == 1
    assert calls[0][calls[0].index('--entrypoint') + 1] == '/bin/bash'
    assert calls[1][calls[1].index('--entrypoint') + 1] == '/usr/bin/env'


@pytest.mark.parametrize('payload,code', [(b'PATH=/image\0PATH=/other\0', 0), (b'SECRET_CAPTURE', 1)])
def test_invalid_image_observation_refuses_without_host_fallback(tmp_path, monkeypatch, image_runtime, payload, code):
    monkeypatch.setattr(execution, '_execute', lambda argv, **kw:
                        subprocess.CompletedProcess(argv, code, payload, b'SECRET_CAPTURE'))
    cfg = make_config(sandbox_bash=True, sandbox_backend='container',
                      sandbox_container_image='ordinary/image:fixture')
    with pytest.raises(EnvironmentPolicyError) as error:
        _effective_command_environment(cfg, cwd=tmp_path)
    assert 'SECRET_CAPTURE' not in str(error.value)


@pytest.mark.parametrize('inspection_seconds', [1, 5])
def test_image_probe_uses_remaining_budget(tmp_path, monkeypatch, image_runtime, inspection_seconds):
    from scripts.llm_solver.harness import time_budget
    now = [100.0]
    monkeypatch.setattr(time_budget.time, 'monotonic', lambda: now[0])
    def inspect(*args, **kwargs):
        now[0] += inspection_seconds
        return DIGEST
    monkeypatch.setattr(ContainerBackend, 'image_digest', inspect)
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        assert kwargs['timeout'] == 4
        assert argv[argv.index('--memory') + 1] == '256m'
        return subprocess.CompletedProcess(argv, 0, b'PATH=/image/bin\0', b'')
    monkeypatch.setattr(execution, '_execute', execute)
    cfg = make_config(sandbox_bash=True, sandbox_backend='container',
                      sandbox_container_image='ordinary/image:fixture',
                      sandbox_container_flags=('--memory', '256m'))
    with time_budget.run_time_budget(5):
        if inspection_seconds == 5:
            with pytest.raises(time_budget.BudgetExhausted):
                _effective_command_environment(cfg, cwd=tmp_path)
            assert calls == []
        else:
            effective, _ = _effective_command_environment(cfg, cwd=tmp_path)
            assert effective['PATH'] == '/image/bin'
            assert len(calls) == 1
