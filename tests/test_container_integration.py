"""Integration coverage for the first-class container sandbox selector."""
from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver._resource_contract import PACKAGE_RUNTIME_FILES
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._driver_setup import (
    compute_runtime_envelope_fields,
)
from scripts.llm_solver.harness._loop.persistent_bash import (
    maybe_install_persistent_bash,
)
from scripts.llm_solver.harness._tools._run_in_sandbox import _run_in_sandbox
from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
from scripts.llm_solver.harness.sandbox.container_backend import ContainerBackend


IMAGE = "local/yuj-fixture:latest"
DIGEST = "sha256:" + ("d" * 64)
MISSING_BWRAP = "/not/installed/bwrap"


def test_container_config_defaults_overlay_and_validation(tmp_path: Path) -> None:
    defaults = load_config()
    assert defaults.sandbox_backend == "bwrap"
    assert defaults.sandbox_container_runtime == "docker"
    assert defaults.sandbox_container_image == ""
    assert defaults.sandbox_container_flags == ()

    overlay = tmp_path / "container.toml"
    overlay.write_text(
        """
[sandbox]
backend = "container"
container_runtime = "podman"
container_image = "local/task:sealed"
container_flags = ["--memory", "2g", "--pids-limit=64"]
""".strip()
    )
    configured = load_config(user_config=overlay)
    # The legacy shared-container spelling migrates to the exact named
    # runtime so provenance cannot hide Docker versus Podman.
    assert configured.sandbox_backend == "podman"
    assert configured.sandbox_container_runtime == "podman"
    assert configured.sandbox_container_image == "local/task:sealed"
    assert configured.sandbox_container_flags == (
        "--memory", "2g", "--pids-limit=64",
    )

    invalid_values = [
        'backend = "namespace"\ncontainer_image = "local/task"',
        'backend = "container"\ncontainer_runtime = "nerdctl"\n'
        'container_image = "local/task"',
        'backend = "container"\ncontainer_image = ""',
        'backend = "container"\ncontainer_image = "local/task"\n'
        'container_flags = ["--network", "host"]',
    ]
    for value in invalid_values:
        overlay.write_text("[sandbox]\n" + value + "\n")
        with pytest.raises(ValueError):
            load_config(user_config=overlay)


def test_run_in_sandbox_executes_container_argv_without_shell(
    monkeypatch, tmp_path: Path,
) -> None:
    runner_module = importlib.import_module(
        "scripts.llm_solver.harness._tools._run_in_sandbox"
    )
    calls = []

    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:3] == ['image', 'inspect']:
            return SimpleNamespace(returncode=0, stdout=DIGEST, stderr='')
        return SimpleNamespace(returncode=0, stdout="container-ok\n", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0,
                               communicate=lambda **kw: ('container-ok\n', ''))

    monkeypatch.setattr(runner_module.subprocess, 'Popen', fake_popen)
    out, exit_code, timed_out = _run_in_sandbox(
        "pwd", cwd=str(tmp_path), timeout=10, sandbox=True,
        bwrap_bin=MISSING_BWRAP, sandbox_required=True,
        sandbox_backend="container", container_runtime="docker",
        container_image=IMAGE, container_flags=("--memory", "1g"),
    )

    assert calls[0][0][1:3] == ['image', 'inspect']
    argv, kwargs = calls[1]
    assert DIGEST in argv and IMAGE not in argv
    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert "--network" in argv and "none" in argv
    assert str(tmp_path) in argv
    assert kwargs.get("shell") is not True
    assert kwargs["cwd"] is None
    assert (out, exit_code, timed_out) == ("container-ok\n", 0, False)


def test_missing_selected_container_runtime_fails_closed_for_both_required_settings(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.shutil.which",
        lambda _name: None,
    )

    strict = _run_in_sandbox(
        "printf strict", cwd=str(tmp_path), timeout=10, sandbox=True,
        bwrap_bin=MISSING_BWRAP, sandbox_required=True,
        sandbox_backend="container", container_image=IMAGE,
    )
    assert strict[1:] == (None, False)
    assert "runtime 'docker' is missing" in strict[0]

    optional = _run_in_sandbox(
        "touch must-not-run", cwd=str(tmp_path), timeout=10, sandbox=True,
        bwrap_bin=MISSING_BWRAP, sandbox_required=False,
        sandbox_backend="container", container_image=IMAGE,
    )
    assert optional[1:] == (None, False)
    assert "runtime 'docker' is missing" in optional[0]
    assert not (tmp_path / 'must-not-run').exists()


def test_first_class_container_rejects_legacy_container_mode(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")

    out, exit_code, timed_out = _run_in_sandbox(
        "true", cwd=str(tmp_path), timeout=10, sandbox=True,
        bwrap_bin=MISSING_BWRAP, sandbox_required=True,
        sandbox_backend="container", container_image=IMAGE,
    )

    assert exit_code is None and timed_out is False
    assert "cannot be combined" in out


def test_bash_container_backend_disables_host_trivial_read_fast_path(
    monkeypatch, tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness import tools as tools_module
    from scripts.llm_solver.harness._tools.bash import bash

    (tmp_path / "value.txt").write_text("host-value\n")
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return "container-value\n", 0, False

    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(tools_module, "_run_in_sandbox", fake_runner)
    result = bash(
        "cat value.txt", cwd=str(tmp_path), timeout=10, sandbox=True,
        sandbox_backend="container", container_image=IMAGE,
    )

    assert result == "container-value\n"
    assert calls[0][1]["sandbox_backend"] == "container"


def test_run_tests_threads_container_settings(monkeypatch, tmp_path: Path) -> None:
    from scripts.llm_solver.harness import tools as tools_module
    from scripts.llm_solver.harness._tools.run_tests import run_tests

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return "1 passed\n", 0, False

    monkeypatch.setattr(tools_module, "_run_in_sandbox", fake_runner)
    result = run_tests(
        cwd=str(tmp_path),
        cfg=make_config(
            tools_run_tests_enabled=True,
            analysis_task_format="pytest",
            sandbox_backend="container",
            sandbox_container_runtime="podman",
            sandbox_container_image=IMAGE,
            sandbox_container_flags=("--memory", "1g"),
        ),
    )

    assert 'status="passed"' in result
    assert calls[0][1]["sandbox_backend"] == "container"
    assert calls[0][1]["container_runtime"] == "podman"
    assert calls[0][1]["container_image"] == IMAGE
    assert calls[0][1]["container_flags"] == ("--memory", "1g")


def test_post_edit_check_threads_container_settings(
    monkeypatch, tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness import post_edit

    calls = []

    def fake_bash(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return "ok"

    monkeypatch.setattr("scripts.llm_solver.harness.tools.bash", fake_bash)
    cfg = make_config(
        post_edit_check_enabled=True,
        post_edit_checks=[{
            "name": "syntax", "trigger": "write", "when": "",
            "cmd": "check {path}", "on_fail": "block",
        }],
        sandbox_backend="container",
        sandbox_container_runtime="docker",
        sandbox_container_image=IMAGE,
        sandbox_container_flags=("--memory", "1g"),
    )
    result = post_edit.run_post_edit_checks(
        "src.py", cwd=str(tmp_path), cfg=cfg, trigger="write",
    )

    assert result.action == "ok"
    assert calls[0][1]["sandbox_backend"] == "container"
    assert calls[0][1]["container_image"] == IMAGE


def test_container_backend_disables_persistent_bwrap(tmp_path: Path) -> None:
    session = SimpleNamespace(
        cwd=str(tmp_path),
        cfg=make_config(
            sandbox_bash=True,
            sandbox_backend="container",
            sandbox_container_image=IMAGE,
        ),
    )
    assert maybe_install_persistent_bash(session) is None


def test_named_container_session_owns_one_lazy_connection(monkeypatch, tmp_path):
    from scripts.llm_solver.harness._loop import persistent_bash as lifecycle
    from scripts.llm_solver.harness.process_identity import ProcessIdentity
    from scripts.llm_solver.harness.task_environment import TaskEnvironment

    environment = TaskEnvironment(str(tmp_path), '/workspace', ('/workspace',),
                                  container='selected', container_id='immutable-id',
                                  process_identity=ProcessIdentity((1000,) * 4, (1000,) * 4,
                                                                   (1000,), ('0' * 16,) * 5))
    installed = []
    monkeypatch.setenv('YUJ_PERSISTENT_BASH', '1')
    monkeypatch.setattr(lifecycle, 'container_mode', lambda: 'selected')
    monkeypatch.setattr(lifecycle, 'get_persistent_runner', lambda: None)
    monkeypatch.setattr(lifecycle, 'set_persistent_runner', installed.append)
    monkeypatch.setattr('scripts.llm_solver.harness.task_environment.discover_task_environment',
                        lambda cwd: environment)
    monkeypatch.setattr(subprocess, 'Popen',
                        lambda *a, **kw: pytest.fail('connection must start lazily'))
    session = SimpleNamespace(cwd=str(tmp_path), _effective_env={'PATH': '/usr/bin:/bin'},
                              cfg=make_config(sandbox_bash=True, bwrap_bin=MISSING_BWRAP))
    runner = lifecycle.maybe_install_persistent_bash(session)
    assert runner is not None and runner.supports(environment, str(tmp_path))
    assert installed == [runner]
    closed = []
    monkeypatch.setattr(runner, 'close', lambda: closed.append(True))
    lifecycle.teardown_persistent_bash(runner)
    assert installed == [runner, None] and closed == [True]


@pytest.mark.parametrize('mode', ['ambient', 'nested', 'disabled'])
def test_persistent_container_admission_preserves_other_session_modes(monkeypatch, tmp_path, mode):
    from scripts.llm_solver.harness._loop import persistent_bash as lifecycle

    monkeypatch.setattr(lifecycle, 'container_mode',
                        lambda: lifecycle.AMBIENT_CONTAINER if mode == 'ambient' else 'selected')
    monkeypatch.setattr(lifecycle, 'get_persistent_runner',
                        lambda: object() if mode == 'nested' else None)
    monkeypatch.setenv('YUJ_PERSISTENT_BASH', '0' if mode == 'disabled' else '1')
    monkeypatch.setattr(lifecycle, 'PersistentContainerSession',
                        lambda *a, **kw: pytest.fail('ineligible session constructed a runner'))
    session = SimpleNamespace(cwd=str(tmp_path),
                              cfg=make_config(sandbox_bash=True, bwrap_bin=MISSING_BWRAP))
    assert lifecycle.maybe_install_persistent_bash(session) is None


def test_bwrap_call_cannot_borrow_a_container_connection(monkeypatch, tmp_path):
    runner_module = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    binary = tmp_path / 'bwrap'
    binary.touch()

    class ContainerRunner:
        cwd = str(tmp_path)

    calls = []
    monkeypatch.setattr(runner_module, 'container_mode', lambda: None)
    monkeypatch.setattr(runner_module, 'PersistentContainerSession', ContainerRunner)
    monkeypatch.setattr(runner_module, 'get_persistent_runner', lambda: ContainerRunner())
    monkeypatch.setattr(runner_module, '_build_bwrap_argv', lambda *a, **kw: ['own-bwrap'])

    def execute(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, 'own-view', '')

    monkeypatch.setattr(runner_module, '_execute', execute)
    assert _run_in_sandbox(':', cwd=str(tmp_path), timeout=3, sandbox=True,
                           bwrap_bin=str(binary)) == ('own-view', 0, False)
    assert calls == [['own-bwrap']]


@pytest.mark.parametrize('scenario', ['reuse', 'other_binding', 'opt_out', 'unsupported', 'failed', 'timeout'])
def test_named_container_dispatch_reuses_only_admitted_calls(monkeypatch, tmp_path, scenario):
    runner_module = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    from scripts.llm_solver.harness import time_budget
    from scripts.llm_solver.harness.task_environment import TaskEnvironment

    environment = TaskEnvironment(str(tmp_path), '/workspace', ('/workspace',),
                                  container='selected', container_id='immutable-id')
    argv = ['docker', 'exec', '--workdir', '/workspace', 'immutable-id', 'prepared-command']
    result = subprocess.CompletedProcess(argv, 0, b'raw\0output', b'diagnostic\r\n')
    retained, original = [], []
    clock = [10.0]
    monkeypatch.setattr(time_budget.time, 'monotonic', lambda: clock[0])

    class Runner:
        def supports(self, selected, cwd):
            assert selected is environment and cwd == str(tmp_path)
            return scenario != 'other_binding'

        def execute(self, prepared, **kwargs):
            retained.append((prepared, kwargs))
            if scenario == 'unsupported':
                clock[0] += 2
                raise runner_module.UnsupportedPersistentContainer('missing optional transport utility')
            if scenario == 'failed':
                raise RuntimeError('connection ended after command started')
            if scenario == 'timeout':
                raise subprocess.TimeoutExpired(prepared, kwargs['timeout'])
            return result

    def one_shot(prepared, **kwargs):
        if scenario == 'unsupported':
            assert time_budget.command_timeout(kwargs['timeout']) == 1
        original.append((prepared, kwargs))
        return result

    monkeypatch.setattr(runner_module, 'container_mode', lambda: 'selected')
    monkeypatch.setattr(runner_module, 'PersistentContainerSession', Runner)
    monkeypatch.setattr(runner_module, 'get_persistent_runner', lambda: Runner())
    monkeypatch.setattr(runner_module, '_build_bwrap_argv', lambda *a, **kw: list(argv))
    monkeypatch.setattr(runner_module, '_execute', one_shot)
    monkeypatch.setattr('scripts.llm_solver.harness.task_environment.discover_task_environment',
                        lambda *a, **kw: environment)
    options = dict(cwd=str(tmp_path), timeout=3, sandbox=True, bwrap_bin=MISSING_BWRAP,
                   raw_result=True, input_bytes=b'input\0bytes', use_persistent=scenario != 'opt_out')
    if scenario in {'failed', 'timeout'}:
        error = RuntimeError if scenario == 'failed' else subprocess.TimeoutExpired
        with pytest.raises(error):
            _run_in_sandbox('task-command', **options)
        assert len(retained) == 1 and original == []
    else:
        assert _run_in_sandbox('task-command', **options) is result
        assert len(retained) == (0 if scenario in {'other_binding', 'opt_out'} else 1)
        assert len(original) == (0 if scenario == 'reuse' else 1)
    for prepared, kwargs in retained + original:
        assert prepared == [*argv[:2], '-i', *argv[2:]]
        assert kwargs == {'timeout': 3, 'binary': True, 'input_bytes': b'input\0bytes'}


def test_container_runtime_envelope_inspects_local_image_once(
    monkeypatch, tmp_path: Path,
) -> None:
    calls = []
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(
        ContainerBackend,
        "resolve_runtime",
        lambda self, *, sandbox_required: "/usr/bin/docker",
    )

    def image_digest(self, runtime_bin, *, timeout=None):
        calls.append((runtime_bin, timeout))
        return DIGEST

    monkeypatch.setattr(ContainerBackend, "image_digest", image_digest)
    fields = compute_runtime_envelope_fields(
        make_config(
            sandbox_bash=True,
            sandbox_required=True,
            sandbox_backend="container",
            sandbox_container_runtime="docker",
            sandbox_container_image=IMAGE,
        ),
        tmp_path,
    )

    assert calls == [("/usr/bin/docker", None)]
    assert fields["sandbox_mode"] == "container"
    assert fields["sandbox_engaged"] is True
    assert fields["sandbox_backend"] == "container"
    assert fields["container_runtime"] == "docker"
    assert fields["container_image_digest"] == DIGEST
    assert fields["container_preflight_error"] is None
    assert fields["bwrap_preflight_passed"] is None
    assert fields["sandbox_policy_version"] == 3
    assert tuple(fields["quirk_hashes"]) == PACKAGE_RUNTIME_FILES
    assert all(len(value) == 12 for value in fields["quirk_hashes"].values())


def test_required_missing_runtime_stops_before_the_model(
    monkeypatch, tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness.loop import solve_task

    (tmp_path / "prompt.txt").write_text("must not run")
    client = MagicMock()

    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.policy.shutil.which",
        lambda _name: None,
    )
    cfg = make_config(
        max_sessions=1,
        sandbox_bash=True,
        sandbox_required=True,
        sandbox_backend="container",
        sandbox_container_image=IMAGE,
    )

    with pytest.raises(RuntimeError, match="'docker'.*not installed"):
        solve_task(tmp_path, cfg, client)
    client.chat.assert_not_called()


def test_lsp_builder_uses_container_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ContainerBackend, 'image_digest', lambda *a, **kw: DIGEST)
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    monkeypatch.setattr(
        ContainerBackend,
        "resolve_runtime",
        lambda self, *, sandbox_required: "/usr/bin/docker",
    )

    argv = build_lsp_sandbox_argv(
        ("fake-lsp", "--stdio"), cwd=str(tmp_path),
        bwrap_bin=MISSING_BWRAP, sandbox_required=True,
        sandbox_backend="container", container_image=IMAGE,
    )

    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert argv[-1] == "fake-lsp --stdio"
    assert "--network" in argv and "none" in argv


@pytest.mark.parametrize('required', [False, True])
@pytest.mark.parametrize('engaged', [False, True])
def test_solve_task_records_preflight_and_never_disables_a_failed_selected_sandbox(
    monkeypatch, tmp_path: Path, required, engaged,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.server.types import TurnResult, Usage
    from dataclasses import replace
    from scripts.llm_solver.harness import task_file_runtime

    # This test owns trace emission, not container transport. Supply an explicit
    # local fixture executor alongside its synthetic runtime-envelope record.
    make_files = task_file_runtime.make_task_files
    monkeypatch.setattr(task_file_runtime, 'make_task_files',
                        lambda cwd, cfg, **kw: make_files(
                            cwd, replace(cfg, sandbox_bash=False, sandbox_required=False,
                                sandbox_backend='none', sandbox_resolved_backend='none'), **kw))

    (tmp_path / "prompt.txt").write_text("finish")
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content="done", tool_calls=[], finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant", "content": "done",
    }
    fields = {
        "session": 1,
        "sandbox_mode": "container",
        "sandbox_engaged": engaged,
        "sandbox_backend": "container",
        "sandbox_backend_executable": "/private/host/bin/docker",
        "container_runtime": "docker",
        "container_image_digest": DIGEST,
        "container_preflight_error": None if engaged else 'selected runtime unavailable',
        "sandbox_bash_cfg": True,
        "sandbox_required_cfg": required,
        "bwrap_bin": MISSING_BWRAP,
        "bwrap_present": False,
        "bwrap_preflight_passed": None,
        "bwrap_preflight_error": None,
        "yuj_container": None,
        "ambient_unshare_net": None,
        "task_id": tmp_path.name,
        "guardrail_map": {},
        "quirk_hashes": {},
        "detected_runner": "pytest",
        "unreadable_paths_n_files": 0,
        "unreadable_paths_n_dirs": 0,
        "unreadable_paths_zero_match_patterns": [],
        "sandbox_policy_version": 3,
    }
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.driver.compute_runtime_envelope_fields",
        lambda _cfg, _repo: dict(fields),
    )
    from scripts.llm_solver.harness.sandbox.policy import SandboxCapabilities, SandboxResolution
    monkeypatch.setattr('scripts.llm_solver.harness.sandbox.policy.preflight_sandbox',
        lambda cfg: SandboxResolution('docker', 'docker',
            SandboxCapabilities('linux', ('docker',), ('docker',), {'docker': '/fixture/docker'}),
            '/fixture/docker', False, True, DIGEST))
    cfg = make_config(
        max_sessions=1,
        sandbox_env_inherit="none",
        sandbox_bash=True,
        sandbox_required=required,
        sandbox_backend="container",
        sandbox_container_runtime="docker",
        sandbox_container_image=IMAGE,
    )
    client.cfg = cfg
    pretest = MagicMock(return_value='')
    monkeypatch.setattr('scripts.llm_solver.harness._loop.driver.run_pretest', pretest)

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        if engaged:
            assert solve_task(tmp_path, cfg, client) is True
        else:
            with pytest.raises(RuntimeError, match='sandbox_engaged=false'):
                solve_task(tmp_path, cfg, client)
            client.chat.assert_not_called()
            pretest.assert_not_called()
            assert client.cfg.sandbox_bash is True

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    envelope = next(event for event in events if event['event'] == 'runtime_envelope')
    assert envelope['sandbox_engaged'] is engaged
    assert envelope['sandbox_required_cfg'] is required
    if not engaged:
        assert not any(event['event'] in ('session_start', 'pretest_run') for event in events)
        return
    start = next(event for event in events if event["event"] == "session_start")
    envelope = next(
        event for event in events if event["event"] == "runtime_envelope"
    )
    assert "sandbox_backend_executable" not in envelope
    assert start["sandbox_backend"] == "container"
    assert start["container_runtime"] == "docker"
    assert start["container_image_digest"] == DIGEST

    from scripts.llm_solver.harness._loop.trace_schema import (
        TRACE_EVENT_REQUIRED_FIELDS,
    )
    assert {
        "sandbox_backend", "container_runtime", "container_image_digest",
    } <= TRACE_EVENT_REQUIRED_FIELDS["session_start"]
