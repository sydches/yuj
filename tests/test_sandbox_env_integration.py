"""Runtime integration coverage for the public ``[sandbox.env]`` policy."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import dump_config, load_config
from scripts.llm_solver.harness import tools as tools_mod
from scripts.llm_solver.harness._tools.run_tests import run_tests
from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
from scripts.llm_solver.harness.post_edit import run_post_edit_checks
from scripts.llm_solver.harness.process_manager import (
    build_background_sandbox_argv,
)
from scripts.llm_solver.harness.sandbox import _build_bwrap_argv
from scripts.llm_solver.harness.sandbox import env_policy
from scripts.llm_solver.harness.tools import dispatch


def test_canonical_config_loads_validates_and_redacts_env_policy(
    tmp_path: Path,
) -> None:
    defaults = load_config()
    assert defaults.sandbox_env_inherit == "core"
    assert defaults.sandbox_env_allow_login_shell is False
    assert defaults.sandbox_env_ignore_default_excludes is False
    assert defaults.sandbox_env_set["TERM"] == "dumb"
    assert set(dump_config(defaults)["sandbox_env_set"].values()) == {
        "<redacted>"
    }

    overlay = tmp_path / "env.toml"
    overlay.write_text(
        """
[sandbox.env]
inherit = "none"
set = { FIXED = "private-value" }
ignore_default_excludes = true
allow_login_shell = true

[sandbox.env.filters]
FIXED = "include"
""".strip()
    )
    configured = load_config(user_config=overlay)
    assert configured.sandbox_env_inherit == "none"
    assert configured.sandbox_env_set["FIXED"] == "private-value"
    assert configured.sandbox_env_filters == {"FIXED": "include"}
    assert configured.sandbox_env_ignore_default_excludes is True
    assert configured.sandbox_env_allow_login_shell is True

    overlay.write_text('[sandbox.env]\ninherit = "host"\n')
    with pytest.raises(ValueError, match="sandbox.env.inherit"):
        load_config(user_config=overlay)


def test_dispatch_applies_policy_without_mutating_harness_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("VISIBLE_HOST", "host-value")
    monkeypatch.setenv("SERVICE_TOKEN", "host-secret")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    cfg = make_config(
        sandbox_bash=False,
        sandbox_env_inherit="all",
        sandbox_env_set={
            "PATH": "/usr/bin:/bin",
            "VISIBLE_SET": "fixed-value",
        },
    )

    result = dispatch(
        "bash",
        {
            "cmd": (
                "printf '%s|%s|%s' \"$VISIBLE_HOST\" "
                "\"${SERVICE_TOKEN-unset}\" \"$VISIBLE_SET\""
            )
        },
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert "host-value|unset|fixed-value" in result
    assert os.environ["SERVICE_TOKEN"] == "host-secret"


def test_ambient_container_applies_the_same_explicit_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    monkeypatch.setenv("YUJ_AMBIENT_UNSHARE_NET", "0")
    monkeypatch.setenv("VISIBLE_HOST", "host-value")
    monkeypatch.setenv("SERVICE_TOKEN", "host-secret")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        "scripts.llm_solver.harness._tools._run_in_sandbox."
        "_probe_ambient_unshare_net",
        lambda: False,
    )
    cfg = make_config(
        sandbox_bash=True,
        sandbox_required=True,
        sandbox_env_inherit="all",
        sandbox_env_set={
            "PATH": "/usr/bin:/bin",
            "VISIBLE_SET": "fixed-value",
        },
    )

    result = dispatch(
        "bash",
        {
            "cmd": (
                "printf '%s|%s|%s' \"$VISIBLE_HOST\" "
                "\"${SERVICE_TOKEN-unset}\" \"$VISIBLE_SET\""
            )
        },
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert "host-value|unset|fixed-value" in result
    assert os.environ["SERVICE_TOKEN"] == "host-secret"


@pytest.mark.parametrize("path,home", [
    ("/opt/runtime/bin:/usr/bin", "/home/worker"),
    ("/workspace/tools:/bin", "/tmp/work home"),
])
def test_container_inheritance_uses_observed_values_and_existing_policy(
    monkeypatch, path, home, bound_container_environment,
):
    monkeypatch.setenv("YUJ_CONTAINER", "observed-container")
    monkeypatch.setenv("PATH", "/host-only/bin")
    monkeypatch.setenv("HOME", "/host-only/home")
    monkeypatch.setenv("HOST_ONLY", "must-not-cross")
    calls = []

    def probe(argv, **kwargs):
        calls.append(argv)
        assert 0 < kwargs["timeout"] <= 17 and kwargs['binary'] is True
        return SimpleNamespace(returncode=0, stdout=(
            f"PATH={path}\0HOME={home}\0TERM=container-term\0"
            "SERVICE_TOKEN=container-secret\0DROP=unwanted\0"
            "LITERAL=line1\nline2=$(do-not-execute)=value\0"
        ).encode())

    monkeypatch.setattr(bound_container_environment, "_execute", probe)
    cfg = make_config(
        sandbox_bash=True, bash_timeout=17,
        sandbox_env_inherit="all", sandbox_env_set={"TERM": "policy-term"},
        sandbox_env_filters={"DROP": "exclude"},
    )
    effective, login = tools_mod._effective_command_environment(cfg)
    assert effective == {
        "PATH": path, "HOME": home, "TERM": "policy-term",
        "LITERAL": "line1\nline2=$(do-not-execute)=value",
    }
    assert login is False
    assert len(calls) == 1
    assert calls[0][:5] == ['docker', 'exec', '--workdir', '/selected/task', 'a' * 64]
    assert 'observed-container' not in calls[0]
    assert os.environ["PATH"] == "/host-only/bin"


@pytest.mark.parametrize("failure", [
    FileNotFoundError("private diagnostic"),
    subprocess.CalledProcessError(1, ["docker"], stderr="private diagnostic"),
    subprocess.TimeoutExpired(["docker"], 10, output="private diagnostic"),
])
def test_failed_container_probe_never_falls_back_or_exposes_output(monkeypatch, failure, bound_container_environment):
    monkeypatch.setenv("YUJ_CONTAINER", "unavailable-container")
    monkeypatch.setenv("PATH", "/host-fallback/bin")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bound_container_environment, "_execute", fail)
    with pytest.raises(env_policy.EnvironmentPolicyError, match="cannot inspect") as caught:
        tools_mod._effective_command_environment(make_config(sandbox_bash=True))
    assert "private diagnostic" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("payload", [
    b"PATH=/truncated", b"no-equals\0", b"=empty-name\0",
    b"PATH=/one\0PATH=/two\0",
])
def test_malformed_container_environment_is_not_used(monkeypatch, payload, bound_container_environment):
    monkeypatch.setenv("YUJ_CONTAINER", "container")
    monkeypatch.setattr(bound_container_environment, "_execute", lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload))
    with pytest.raises(env_policy.EnvironmentPolicyError, match="invalid container"):
        tools_mod._effective_command_environment(make_config(sandbox_bash=True))


def test_disabled_inheritance_does_not_require_a_container_probe(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "container")

    def unexpected(*args, **kwargs):
        pytest.fail("inherit=none must not inspect an unused source")

    monkeypatch.setattr(env_policy.subprocess, "run", unexpected)
    effective, _ = tools_mod._effective_command_environment(make_config(
        sandbox_env_inherit="none", sandbox_env_set={"ONLY": "declared"},
    ))
    assert effective == {"ONLY": "declared"}


def test_unused_container_selector_does_not_change_local_command_environment(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "unused-container")
    monkeypatch.setenv("PATH", "/local/bin")
    monkeypatch.setattr(env_policy.subprocess, "run", lambda *a, **k: pytest.fail("unused container"))
    effective, _ = tools_mod._effective_command_environment(make_config(
        sandbox_bash=False, sandbox_env_set={},
    ))
    assert effective["PATH"] == "/local/bin"


@pytest.mark.parametrize('mismatched_credentials', [False, True])
def test_container_environment_probe_checks_credentials_and_reads_initial_environment(
    monkeypatch, tmp_path, mismatched_credentials,
):
    import importlib
    from dataclasses import replace
    from scripts.llm_solver.harness import task_environment
    from scripts.llm_solver.harness.process_identity import (
        OBSERVE_SCRIPT, GuardedProcessArgv, parse_process_identity,
    )
    execution = importlib.import_module('scripts.llm_solver.harness._tools._run_in_sandbox')
    result = subprocess.run(['bash', '--noprofile', '--norc', '-p', '-c', OBSERVE_SCRIPT],
                            capture_output=True, check=True)
    identity = parse_process_identity(result.stdout)
    if mismatched_credentials:
        identity = replace(identity, uids=(identity.uids[0] + 1,) * 4)
    selected = task_environment.TaskEnvironment(
        str(tmp_path), '/selected/task', ('/selected/task',),
        container='reusable-name', container_id='b' * 64, process_identity=identity,
    )
    seen = []

    def discover(cwd, **kwargs):
        assert cwd == tmp_path
        return selected

    # Substitute only the Docker transport with a real local entry process.
    # The credential guard and environment observation execute unchanged.
    native_execute = execution._execute
    initial = {'PATH': os.environ['PATH'], 'HOME': '/native-home', 'SHLVL': '37',
               'PWD': '/original-spelling', 'LITERAL': 'line1\nline2=$(unused)=value'}

    def execute(argv, **kwargs):
        assert isinstance(argv, GuardedProcessArgv)
        assert argv[:5] == ['docker', 'exec', '--workdir', '/selected/task', 'b' * 64]
        seen.append(True)
        return native_execute(GuardedProcessArgv(argv[5:]), env=initial, **kwargs)

    monkeypatch.setenv('YUJ_CONTAINER', 'reusable-name')
    monkeypatch.setattr(task_environment, 'discover_task_environment', discover)
    monkeypatch.setattr(execution, '_execute', execute)
    if mismatched_credentials:
        with pytest.raises(env_policy.EnvironmentPolicyError, match='cannot inspect'):
            env_policy.discover_execution_environment(cwd=tmp_path, timeout=7)
    else:
        assert env_policy.discover_execution_environment(cwd=tmp_path, timeout=7) == initial
    assert len(seen) == 1


def test_bwrap_background_and_lsp_builders_share_the_explicit_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    effective = {"ONLY": "visible"}

    bwrap = _build_bwrap_argv(
        "env", str(tmp_path), "/usr/bin/bwrap", effective_env=effective,
    )
    clear_index = bwrap.index("--clearenv")
    assert bwrap[clear_index:clear_index + 4] == [
        "--clearenv", "--setenv", "ONLY", "visible",
    ]
    assert bwrap[-7:] == [
        "bash", "--noprofile", "--norc", "-o", "pipefail", "-c", "env",
    ]

    background = build_background_sandbox_argv(
        "env", cwd=str(tmp_path), bwrap_bin="missing", sandbox=False,
        effective_env=effective,
    )
    lsp = build_lsp_sandbox_argv(
        ("fake-lsp", "--stdio"), cwd=str(tmp_path), bwrap_bin="missing",
        sandbox=False, effective_env=effective,
    )
    assert background[:4] == [
        "/usr/bin/env", "-i", "ONLY=visible", "bash",
    ]
    assert lsp == [
        "/usr/bin/env", "-i", "ONLY=visible", "fake-lsp", "--stdio",
    ]


def test_run_tests_and_post_edit_receive_the_same_policy(tmp_path: Path) -> None:
    effective = {"ONLY": "check-value", "PATH": "/usr/bin:/bin"}
    cfg = make_config(
        analysis_task_format="pytest",
        tools_run_tests_enabled=True,
        sandbox_env_inherit="none",
        sandbox_env_set=effective,
        sandbox_env_allow_login_shell=True,
        post_edit_check_enabled=True,
        post_edit_checks=[{
            "name": "syntax",
            "trigger": "write",
            "when": "",
            "cmd": "true",
            "on_fail": "append",
        }],
    )
    captured_test: dict[str, object] = {}
    captured_check: dict[str, object] = {}

    def fake_sandbox(_cmd, **kwargs):
        captured_test.update(kwargs)
        return "ok", 0, False

    def fake_bash(_cmd, **kwargs):
        captured_check.update(kwargs)
        return ""

    with (
        patch.object(tools_mod, "_run_in_sandbox", side_effect=fake_sandbox),
        patch.object(tools_mod, "bash", side_effect=fake_bash),
    ):
        run_tests(cwd=str(tmp_path), cfg=cfg)
        run_post_edit_checks(
            "source.py", cwd=str(tmp_path), cfg=cfg, trigger="write",
        )

    assert captured_test["effective_env"] == effective
    assert captured_check["effective_env"] == effective
    assert captured_test["allow_login_shell"] is True
    assert captured_check["allow_login_shell"] is True


def test_driver_resolves_once_and_traces_names_without_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness import loop as loop_mod
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.server.types import TurnResult, Usage

    (tmp_path / "prompt.txt").write_text("continue")
    monkeypatch.delenv("LATE_VISIBLE", raising=False)
    discoveries = []
    original_discover = tools_mod.discover_execution_environment

    def discover_once(**kwargs):
        discoveries.append(True)
        return original_discover(**kwargs)

    monkeypatch.setattr(tools_mod, "discover_execution_environment", discover_once)
    captured_envs: list[object] = []
    original_init = loop_mod.Session.__init__

    def recording_init(self, *args, **kwargs):
        captured_envs.append(kwargs["effective_env"])
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(loop_mod.Session, "__init__", recording_init)
    calls = 0

    def chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            monkeypatch.setenv("LATE_VISIBLE", "too-late")
        return TurnResult(
            content="continue", tool_calls=[], finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )

    client = MagicMock()
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {
        "role": "assistant", "content": "continue",
    }
    cfg = make_config(
        max_sessions=2,
        max_turns=1,
        allow_implicit_done=False,
        state_writer_enabled=False,
        sandbox_env_inherit="all",
        sandbox_env_set={"TRACE_FIXED": "sensitive-value"},
    )

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(tmp_path, cfg, client) is False

    assert len(captured_envs) == 2
    assert len(discoveries) == 1
    assert captured_envs[0] is captured_envs[1]
    assert "LATE_VISIBLE" not in captured_envs[0]
    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event["event"] == "session_start"]
    assert len(starts) == 2
    assert all("TRACE_FIXED" in event["sandbox_env_names"] for event in starts)
    assert all("LATE_VISIBLE" not in event["sandbox_env_names"] for event in starts)
    assert all(event["task_environment"] == {
        "host_root": str(tmp_path), "working_directory": str(tmp_path), "aliases": [],
    } for event in starts)
    assert "sensitive-value" not in trace_path(tmp_path).read_text()
    from scripts.llm_solver.harness._loop.trace_schema import (
        TRACE_EVENT_REQUIRED_FIELDS,
    )
    assert "sandbox_env_names" in TRACE_EVENT_REQUIRED_FIELDS["session_start"]
