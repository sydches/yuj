"""Runtime integration coverage for the public ``[sandbox.env]`` policy."""
from __future__ import annotations

import json
import os
from pathlib import Path
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
    assert "sensitive-value" not in trace_path(tmp_path).read_text()
    from scripts.llm_solver.harness._loop.trace_schema import (
        TRACE_EVENT_REQUIRED_FIELDS,
    )
    assert "sandbox_env_names" in TRACE_EVENT_REQUIRED_FIELDS["session_start"]
