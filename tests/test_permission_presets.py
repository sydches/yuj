"""Assistant permission-preset configuration and enforcement contract."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.llm_assist.__main__ import (
    _render_provider_overlay,
    _transport_overrides_from_args,
    main as assist_main,
)
from scripts.llm_solver import config as config_module
from scripts.llm_solver._permission_presets import (
    ASSISTANT_PERMISSION_PRESET_NAMES,
    PERMISSION_PRESET_SPECS,
)
from scripts.llm_solver.config import (
    ConfigLayerSpec,
    dump_config,
    load_config,
    resolve_config,
)
from scripts.llm_solver.config_inspection import (
    build_inspection_document,
    render_inspection_human,
)
from scripts.llm_solver.harness.approvals import approval_decision
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.plan_mode import PlanModeController
from scripts.llm_solver.harness.sandbox._preflight import bwrap_preflight
from scripts.llm_solver.harness.tool_policy import PermissionPolicy
from scripts.llm_solver.harness.tools import dispatch


EXPECTED_NAMES = (
    "read-only",
    "ask-before-changes",
    "allow-edits",
)
EXPECTED_DECISIONS = {
    "read-only": ("allow", "deny", "deny"),
    "ask-before-changes": ("allow", "ask", "ask"),
    "allow-edits": ("allow", "allow", "ask"),
}
EXPECTED_PLAN_MODES = {
    "read-only": "off",
    "ask-before-changes": "required",
    "allow-edits": "off",
}
BWRAP_AVAILABLE, BWRAP_FAILURE = bwrap_preflight("/usr/bin/bwrap")


def _without_machine_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )


def _preset_overlay(
    tmp_path: Path,
    name: str,
    *,
    extra: str = "",
) -> Path:
    overlay = tmp_path / f"{name}.toml"
    overlay.write_text(
        "[runtime]\n"
        'mode = "assistant"\n\n'
        "[assistant]\n"
        f'permission_preset = "{name}"\n'
        f"{extra}"
    )
    return overlay


def _policy(cfg) -> PermissionPolicy:
    return PermissionPolicy.from_rule_tables(
        cfg.permissions_preset_rules,
        cfg.permissions_rules,
    )


def _decision(cfg, tool_name: str, arguments: dict) -> str:
    return _policy(cfg).evaluate(
        tool_name=tool_name,
        arguments=arguments,
        runtime_mode=cfg.runtime_mode,
        ask_fallback=cfg.permissions_ask_fallback,
        approval_available=True,
    ).decision


def test_preset_enum_and_mapping_are_fixed_and_immutable() -> None:
    assert ASSISTANT_PERMISSION_PRESET_NAMES == EXPECTED_NAMES
    assert tuple(PERMISSION_PRESET_SPECS) == EXPECTED_NAMES

    with pytest.raises(TypeError):
        PERMISSION_PRESET_SPECS["new"] = PERMISSION_PRESET_SPECS["read-only"]
    with pytest.raises(FrozenInstanceError):
        PERMISSION_PRESET_SPECS["read-only"].plan_mode = "required"


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_each_preset_expands_to_documented_representative_decisions(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(_preset_overlay(tmp_path, name))

    assert cfg.plan_mode == EXPECTED_PLAN_MODES[name]
    assert cfg.permissions_ask_fallback == "deny"
    assert _decision(cfg, "list_functions", {}) == "allow"
    assert (
        _decision(cfg, "read", {"path": "README.md"}),
        _decision(
            cfg,
            "edit",
            {"path": "app.py", "old_str": "old", "new_str": "new"},
        ),
        _decision(cfg, "bash", {"cmd": "pytest -q"}),
    ) == EXPECTED_DECISIONS[name]


def test_explicit_controls_override_preset_with_leaf_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _preset_overlay(
        tmp_path,
        "allow-edits",
        extra=(
            "\n[loop]\n"
            'plan_mode = "required"\n\n'
            "[permissions]\n"
            'ask_fallback = "allow"\n\n'
            "[permissions.rules.edit]\n"
            '"private/*" = "deny"\n\n'
            "[permissions.rules.bash]\n"
            '"pytest *" = "allow"\n'
        ),
    )
    resolved = resolve_config(
        user_config=[overlay],
        layer_specs=[
            ConfigLayerSpec(
                overlay, "user-overlay", "overlay", "--config[1]"
            )
        ],
    )
    cfg = resolved.config

    assert cfg.plan_mode == "required"
    assert cfg.permissions_ask_fallback == "allow"
    assert _decision(
        cfg,
        "edit",
        {"path": "public/app.py", "old_str": "old", "new_str": "new"},
    ) == "allow"
    assert _decision(
        cfg,
        "edit",
        {"path": "private/key.py", "old_str": "old", "new_str": "new"},
    ) == "deny"
    assert _decision(cfg, "bash", {"cmd": "pytest -q"}) == "allow"
    assert _decision(cfg, "bash", {"cmd": "git status"}) == "ask"

    assert resolved.provenance[("loop", "plan_mode")].layer_id == (
        "user-overlay"
    )
    assert resolved.provenance[
        ("permissions", "ask_fallback")
    ].layer_id == "user-overlay"
    assert resolved.provenance[
        ("permissions", "rules", "edit", "private/*")
    ].layer_id == "user-overlay"
    assert resolved.provenance[
        ("permissions", "preset_rules", "edit", "*")
    ].layer_id == "assistant-permission-preset"


def test_explicit_rule_order_remains_authoritative_after_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(
        _preset_overlay(
            tmp_path,
            "read-only",
            extra=(
                "\n[permissions.rules.edit]\n"
                '"*" = "deny"\n\n'
                '[permissions.rules."*"]\n'
                '"*" = "allow"\n'
            ),
        )
    )

    assert list(cfg.permissions_rules) == ["edit", "*"]
    assert _decision(
        cfg,
        "edit",
        {"path": "app.py", "old_str": "old", "new_str": "new"},
    ) == "allow"


def test_explicit_global_deny_overrides_every_representative_preset_allow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(
        _preset_overlay(
            tmp_path,
            "allow-edits",
            extra=(
                '\n[permissions.rules."*"]\n'
                '"*" = "deny"\n'
            ),
        )
    )

    assert _decision(cfg, "read", {"path": "README.md"}) == "deny"
    assert _decision(
        cfg,
        "edit",
        {"path": "app.py", "old_str": "old", "new_str": "new"},
    ) == "deny"
    assert _decision(cfg, "bash", {"cmd": "pytest -q"}) == "deny"


def test_resolved_views_show_selection_every_expanded_rule_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _preset_overlay(tmp_path, "allow-edits")
    resolved = resolve_config(
        user_config=[overlay],
        layer_specs=[
            ConfigLayerSpec(
                overlay, "user-overlay", "overlay", "--config[1]"
            )
        ],
    )
    document = build_inspection_document(resolved, success=True)
    entries = {
        tuple(entry["path_components"]): entry
        for entry in document["settings"]
    }

    assert entries[("assistant", "permission_preset")] == {
        "path": "assistant.permission_preset",
        "path_components": ["assistant", "permission_preset"],
        "value": "allow-edits",
        "source_layer": "user-overlay",
        "redacted": False,
        "redaction_reasons": [],
    }
    expected_rule_paths = {
        ("permissions", "preset_rules", tool_name, "*")
        for tool_name in PERMISSION_PRESET_SPECS["allow-edits"].tool_decisions
    }
    assert expected_rule_paths <= entries.keys()
    assert all(
        entries[path]["source_layer"] == "assistant-permission-preset"
        for path in expected_rule_paths
    )
    assert entries[("loop", "plan_mode")]["source_layer"] == (
        "assistant-permission-preset"
    )
    assert entries[("permissions", "ask_fallback")]["source_layer"] == (
        "assistant-permission-preset"
    )
    assert [layer["id"] for layer in document["layers"]] == [
        "checked-in-defaults",
        "assistant-permission-preset",
        "machine-local",
        "user-overlay",
        "command-line",
    ]

    human = render_inspection_human(document)
    assert "assistant.permission_preset = \"allow-edits\" [user-overlay]" in human
    assert (
        'permissions.preset_rules.edit."*" = "allow" '
        "[assistant-permission-preset]"
    ) in human


def test_public_json_view_reports_cli_selection_and_expansion_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "no-sandbox.toml"
    overlay.write_text('[sandbox]\nbackend = "none"\n')

    rc = assist_main([
        "config",
        "--json",
        "--no-treatment",
        "--permission-preset",
        "read-only",
        "--config",
        str(overlay),
    ])

    payload = json.loads(capsys.readouterr().out)
    entries = {
        tuple(entry["path_components"]): entry
        for entry in payload["settings"]
    }
    assert rc == 0
    assert entries[("assistant", "permission_preset")]["source_layer"] == (
        "command-line"
    )
    assert entries[
        ("permissions", "preset_rules", "*", "*")
    ]["source_layer"] == "assistant-permission-preset"
    assert entries[("permissions", "rules")]["source_layer"] == (
        "checked-in-defaults"
    )


def test_invalid_preset_fails_configuration_validation_precisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _preset_overlay(tmp_path, "automatic")

    with pytest.raises(
        ValueError,
        match=(
            "config error: assistant.permission_preset must be one of: "
            "read-only, ask-before-changes, allow-edits; got 'automatic'"
        ),
    ):
        load_config(overlay)


def test_invalid_preset_has_the_same_precise_public_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _preset_overlay(tmp_path, "automatic")

    rc = assist_main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(overlay),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["diagnostics"] == [{
        "level": "error",
        "code": "config_invalid",
        "message": (
            "config error: assistant.permission_preset must be one of: "
            "read-only, ask-before-changes, allow-edits; got 'automatic'."
        ),
    }]


def test_no_preset_preserves_defaults_and_measurement_ignores_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    baseline = load_config(overrides={"runtime_mode": "measurement"})
    overlay = tmp_path / "measurement-preset.toml"
    overlay.write_text(
        "[assistant]\n"
        'permission_preset = "read-only"\n'
    )
    selected = load_config(
        overlay,
        overrides={"runtime_mode": "measurement"},
    )

    assert selected == baseline
    assert dump_config(selected) == dump_config(baseline)
    assert baseline.plan_mode == "off"
    assert baseline.permissions_preset_rules == {}
    assert baseline.permissions_rules == {}
    assert baseline.permissions_ask_fallback == "deny"
    for tool_name, arguments in (
        ("read", {"path": "README.md"}),
        ("edit", {"path": "app.py", "old_str": "a", "new_str": "b"}),
        ("bash", {"cmd": "pytest -q"}),
    ):
        assert _decision(selected, tool_name, arguments) == _decision(
            baseline, tool_name, arguments
        )


def test_plan_mode_still_blocks_a_preset_ask_before_permission_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(_preset_overlay(tmp_path, "ask-before-changes"))
    controller = PlanModeController(
        cwd=str(tmp_path),
        cfg=cfg,
        events=(),
        event_sink=lambda _event: None,
    )

    assert _decision(
        cfg,
        "edit",
        {"path": "app.py", "old_str": "a", "new_str": "b"},
    ) == "ask"
    assert controller.check("read", {"path": "README.md"}, turn=0).allowed
    assert not controller.check(
        "edit",
        {"path": "app.py", "old_str": "a", "new_str": "b"},
        turn=0,
    ).allowed


def test_preset_ask_still_requires_the_existing_approval_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(_preset_overlay(tmp_path, "allow-edits"))
    session = Session(cfg, MagicMock(), "system", "task", str(tmp_path))
    resolution = session._permission_policy.evaluate(
        tool_name="bash",
        arguments={"cmd": "pytest -q"},
        runtime_mode=cfg.runtime_mode,
        ask_fallback=cfg.permissions_ask_fallback,
        approval_available=True,
    )

    allowed, reason = approval_decision(
        runtime_mode=cfg.runtime_mode,
        cwd=str(tmp_path),
        trace_path=tmp_path / ".trace.jsonl",
        tool_name="bash",
        tool_args={"cmd": "pytest -q"},
        args_summary="cmd='pytest -q'",
        required_reason=resolution.approval_reason(),
        permission_rule=resolution.rule,
    )

    assert allowed is False
    assert "requires operator approval" in str(reason)
    request = json.loads((tmp_path / "approval_request.json").read_text())
    assert request["status"] == "pending"
    assert request["tool_name"] == "bash"
    assert request["permission_rule"] == "*"


@pytest.mark.skipif(
    not BWRAP_AVAILABLE,
    reason=f"operational bwrap is required: {BWRAP_FAILURE or 'unavailable'}",
)
def test_explicit_shell_allow_still_cannot_bypass_the_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    cfg = load_config(
        _preset_overlay(
            tmp_path,
            "allow-edits",
            extra=(
                "\n[permissions.rules.bash]\n"
                '"*" = "allow"\n'
            ),
        )
    )
    target = Path("/usr/local/lib/yuj_permission_preset_escape_issue_46")
    assert not target.exists()
    assert _decision(cfg, "bash", {"cmd": f"touch {target}"}) == "allow"

    result = dispatch(
        "bash",
        {"cmd": f"touch {target} 2>&1"},
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert "Read-only file system" in result or "read-only" in result.lower()
    assert not target.exists()


def test_assistant_cli_maps_and_persists_permission_preset() -> None:
    args = SimpleNamespace(
        provider=None,
        base_url=None,
        api_key_env=None,
        thinking=None,
        plan_mode=None,
        edit_format=None,
        permission_preset="allow-edits",
    )
    assert _transport_overrides_from_args(args) == {
        "assistant_permission_preset": "allow-edits"
    }
    assert _render_provider_overlay(
        {"assistant_permission_preset": "allow-edits"}
    ) == (
        "[assistant]\n"
        'permission_preset = "allow-edits"\n'
    )


def test_setup_saves_permission_preset_in_machine_local_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert assist_main([
        "setup",
        "--provider",
        "local",
        "--model",
        "local-model",
        "--permission-preset",
        "read-only",
        "--sandbox",
        "none",
        "--force",
    ]) == 0

    config_text = config_path.read_text()
    assert (
        "[assistant]\n"
        'permission_preset = "read-only"\n'
    ) in config_text
    assert "[sandbox]\n" in config_text
    assert 'backend = "none"\n' in config_text


@pytest.mark.parametrize(
    ("command", "handler"),
    [
        ("code", "cmd_run"),
        ("run", "cmd_run"),
        ("smoke", "cmd_smoke"),
        ("config", "cmd_config"),
        ("setup", "cmd_setup"),
    ],
)
def test_assistant_cli_exposes_permission_preset(
    command: str,
    handler: str,
) -> None:
    from unittest.mock import patch

    with patch(
        f"scripts.llm_assist.__main__.{handler}", return_value=0
    ) as invoked:
        assert assist_main(
            [command, "--permission-preset", "read-only"]
        ) == 0

    assert invoked.call_args.args[0].permission_preset == "read-only"
