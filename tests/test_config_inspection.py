"""Public resolved-configuration inspection and provenance contract."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_solver import config as config_module
from scripts.llm_solver.config import (
    ConfigLayerSpec,
    dump_transformations,
    load_config,
    resolve_config,
    resolve_transformation_context_mode,
)
from scripts.llm_solver.config_inspection import (
    build_inspection_document,
    render_inspection_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _without_machine_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )


def _no_sandbox_overlay(tmp_path: Path) -> Path:
    overlay = tmp_path / "no-sandbox.toml"
    overlay.write_text('[sandbox]\nbackend = "none"\n')
    return overlay


def test_long_context_runtime_sets_its_own_read_timeout() -> None:
    runtime = (
        PROJECT_ROOT
        / "configs/runtime"
        / "llama-qwen36-35b-a3b-q4kxl-5090-no-offload-mtp.toml"
    )

    with (PROJECT_ROOT / "config.toml").open("rb") as handle:
        defaults = tomllib.load(handle)
    with runtime.open("rb") as handle:
        overlay = tomllib.load(handle)

    assert defaults["server"]["timeout_read"] == 145
    assert overlay["server"] == {"timeout_read": 240}


def _flatten_paths(value: object, prefix: tuple[str, ...] = ()):
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from _flatten_paths(value[key], (*prefix, str(key)))
        return
    if prefix:
        yield prefix


def _entry_map(document: dict) -> dict[tuple[str, ...], dict]:
    return {
        tuple(entry["path_components"]): entry
        for entry in document["settings"]
    }


def _resolved_for_public_command(
    *,
    treatment: bool = True,
    overlays: list[Path] | None = None,
    overrides: dict | None = None,
):
    base = (
        PROJECT_ROOT / "configs/regimes/treatment.toml"
        if treatment
        else PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"
    )
    paths = [base, *(overlays or [])]
    specs = [
        ConfigLayerSpec(
            path=base,
            layer_id="base",
            kind="base",
            label="treatment" if treatment else "plain",
        ),
        *[
            ConfigLayerSpec(
                path=path,
                layer_id=f"overlay-{index}",
                kind="overlay",
                label=f"--config[{index}]",
            )
            for index, path in enumerate(overlays or [], 1)
        ],
    ]
    effective_overrides = {"runtime_mode": "assistant", "max_sessions": 1}
    effective_overrides.update(overrides or {})
    return resolve_config(
        user_config=paths,
        overrides=effective_overrides,
        layer_specs=specs,
    )


def test_resolution_tracks_every_real_layer_and_cli_wins_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "config.local.toml"
    local.write_text(
        '[server]\nbase_url = "http://machine.invalid/v1"\n'
        "[loop]\nmax_turns = 210\n"
    )
    monkeypatch.setattr(config_module, "_LOCAL_CONFIG", local)

    base = tmp_path / "base.toml"
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    base.write_text("[loop]\nmax_turns = 220\n")
    first.write_text('[model]\nname = "overlay-model"\n[loop]\nmax_turns = 230\n')
    second.write_text("[loop]\nmax_turns = 240\n")

    resolved = resolve_config(
        user_config=[base, first, second],
        overrides={"model": "cli-model"},
        layer_specs=[
            ConfigLayerSpec(base, "base", "base", "treatment"),
            ConfigLayerSpec(first, "overlay-1", "overlay", "--config[1]"),
            ConfigLayerSpec(second, "overlay-2", "overlay", "--config[2]"),
        ],
    )

    assert resolved.config.max_turns == 240
    assert resolved.config.model == "cli-model"
    assert resolved.config.base_url == "http://machine.invalid/v1"
    assert resolved.provenance[("loop", "max_turns")].layer_id == "overlay-2"
    assert resolved.provenance[("model", "name")].layer_id == "command-line"
    assert resolved.provenance[("server", "base_url")].layer_id == "machine-local"
    assert (
        resolved.provenance[("output", "truncate_head_ratio")].layer_id
        == "checked-in-defaults"
    )
    assert [layer.kind for layer in resolved.layers] == [
        "defaults",
        "machine-local",
        "base",
        "overlay",
        "overlay",
        "command-line",
    ]


def test_inspection_redacts_literal_nested_environment_and_future_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    sentinels = {
        "environment": "SENSITIVE_SENTINEL_ENVIRONMENT",
        "header": "SENSITIVE_SENTINEL_HEADER",
        "role": "SENSITIVE_SENTINEL_ROLE",
        "fixed_env": "SENSITIVE_SENTINEL_FIXED_ENV",
        "future": "SENSITIVE_SENTINEL_FUTURE",
        "future_camel": "SENSITIVE_SENTINEL_FUTURE_CAMEL",
        "future_keys": "SENSITIVE_SENTINEL_FUTURE_KEYS",
        "future_tokens": "SENSITIVE_SENTINEL_FUTURE_TOKENS",
    }
    monkeypatch.setenv("YUJ_INSPECTION_CREDENTIAL", sentinels["environment"])
    overlay = tmp_path / "sensitive.toml"
    overlay.write_text(
        """
[server]
api_key = "$ENV:YUJ_INSPECTION_CREDENTIAL"

[server.request_extra]
headers = { Authorization = "SENSITIVE_SENTINEL_HEADER" }

[models.roles.weak]
profile = "_base"
api_key = "SENSITIVE_SENTINEL_ROLE"

[sandbox.env]
set = { LOOKS_SAFE = "SENSITIVE_SENTINEL_FIXED_ENV" }

[future]
access_token = "SENSITIVE_SENTINEL_FUTURE"
apiKey = "SENSITIVE_SENTINEL_FUTURE_CAMEL"
api_keys = "SENSITIVE_SENTINEL_FUTURE_KEYS"
access_tokens = "SENSITIVE_SENTINEL_FUTURE_TOKENS"
""".strip()
    )

    resolved = _resolved_for_public_command(overlays=[overlay])
    assert resolved.config.api_key == sentinels["environment"]
    document = build_inspection_document(resolved, success=True)
    encoded = render_inspection_json(document)

    for sentinel in sentinels.values():
        assert sentinel not in encoded
    assert "YUJ_INSPECTION_CREDENTIAL" in encoded
    entries = _entry_map(document)
    for path in (
        ("server", "api_key"),
        ("server", "request_extra", "headers", "Authorization"),
        ("models", "roles", "weak", "api_key"),
        ("sandbox", "env", "set", "LOOKS_SAFE"),
        ("future", "access_token"),
        ("future", "apiKey"),
        ("future", "api_keys"),
        ("future", "access_tokens"),
    ):
        assert entries[path]["redacted"] is True
        assert entries[path]["value"] == "<redacted>"
    assert entries[("server", "api_key")]["environment_variable"] == (
        "YUJ_INSPECTION_CREDENTIAL"
    )


def test_environment_profile_value_is_redacted_from_derived_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    profile = "qwen38-27b"
    monkeypatch.setenv("YUJ_INSPECTION_PROFILE", profile)
    overlay = tmp_path / "environment-profile.toml"
    overlay.write_text(
        '[model]\nprofile_name = "$ENV:YUJ_INSPECTION_PROFILE"\n'
    )

    rc = main([
        "config", "--json", "--no-treatment",
        "--config", str(overlay),
        "--config", str(_no_sandbox_overlay(tmp_path)),
    ])

    encoded = capsys.readouterr().out
    document = json.loads(encoded)
    assert rc == 0
    assert profile not in encoded
    assert document["references"]["profile"] == {
        "requested": "<redacted>",
        "resolved": "<redacted>",
    }
    entry = _entry_map(document)[("model", "profile_name")]
    assert entry["environment_variable"] == "YUJ_INSPECTION_PROFILE"
    assert entry["value"] == "<redacted>"


def test_json_is_versioned_deterministic_and_every_entry_has_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    first = build_inspection_document(_resolved_for_public_command(), success=True)
    second = build_inspection_document(_resolved_for_public_command(), success=True)

    assert render_inspection_json(first) == render_inspection_json(second)
    assert first["schema"] == "yuj.config-inspection"
    assert first["schema_version"] == 1
    assert first["success"] is True
    assert first["status"] == "ok"
    assert first["diagnostics"] == []
    paths = [tuple(item["path_components"]) for item in first["settings"]]
    assert paths == sorted(paths)
    assert all(item["source_layer"] for item in first["settings"])
    assert all(isinstance(item["redacted"], bool) for item in first["settings"])


def test_public_config_command_never_constructs_a_client_or_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    with (
        patch("scripts.llm_assist.__main__._make_client") as make_client,
        patch("scripts.llm_assist.__main__.resolve_served_model") as resolve_model,
        patch("scripts.llm_assist.__main__.SessionStore") as session_store,
        patch("openai.OpenAI") as openai_client,
    ):
        rc = main([
            "config", "--json", "--no-treatment",
            "--config", str(_no_sandbox_overlay(tmp_path)),
        ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["success"] is True
    make_client.assert_not_called()
    resolve_model.assert_not_called()
    session_store.assert_not_called()
    openai_client.assert_not_called()


def test_public_cli_model_and_service_overrides_report_command_line_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    sentinel = "SENSITIVE_SENTINEL_CLI_KEY"
    monkeypatch.setenv("YUJ_INSPECTION_CLI_KEY", sentinel)

    rc = main(
        [
            "config",
            "--json",
            "--no-treatment",
            "--provider",
            "custom",
            "--base-url",
            "https://service.invalid/v1",
            "--api-key-env",
            "YUJ_INSPECTION_CLI_KEY",
            "--model",
            "cli-model",
            "--config",
            str(_no_sandbox_overlay(tmp_path)),
        ]
    )

    encoded = capsys.readouterr().out
    document = json.loads(encoded)
    entries = _entry_map(document)
    assert rc == 0
    assert sentinel not in encoded
    for path in (
        ("server", "provider"),
        ("server", "base_url"),
        ("server", "api_key"),
        ("model", "name"),
    ):
        assert entries[path]["source_layer"] == "command-line"
    assert entries[("server", "api_key")]["redacted"] is True
    assert entries[("server", "api_key")]["environment_variable"] == (
        "YUJ_INSPECTION_CLI_KEY"
    )


@pytest.mark.parametrize(
    ("kind", "body", "extra_args", "message"),
    [
        ("malformed", "[loop\nmax_turns = 1", (), "malformed TOML"),
        (
            "incompatible",
            "[tools]\nlazy_loading_enabled = true\nexec_cell_enabled = true\n",
            (),
            "cannot be enabled together",
        ),
        ("invalid-type", '[model]\ncontext_size = "large"\n', (), "context_size"),
        (
            "invalid-boolean",
            '[tools.run_tests]\nstructured_output = "yes"\n',
            (),
            "tools_run_tests_structured_output",
        ),
        (
            "invalid-table",
            '[server]\nrequest_extra = "not-a-table"\n',
            (),
            "server.request_extra",
        ),
        ("invalid-range", "[loop]\nmax_turns = 0\n", (), "max_turns"),
        (
            "missing-environment",
            '[server]\napi_key = "$ENV:YUJ_INSPECTION_MISSING_KEY"\n',
            (),
            "YUJ_INSPECTION_MISSING_KEY",
        ),
        (
            "profile",
            '[model]\nprofile_name = "profile-that-does-not-exist"\n',
            (),
            "profile-that-does-not-exist",
        ),
        ("agent", "", ("--agent", "agent-that-does-not-exist"), "unknown subagent"),
    ],
)
def test_invalid_inputs_return_actionable_versioned_json_and_nonzero_status(
    kind: str,
    body: str,
    extra_args: tuple[str, ...],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    args = ["config", "--json", "--no-treatment"]
    if body:
        overlay = tmp_path / f"{kind}.toml"
        overlay.write_text(body)
        args.extend(("--config", str(overlay)))
    args.extend(extra_args)

    rc = main(args)

    payload = json.loads(capsys.readouterr().out)
    assert rc != 0
    assert payload["schema"] == "yuj.config-inspection"
    assert payload["schema_version"] == 1
    assert payload["success"] is False
    assert payload["status"] == "error"
    assert payload["diagnostics"][0]["level"] == "error"
    assert message in payload["diagnostics"][0]["message"]


@pytest.mark.parametrize(
    "body",
    [
        '[loop]\nplan_mode = "$ENV:YUJ_INSPECTION_FAILURE_SECRET"\n',
        '[model]\nprofile_name = "$ENV:YUJ_INSPECTION_FAILURE_SECRET"\n',
    ],
)
def test_environment_values_are_redacted_from_failure_diagnostics(
    body: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    sentinel = "SENSITIVE_SENTINEL_FAILURE_DIAGNOSTIC"
    monkeypatch.setenv("YUJ_INSPECTION_FAILURE_SECRET", sentinel)
    overlay = tmp_path / "environment-failure.toml"
    overlay.write_text(body)

    rc = main(
        ["config", "--json", "--no-treatment", "--config", str(overlay)]
    )

    encoded = capsys.readouterr().out
    payload = json.loads(encoded)
    assert rc == 1
    assert payload["success"] is False
    assert sentinel not in encoded
    assert "<redacted>" in payload["diagnostics"][0]["message"]


def test_missing_overlay_is_nonzero_and_does_not_disclose_host_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    missing = tmp_path / "missing.toml"

    rc = main(["config", "--json", "--config", str(missing)])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "--config[1]" in payload["diagnostics"][0]["message"]
    assert str(tmp_path) not in json.dumps(payload)


@pytest.mark.parametrize(
    ("target", "label"),
    [
        ("defaults", "config.toml"),
        ("machine-local", "config.local.toml"),
    ],
)
def test_malformed_startup_file_is_reported_after_fresh_import(
    target: str,
    label: str,
    tmp_path: Path,
) -> None:
    default = tmp_path / "config.toml"
    if target == "defaults":
        default.write_text("[server\napi_key = 1\n")
    else:
        default.write_text((PROJECT_ROOT / "config.toml").read_text())
        (tmp_path / "config.local.toml").write_text("[server\napi_key = 1\n")
    environment = os.environ.copy()
    environment["YUJ_CONFIG"] = str(default)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.llm_assist",
            "config",
            "--json",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(process.stdout)
    assert process.returncode == 1
    assert payload["schema"] == "yuj.config-inspection"
    assert payload["success"] is False
    assert f"{label}: malformed TOML" in (
        payload["diagnostics"][0]["message"]
    )
    assert str(tmp_path) not in process.stdout


def test_human_output_is_readable_and_reports_redaction_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)

    rc = main([
        "config", "--no-treatment",
        "--config", str(_no_sandbox_overlay(tmp_path)),
    ])

    output = capsys.readouterr().out
    assert rc == 0
    assert output.startswith("Yuj configuration: valid\n")
    assert "Layers (low to high):" in output
    assert "Settings:" in output
    assert "server.api_key = <redacted>" in output
    assert "checked-in-defaults" in output
    assert "Diagnostics: none" in output


def _documented_toml_paths(document: str) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for body in re.findall(r"```toml\n(.*?)```", document, flags=re.DOTALL):
        try:
            parsed = tomllib.loads(body)
        except tomllib.TOMLDecodeError:
            continue
        paths.update(_flatten_paths(parsed))
    return paths


def test_root_and_documented_public_settings_have_mechanical_inspection_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    root_data = tomllib.loads((PROJECT_ROOT / "config.toml").read_text())
    root_paths = set(_flatten_paths(root_data))
    document = build_inspection_document(_resolved_for_public_command(), success=True)
    inspected_paths = {
        tuple(entry["path_components"])
        for entry in document["settings"]
    }

    assert len(root_paths) >= 275
    assert root_paths <= inspected_paths

    configuration_doc = PROJECT_ROOT / "docs/configuration.md"
    if not configuration_doc.exists():
        # This guide is staging-owned. The source test still checks root
        # settings internally and checks documented settings after extraction.
        return

    documented = _documented_toml_paths(configuration_doc.read_text())
    assert documented

    def covered(path: tuple[str, ...]) -> bool:
        return any(
            candidate == path
            or candidate == path[: len(candidate)]
            or path == candidate[: len(path)]
            for candidate in inspected_paths
        )

    assert not sorted(path for path in documented if not covered(path))


def test_root_help_discovers_config_inspection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "config" in capsys.readouterr().out


TRANSFORMATION_KEYS = {
    "output_cleanup_and_normalization",
    "command_rewrites",
    "task_format_command_output_handling",
    "forbidden_command_replacement",
    "preflight_reclip",
    "halflife_context",
    "detector_and_interventions",
    "detector_activated_guardrails",
}


def test_transformations_file_is_the_complete_true_control_vector() -> None:
    path = PROJECT_ROOT / "configs/transformations.toml"
    values = tomllib.loads(path.read_text())["transformations"]

    assert set(values) == TRANSFORMATION_KEYS
    assert all(value is False for value in values.values())

    cfg = load_config(user_config=path)
    assert cfg.transformations_explicit is True
    assert cfg.output_cleanup_and_normalization is False
    assert cfg.bash_transforms_universal_enabled is False
    assert cfg.bash_transforms_task_format_enabled is False
    assert cfg.bash_quirks_forbidden_enabled is False
    assert cfg.preflight_reclip_enabled is False
    assert cfg.halflife_context is False
    assert cfg.adaptive_control_enabled is False
    assert cfg.llm_hurdle_detector_enabled is False
    assert cfg.detector_activated_guardrails is False
    assert cfg.post_mutation_verification_gate_after == 0

    from scripts.llm_solver.harness._loop.session_io import (
        _load_bash_transforms,
    )

    assert _load_bash_transforms(cfg, force_load_all=True) == (
        None,
        None,
        None,
        None,
        None,
        None,
    )


def test_all_true_transformations_override_plain_arm_gates(tmp_path: Path) -> None:
    overlay = tmp_path / "all-true.toml"
    body = "\n".join(f"{key} = true" for key in sorted(TRANSFORMATION_KEYS))
    overlay.write_text(f"[transformations]\n{body}\n")
    plain = PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"

    cfg = load_config(user_config=[plain, overlay])

    assert cfg.output_cleanup_and_normalization is True
    assert cfg.bash_transforms_universal_enabled is True
    assert cfg.bash_transforms_task_format_enabled is True
    assert cfg.bash_quirks_forbidden_enabled is True
    assert cfg.preflight_reclip_enabled is True
    assert cfg.halflife_context is True
    assert cfg.adaptive_control_enabled is True
    assert cfg.llm_hurdle_detector_enabled is True
    assert cfg.detector_activated_guardrails is True
    assert cfg.post_mutation_verification_gate_after > 0


@pytest.mark.parametrize("enabled", sorted(TRANSFORMATION_KEYS))
def test_each_transformation_switch_can_be_selected_alone(
    enabled: str,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / f"only-{enabled}.toml"
    values = {key: key == enabled for key in TRANSFORMATION_KEYS}
    body = "\n".join(
        f"{key} = {str(value).lower()}"
        for key, value in sorted(values.items())
    )
    overlay.write_text(f"[transformations]\n{body}\n")
    treatment = PROJECT_ROOT / "configs/regimes/treatment.toml"

    cfg = load_config(user_config=[treatment, overlay])

    assert dump_transformations(cfg) == values
    assert cfg.strip_ansi is values["output_cleanup_and_normalization"]
    assert cfg.collapse_blank_lines is values[
        "output_cleanup_and_normalization"
    ]
    assert cfg.search_pagination_enabled is values[
        "output_cleanup_and_normalization"
    ]
    assert cfg.bash_transforms_universal_enabled is values["command_rewrites"]
    assert cfg.bash_transforms_task_format_enabled is values[
        "task_format_command_output_handling"
    ]
    assert cfg.bash_quirks_forbidden_enabled is values[
        "forbidden_command_replacement"
    ]
    assert cfg.preflight_reclip_enabled is values["preflight_reclip"]
    assert cfg.adaptive_control_enabled is values[
        "detector_and_interventions"
    ]
    assert cfg.llm_hurdle_detector_enabled is values[
        "detector_and_interventions"
    ]
    assert (
        cfg.post_mutation_verification_gate_after > 0
    ) is values["detector_activated_guardrails"]


def test_current_paper_arm_transformation_vectors_match_the_file_comments() -> None:
    control = load_config(
        user_config=(
            PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"
        )
    )
    treatment = replace(
        load_config(user_config=PROJECT_ROOT / "configs/regimes/treatment.toml"),
        halflife_context=True,
    )

    assert list(dump_transformations(control).values()) == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(dump_transformations(treatment).values())


def test_halflife_switch_owns_context_selection(tmp_path: Path) -> None:
    overlay = tmp_path / "halflife.toml"
    values = {
        key: key == "halflife_context"
        for key in TRANSFORMATION_KEYS
    }
    body = "\n".join(
        f"{key} = {str(value).lower()}" for key, value in sorted(values.items())
    )
    overlay.write_text(f"[transformations]\n{body}\n")
    cfg = load_config(user_config=overlay)

    assert resolve_transformation_context_mode(cfg, "full") == "halflife"
    with pytest.raises(ValueError, match="halflife_context=true"):
        resolve_transformation_context_mode(
            cfg,
            "full",
            requested_explicitly=True,
        )


def test_public_cli_uses_the_explicit_halflife_switch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transformations = PROJECT_ROOT / "configs/transformations.toml"

    rc = main(["config", "--json", "--config", str(transformations)])

    document = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert document["selection"]["context_mode"] == "full"
    assert document["selection"]["context_source"] == "transformations"


def test_public_cli_rejects_context_that_conflicts_with_the_switch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transformations = PROJECT_ROOT / "configs/transformations.toml"

    rc = main([
        "config",
        "--json",
        "--config",
        str(transformations),
        "--context",
        "halflife",
    ])

    document = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "transformations.halflife_context=false" in str(
        document["diagnostics"]
    )


@pytest.mark.parametrize(
    "table",
    [
        "[transformations]\ncommand_rewrites = false\n",
        "\n".join(
            [
                "[transformations]",
                *(f"{key} = false" for key in sorted(TRANSFORMATION_KEYS)),
                "ninth_switch = false",
            ]
        ),
        "\n".join(
            [
                "[transformations]",
                *(
                    f'{key} = "false"' if key == "command_rewrites" else f"{key} = false"
                    for key in sorted(TRANSFORMATION_KEYS)
                ),
            ]
        ),
    ],
)
def test_transformations_table_rejects_partial_unknown_or_non_boolean_values(
    table: str,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "invalid-transformations.toml"
    overlay.write_text(table + "\n")

    with pytest.raises((TypeError, ValueError), match="transformations"):
        load_config(user_config=overlay)
