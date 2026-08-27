"""Safe persistent configuration editing through the public CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_solver import config as config_module
from scripts.llm_solver.config_edit import (
    ConfigEditError,
    EditDestination,
    _atomic_save,
    edit_toml_text,
)


def _local_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(path))
    monkeypatch.setattr(config_module, "_LOCAL_CONFIG", path)
    return path


def _entry(payload: dict, setting: str) -> dict:
    return next(row for row in payload["settings"] if row["path"] == setting)


def test_surgical_edit_preserves_comments_spacing_and_other_settings() -> None:
    original = (
        "# owner comment\n"
        "[loop]\n"
        "max_turns  =  20  # retain this note\n"
        "max_retries = 2\n"
        "\n"
        "[model]\n"
        "name = \"local\"\n"
    )

    changed = edit_toml_text(
        original,
        path=("loop", "max_turns"),
        value=30,
    )

    assert changed == original.replace(
        "max_turns  =  20  # retain this note",
        "max_turns  =  30  # retain this note",
    )


def test_surgical_edit_replaces_multiline_value_only() -> None:
    original = (
        "[state]\n"
        "ignore_file_names = [\n"
        "  \".first\",\n"
        "  \".second\",\n"
        "]\n"
        "# unrelated\n"
        "ignore_file_enabled = true\n"
    )

    changed = edit_toml_text(
        original,
        path=("state", "ignore_file_names"),
        value=[".one"],
    )

    assert changed == (
        "[state]\n"
        "ignore_file_names = [\".one\"]\n"
        "# unrelated\n"
        "ignore_file_enabled = true\n"
    )


def test_preview_validates_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local_config(monkeypatch, tmp_path)

    rc = main([
        "config",
        "--json",
        "--no-treatment",
        "--set",
        "tools.terminal_max_input_chars",
        "12000",
        "--layer",
        "machine-local",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["schema"] == "yuj.config-edit"
    assert payload["status"] == "preview"
    assert payload["saved"] is False
    assert payload["destination"]["before"] == "<unset>"
    assert payload["destination"]["after"] == 12000
    assert payload["new_effective"]["source_layer"] == "machine-local"
    assert not local.exists()


def test_overlay_apply_is_atomic_and_inspection_reports_saved_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local_config(monkeypatch, tmp_path)
    overlay = tmp_path / "project.toml"
    overlay.write_text(
        "# keep me\n"
        "[tools]\n"
        "terminal_max_input_chars  =  16384  # operator limit\n"
        "terminal_enabled = false\n"
    )

    rc = main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(overlay),
        "--set",
        "tools.terminal_max_input_chars",
        "12000",
        "--layer",
        "overlay-1",
        "--apply",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "applied"
    assert payload["saved"] is True
    assert payload["new_effective"] == {
        "environment_variable": None,
        "source_layer": "overlay-1",
        "value": 12000,
    }
    assert overlay.read_text() == (
        "# keep me\n"
        "[tools]\n"
        "terminal_max_input_chars  =  12000  # operator limit\n"
        "terminal_enabled = false\n"
    )

    inspect_rc = main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(overlay),
    ])
    inspected = json.loads(capsys.readouterr().out)
    assert inspect_rc == 0
    entry = _entry(inspected, "tools.terminal_max_input_chars")
    assert entry["value"] == 12000
    assert entry["source_layer"] == "overlay-1"


def test_unset_reveals_lower_precedence_value_without_reformatting_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local_config(monkeypatch, tmp_path)
    overlay = tmp_path / "remove.toml"
    overlay.write_text(
        "# before\n"
        "[tools]\n"
        "terminal_max_input_chars = 12000\n"
        "terminal_enabled = false\n"
    )

    rc = main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(overlay),
        "--unset",
        "tools.terminal_max_input_chars",
        "--layer",
        "overlay-1",
        "--apply",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["destination"]["after"] == "<unset>"
    assert payload["new_effective"]["value"] == 16384
    assert payload["new_effective"]["source_layer"] == "checked-in-defaults"
    assert overlay.read_text() == (
        "# before\n"
        "[tools]\n"
        "terminal_enabled = false\n"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--set", "not.a.setting", "1", "--layer", "machine-local"],
            "unknown or non-editable setting",
        ),
        (
            [
                "--set",
                "tools.terminal_max_input_chars",
                "false",
                "--layer",
                "machine-local",
            ],
            "wrong type",
        ),
        (
            [
                "--set",
                "tools.terminal_max_input_chars",
                "12000",
                "--layer",
                "base",
            ],
            "read-only",
        ),
    ],
)
def test_invalid_key_type_and_layer_do_not_create_destination(
    arguments: list[str],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local_config(monkeypatch, tmp_path)

    rc = main(["config", "--json", "--no-treatment", *arguments, "--apply"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["schema"] == "yuj.config-edit"
    assert message in payload["diagnostics"][0]["message"]
    assert not local.exists()


def test_invalid_resolved_configuration_is_not_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local_config(monkeypatch, tmp_path)
    local.write_text("# unchanged\n")

    rc = main([
        "config",
        "--json",
        "--no-treatment",
        "--set",
        "formatter.enabled",
        "true",
        "--layer",
        "machine-local",
        "--apply",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "requires at least one formatter" in payload["diagnostics"][0]["message"]
    assert local.read_text() == "# unchanged\n"


def test_secret_literal_is_refused_but_environment_reference_is_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local_config(monkeypatch, tmp_path)

    refused = main([
        "config",
        "--json",
        "--no-treatment",
        "--set",
        "server.api_key",
        "literal-secret",
        "--layer",
        "machine-local",
        "--apply",
    ])
    refused_output = capsys.readouterr().out

    assert refused == 1
    assert "literal-secret" not in refused_output
    assert not local.exists()

    monkeypatch.setenv("YUJ_CONFIG_EDIT_KEY", "resolved-secret-value")
    accepted = main([
        "config",
        "--json",
        "--no-treatment",
        "--set",
        "server.api_key",
        "$ENV:YUJ_CONFIG_EDIT_KEY",
        "--layer",
        "machine-local",
        "--apply",
    ])
    accepted_output = capsys.readouterr().out
    payload = json.loads(accepted_output)

    assert accepted == 0
    assert payload["new_effective"]["value"] == "<redacted>"
    assert payload["new_effective"]["environment_variable"] == (
        "YUJ_CONFIG_EDIT_KEY"
    )
    assert "resolved-secret-value" not in accepted_output
    assert "resolved-secret-value" not in local.read_text()
    assert '$ENV:YUJ_CONFIG_EDIT_KEY' in local.read_text()
    assert local.stat().st_mode & 0o777 == 0o600


def test_layer_and_apply_without_an_edit_fail_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["config", "--json", "--layer", "machine-local"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["schema"] == "yuj.config-edit"
    assert "require --set or --unset" in payload["diagnostics"][0]["message"]


def test_remove_missing_destination_setting_is_rejected() -> None:
    with pytest.raises(ConfigEditError, match="not present"):
        edit_toml_text(
            "[loop]\nmax_turns = 20\n",
            path=("loop", "max_retries"),
            remove=True,
        )


def test_read_only_and_symbolic_link_layers_are_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _local_config(monkeypatch, tmp_path)
    target = tmp_path / "target.toml"
    target.write_text("[tools]\nterminal_max_input_chars = 16384\n")
    target.chmod(0o444)

    read_only = main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(target),
        "--set",
        "tools.terminal_max_input_chars",
        "12000",
        "--layer",
        "overlay-1",
        "--apply",
    ])
    read_only_payload = json.loads(capsys.readouterr().out)

    assert read_only == 1
    assert "not writable" in read_only_payload["diagnostics"][0]["message"]
    assert "16384" in target.read_text()

    target.chmod(0o644)
    link = tmp_path / "link.toml"
    link.symlink_to(target)
    symbolic = main([
        "config",
        "--json",
        "--no-treatment",
        "--config",
        str(link),
        "--set",
        "tools.terminal_max_input_chars",
        "12000",
        "--layer",
        "overlay-1",
        "--apply",
    ])
    symbolic_payload = json.loads(capsys.readouterr().out)

    assert symbolic == 1
    assert "symbolic link" in symbolic_payload["diagnostics"][0]["message"]
    assert "16384" in target.read_text()


def test_conflicting_edit_fails_without_overwriting_new_bytes(tmp_path: Path) -> None:
    path = tmp_path / "layer.toml"
    path.write_bytes(b"before\n")
    destination = EditDestination(
        layer_id="overlay-1",
        label="--config[1]",
        path=path,
        user_layer_index=1,
    )
    path.write_bytes(b"changed elsewhere\n")

    with pytest.raises(ConfigEditError, match="changed after preview"):
        _atomic_save(
            destination,
            expected=b"before\n",
            proposed=b"proposed\n",
        )

    assert path.read_bytes() == b"changed elsewhere\n"
