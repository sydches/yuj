"""Redacted, deterministic support-report contract."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist import support_report as support_module
from scripts.llm_assist.support_report import (
    SupportReportError,
    build_support_report,
    render_support_report,
    write_support_report,
)
from scripts.llm_solver import config as config_module
from scripts.llm_solver.config import ConfigLayerSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAIN_BASE = PROJECT_ROOT / "configs/regimes/baselines/plain_long_solve.toml"


def _without_machine_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )


def _overlay(tmp_path: Path, text: str = "") -> Path:
    path = tmp_path / "support-overlay.toml"
    path.write_text('[sandbox]\nbackend = "none"\n' + text)
    return path


def _specs(overlay: Path) -> list[ConfigLayerSpec]:
    return [
        ConfigLayerSpec(PLAIN_BASE, "base", "base", "plain"),
        ConfigLayerSpec(overlay, "overlay-1", "overlay", "--config[1]"),
    ]


def _build(overlay: Path, **kwargs):
    return build_support_report(
        version="9.8.7",
        specs=_specs(overlay),
        overrides={"runtime_mode": "assistant", "max_sessions": 1},
        **kwargs,
    )


def test_report_is_deterministic_complete_and_contains_no_config_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    secret = "SUPPORT_SECRET_VALUE"
    private_host = "private-host.example.invalid"
    model = "private-model-name"
    environment_name = "YUJ_SUPPORT_SECRET_NAME"
    monkeypatch.setenv(environment_name, secret)
    overlay = _overlay(
        tmp_path,
        (
            f'[server]\napi_key = "$ENV:{environment_name}"\n'
            f'base_url = "https://{private_host}/v1"\n'
            f'[model]\nname = "{model}"\n'
        ),
    )
    called = False

    def forbidden_network(_cfg):
        nonlocal called
        called = True
        raise AssertionError("network check must be opt-in")

    first = _build(overlay, network_check=forbidden_network)
    second = _build(overlay, network_check=forbidden_network)
    encoded = render_support_report(first).decode("utf-8")

    assert first == second
    assert called is False
    assert first["schema"] == "yuj.support-report"
    assert first["schema_version"] == 1
    assert first["sections"]["network"]["status"] == "omitted"
    assert first["sections"]["sandbox"]["resolved"] == "none"
    assert first["privacy"] == {
        "environment_level_only": True,
        "target_repository_read": False,
        "session_store_read": False,
        "uploaded": False,
        "external_issue_opened": False,
        "network_requested": False,
    }
    settings = first["sections"]["configuration"]["settings"]
    assert settings
    assert all("value" not in setting for setting in settings)
    assert all(setting["source_layer"] for setting in settings)
    for private_value in (secret, private_host, model, environment_name):
        assert private_value not in encoded
    assert "network diagnostics" in first["inventory"]["omitted"]
    assert first["checks"] == sorted(
        first["checks"], key=lambda check: check["name"]
    )


def test_network_diagnostics_are_bounded_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _overlay(tmp_path)
    calls = 0

    def check(_cfg):
        nonlocal calls
        calls += 1
        return {
            "model_count": 2,
            "selected_model_listed": True,
            "models": ["PRIVATE_MODEL_A", "PRIVATE_MODEL_B"],
            "credential_id": "PRIVATE_CREDENTIAL_ID",
        }

    report = _build(
        overlay,
        network_requested=True,
        network_check=check,
    )
    encoded = render_support_report(report).decode("utf-8")

    assert calls == 1
    assert report["sections"]["network"] == {
        "status": "ok",
        "requested": True,
        "model_count": 2,
        "selected_model_listed": True,
    }
    assert "PRIVATE_MODEL" not in encoded
    assert "PRIVATE_CREDENTIAL_ID" not in encoded
    assert "network diagnostics" not in report["inventory"]["omitted"]


def test_one_failed_section_keeps_the_other_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _overlay(tmp_path)

    def broken_git():
        raise RuntimeError("PRIVATE_FAILURE_DETAIL")

    monkeypatch.setattr(support_module, "_collect_git", broken_git)
    report = _build(overlay)
    encoded = render_support_report(report).decode("utf-8")

    assert report["sections"]["git"] == {
        "status": "unavailable",
        "error_type": "RuntimeError",
        "summary": "git collection failed",
    }
    assert report["sections"]["installation"]["status"] == "ok"
    assert report["sections"]["configuration"]["status"] == "ok"
    assert report["inventory"]["unavailable"] == ["git"]
    assert "PRIVATE_FAILURE_DETAIL" not in encoded


def test_writer_is_atomic_private_and_requires_explicit_replacement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "support.json"
    document = {"schema": "example", "value": 1}
    payload = render_support_report(document)

    byte_count, digest = write_support_report(
        destination,
        document,
        force=False,
    )

    assert destination.read_bytes() == payload
    assert byte_count == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(SupportReportError, match="already exists"):
        write_support_report(destination, document, force=False)

    replacement = {"schema": "example", "value": 2}
    write_support_report(destination, replacement, force=True)
    assert json.loads(destination.read_text())["value"] == 2

    link = tmp_path / "support-link.json"
    link.symlink_to(destination)
    with pytest.raises(SupportReportError, match="symbolic link"):
        write_support_report(link, document, force=True)
    with pytest.raises(SupportReportError, match="parent directory"):
        write_support_report(
            tmp_path / "missing" / "support.json",
            document,
            force=False,
        )


def test_cli_writes_report_without_model_or_session_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _overlay(tmp_path)
    destination = tmp_path / "cli-support.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default support report must stay local")

    monkeypatch.setattr("scripts.llm_assist.__main__._make_client", forbidden)
    monkeypatch.setattr("scripts.llm_assist.__main__.CredentialStore", forbidden)

    rc = main([
        "support",
        "--output", str(destination),
        "--no-treatment",
        "--config", str(overlay),
    ])

    output = capsys.readouterr().out
    report = json.loads(destination.read_text())
    assert rc == 0
    assert f"support_report: {destination}" in output
    assert "network_requested: false" in output
    assert report["sections"]["network"]["status"] == "omitted"


def test_cli_network_flag_runs_only_the_bounded_health_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = _overlay(tmp_path)
    destination = tmp_path / "network-support.json"

    class Store:
        def active_binding(self):
            return None

    class Client:
        def health_check(self):
            return ["one", "two"]

    monkeypatch.setattr("scripts.llm_assist.__main__.CredentialStore", Store)
    monkeypatch.setattr(
        "scripts.llm_assist.__main__._make_client",
        lambda *_args, **_kwargs: Client(),
    )

    assert main([
        "support",
        "--output", str(destination),
        "--network",
        "--no-treatment",
        "--config", str(overlay),
    ]) == 0

    network = json.loads(destination.read_text())["sections"]["network"]
    assert network["status"] == "ok"
    assert network["requested"] is True
    assert network["model_count"] == 2
    assert set(network) == {
        "status", "requested", "model_count", "selected_model_listed"
    }
