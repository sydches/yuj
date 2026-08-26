"""Acceptance contract for explicit sandbox selection and provenance."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.runner import (
    _record_sandbox_provenance,
    create_session,
)
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver import config as config_module
from scripts.llm_solver.config import (
    ConfigLayerSpec,
    dump_config,
    load_config,
    resolve_config,
)
from scripts.llm_solver._main_helpers import _build_run_metadata
from scripts.llm_solver.harness._loop._driver_setup import (
    compute_runtime_envelope_fields,
)
from scripts.llm_solver.harness.hooks import HookConfigurationError
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
from scripts.llm_solver.harness.process_manager import (
    build_background_sandbox_argv,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.tool_policy import PermissionPolicy
from scripts.llm_solver.harness.sandbox.policy import (
    SandboxResolutionError,
    bind_sandbox_resolution,
    probe_sandbox_capabilities,
    preflight_sandbox,
    resolve_sandbox_selection,
    sandbox_execution_kwargs,
)
from scripts.llm_solver.harness.sandbox.container_backend import (
    ContainerBackendError,
)


def _without_machine_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        config_module,
        "_LOCAL_CONFIG",
        tmp_path / "absent-config.local.toml",
    )


def _capabilities(
    system: str,
    installed: set[str],
):
    return probe_sandbox_capabilities(
        bwrap_bin="/opt/yuj/bin/bwrap",
        platform_name=system,
        which=lambda name: (
            f"/opt/yuj/bin/{name}" if name in installed else None
        ),
        is_file=lambda path: Path(path).name in installed,
    )


@pytest.mark.parametrize(
    ("system", "installed", "supported", "resolved"),
    [
        (
            "Linux",
            {"bwrap", "docker", "podman"},
            ("bwrap", "docker", "podman"),
            "bwrap",
        ),
        ("Darwin", {"docker", "podman"}, ("docker", "podman"), "docker"),
    ],
)
def test_auto_resolution_uses_platform_order_without_selecting_none(
    system: str,
    installed: set[str],
    supported: tuple[str, ...],
    resolved: str,
) -> None:
    capabilities = _capabilities(system, installed)

    resolution = resolve_sandbox_selection("auto", capabilities)

    assert capabilities.supported == supported
    assert resolution.selected == "auto"
    assert resolution.resolved == resolved
    assert resolution.resolved != "none"
    assert resolution.explicit_unsandboxed is False


def test_explicit_none_is_the_only_unsandboxed_selection() -> None:
    capabilities = _capabilities("Linux", set())

    resolution = resolve_sandbox_selection("none", capabilities)

    assert resolution.selected == resolution.resolved == "none"
    assert resolution.explicit_unsandboxed is True


def test_auto_without_an_available_sandbox_fails_closed() -> None:
    capabilities = _capabilities("Linux", set())

    with pytest.raises(
        SandboxResolutionError,
        match="auto.*no installed supported sandbox backend",
    ):
        resolve_sandbox_selection("auto", capabilities)


def test_named_backend_is_exact_and_never_substituted() -> None:
    capabilities = _capabilities("Linux", {"podman"})

    with pytest.raises(
        SandboxResolutionError,
        match="selected sandbox backend 'docker'.*not installed",
    ):
        resolve_sandbox_selection("docker", capabilities)
    assert resolve_sandbox_selection("podman", capabilities).resolved == "podman"


def test_named_bwrap_is_rejected_on_an_incompatible_platform() -> None:
    capabilities = _capabilities("Darwin", {"bwrap", "docker"})

    with pytest.raises(
        SandboxResolutionError,
        match="selected sandbox backend 'bwrap'.*not supported.*macos",
    ):
        resolve_sandbox_selection("bwrap", capabilities)


def test_native_windows_has_no_same_path_sandbox_backend() -> None:
    capabilities = _capabilities("Windows", {"docker", "podman"})

    assert capabilities.supported == ()
    assert capabilities.installed == ()
    assert capabilities.available == ()
    assert capabilities.unavailable == ("bwrap", "docker", "podman")
    with pytest.raises(SandboxResolutionError, match="auto.*supported=none"):
        resolve_sandbox_selection("auto", capabilities)
    with pytest.raises(
        SandboxResolutionError,
        match="'podman'.*not supported.*windows",
    ):
        resolve_sandbox_selection("podman", capabilities)


def test_auto_uses_the_first_operational_installed_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _capabilities("Linux", {"bwrap", "docker"})
    cfg = make_config(
        sandbox_backend="auto",
        sandbox_bash=True,
        sandbox_required=True,
        sandbox_container_image="local/yuj:sealed",
    )
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox._preflight.bwrap_preflight",
        lambda _binary: (False, "namespace unavailable"),
    )
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend."
        "ContainerBackend.image_digest",
        lambda self, runtime_bin: "sha256:" + ("a" * 64),
    )

    resolution = preflight_sandbox(cfg, capabilities=capabilities)

    assert resolution.selected == "auto"
    assert resolution.resolved == "docker"
    assert resolution.engaged is True


def test_auto_reports_all_operational_failures_without_none_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = _capabilities("Linux", {"bwrap", "docker"})
    cfg = make_config(
        sandbox_backend="auto",
        sandbox_bash=True,
        sandbox_required=True,
        sandbox_container_image="local/yuj:sealed",
    )
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox._preflight.bwrap_preflight",
        lambda _binary: (False, "namespace unavailable"),
    )
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend."
        "ContainerBackend.image_digest",
        lambda self, runtime_bin: (_ for _ in ()).throw(
            ContainerBackendError("runtime unavailable")
        ),
    )

    with pytest.raises(
        SandboxResolutionError,
        match="auto.*no operational sandbox backend.*bwrap.*docker.*never",
    ):
        preflight_sandbox(cfg, capabilities=capabilities)


def test_config_accepts_all_canonical_choices_and_preserves_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    for choice in ("none", "auto", "bwrap", "docker", "podman"):
        overlay = tmp_path / f"{choice}.toml"
        overlay.write_text(
            "[sandbox]\n"
            f'backend = "{choice}"\n'
            + ('container_image = "local/yuj:sealed"\n' if choice in {"docker", "podman"} else "")
        )
        resolved = resolve_config(
            user_config=overlay,
            layer_specs=[
                ConfigLayerSpec(
                    overlay, "choice", "overlay", "sandbox choice",
                )
            ],
        )

        assert resolved.config.sandbox_backend == choice
        assert resolved.data["sandbox"]["backend"] == choice
        assert resolved.provenance[("sandbox", "backend")].layer_id == "choice"
        assert resolved.config.sandbox_bash is (choice != "none")
        assert resolved.config.sandbox_required is (choice != "none")


def test_default_configuration_preserves_linux_bwrap_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)

    cfg = load_config()

    assert cfg.sandbox_backend == "bwrap"
    assert cfg.sandbox_bash is True
    assert cfg.sandbox_required is True


def test_legacy_settings_migrate_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    disabled = tmp_path / "disabled.toml"
    disabled.write_text(
        "[tools]\n"
        "sandbox_bash = false\n"
        "sandbox_required = false\n"
    )
    required = tmp_path / "required.toml"
    required.write_text(
        "[tools]\n"
        "sandbox_bash = true\n"
        "sandbox_required = true\n"
    )
    container = tmp_path / "container.toml"
    container.write_text(
        "[sandbox]\n"
        'backend = "container"\n'
        'container_runtime = "podman"\n'
        'container_image = "local/yuj:sealed"\n'
    )

    assert load_config(disabled).sandbox_backend == "none"
    assert load_config(required).sandbox_backend == "bwrap"
    migrated_container = load_config(container)
    assert migrated_container.sandbox_backend == "podman"
    assert migrated_container.sandbox_container_runtime == "podman"


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (
            '[sandbox]\nbackend = "bwrap"\n'
            '\n[tools]\nsandbox_bash = true\nsandbox_required = true\n',
            "bwrap",
        ),
        (
            '[sandbox]\nbackend = "container"\ncontainer_runtime = "podman"\n'
            'container_image = "local/yuj:sealed"\n'
            '\n[tools]\nsandbox_bash = true\nsandbox_required = true\n',
            "podman",
        ),
        (
            '[sandbox]\nbackend = "none"\n'
            '\n[tools]\nsandbox_bash = false\nsandbox_required = false\n',
            "none",
        ),
    ],
)
def test_consistent_same_layer_legacy_full_configs_migrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
    expected: str,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "old-full-config.toml"
    overlay.write_text(contents)

    resolved = resolve_config(user_config=overlay)
    assert resolved.config.sandbox_backend == expected
    assert "sandbox_bash" not in resolved.data["tools"]
    assert "sandbox_required" not in resolved.data["tools"]


@pytest.mark.parametrize(
    "contents",
    [
        '[sandbox]\nbackend = "bwrap"\n'
        '\n[tools]\nsandbox_bash = false\nsandbox_required = false\n',
        '[sandbox]\nbackend = "none"\n'
        '\n[tools]\nsandbox_bash = true\nsandbox_required = true\n',
    ],
)
def test_contradictory_same_layer_legacy_full_configs_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "contradictory-old-config.toml"
    overlay.write_text(contents)

    with pytest.raises(ValueError, match="sandbox.backend contradicts legacy"):
        load_config(overlay)


@pytest.mark.parametrize(
    ("sandbox_bash", "sandbox_required"),
    [(True, False), (False, True)],
)
def test_legacy_boolean_combinations_that_can_degrade_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sandbox_bash: bool,
    sandbox_required: bool,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "invalid-legacy.toml"
    overlay.write_text(
        "[tools]\n"
        f"sandbox_bash = {str(sandbox_bash).lower()}\n"
        f"sandbox_required = {str(sandbox_required).lower()}\n"
    )

    with pytest.raises(ValueError, match="legacy sandbox settings"):
        load_config(overlay)


def test_one_layer_rejects_contradictory_canonical_and_legacy_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "contradictory.toml"
    overlay.write_text(
        "[sandbox]\n"
        'backend = "none"\n'
        "\n[tools]\n"
        "sandbox_bash = true\n"
        "sandbox_required = true\n"
    )

    with pytest.raises(ValueError, match="sandbox.backend contradicts legacy"):
        load_config(overlay)

    container_overlay = tmp_path / "contradictory-container.toml"
    container_overlay.write_text(
        "[sandbox]\n"
        'backend = "podman"\n'
        'container_runtime = "docker"\n'
        'container_image = "local/yuj:sealed"\n'
    )
    with pytest.raises(ValueError, match="container_runtime is a legacy setting"):
        load_config(container_overlay)


def test_later_canonical_choice_wins_over_fixed_legacy_required_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    fixed = tmp_path / "fixed.toml"
    fixed.write_text(
        "[tools]\n"
        "sandbox_bash = true\n"
        "sandbox_required = true\n"
    )
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[sandbox]\nbackend = \"none\"\n")

    cfg = load_config([fixed, explicit])

    assert cfg.sandbox_backend == "none"
    assert cfg.sandbox_bash is False
    assert cfg.sandbox_required is False


def test_command_line_sandbox_override_has_canonical_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)

    resolved = resolve_config(overrides={"sandbox_backend": "none"})

    assert resolved.config.sandbox_backend == "none"
    assert resolved.data["sandbox"]["backend"] == "none"
    assert (
        resolved.provenance[("sandbox", "backend")].layer_id
        == "command-line"
    )


def test_fixed_measurement_configs_keep_required_bwrap_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "configs/regimes/treatment.toml",
        root / "configs/regimes/baselines/plain_long_solve.toml",
    ):
        cfg = load_config(path)
        assert cfg.sandbox_backend == "bwrap"
        assert cfg.sandbox_bash is True
        assert cfg.sandbox_required is True


def test_config_inspection_reports_selected_and_resolved_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "none.toml"
    overlay.write_text("[sandbox]\nbackend = \"none\"\n")

    assert assist_main([
        "config", "--json", "--no-treatment", "--config", str(overlay),
    ]) == 0

    document = json.loads(capsys.readouterr().out)
    sandbox = document["selection"]["sandbox"]
    assert sandbox["selected"] == "none"
    assert sandbox["resolved"] == "none"
    assert sandbox["explicit_unsandboxed"] is True
    assert sandbox["supported"]
    assert isinstance(sandbox["installed"], list)
    assert sandbox["available"] == sandbox["installed"]
    assert isinstance(sandbox["unavailable"], list)
    assert "backend_executable" not in sandbox


def test_runtime_envelope_names_explicit_unsandboxed_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    fields = compute_runtime_envelope_fields(
        make_config(
            sandbox_backend="none",
            sandbox_bash=False,
            sandbox_required=False,
        ),
        tmp_path,
    )

    assert fields["sandbox_selected"] == "none"
    assert fields["sandbox_resolved"] == "none"
    assert fields["sandbox_mode"] == "none"
    assert fields["sandbox_engaged"] is False
    assert fields["sandbox_explicit_unsandboxed"] is True


def test_status_reports_saved_unsandboxed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assist_home = tmp_path / "assist"
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(assist_home))
    store = SessionStore()
    record = create_session(
        store,
        cwd=task,
        prompt_text="inspect status",
        prompt_source="inline",
        model="test-model",
        config_paths=[],
        system_prompt_path=None,
        context_mode="full",
    )
    (record.artifact_path / ".trace.jsonl").write_text(json.dumps({
        "event": "runtime_envelope",
        "session": 1,
        "sandbox_mode": "none",
        "sandbox_engaged": False,
        "sandbox_backend": "none",
        "sandbox_selected": "none",
        "sandbox_resolved": "none",
        "sandbox_explicit_unsandboxed": True,
        "container_runtime": None,
        "container_image_digest": None,
    }) + "\n")

    assert assist_main(["status", record.session_id]) == 0
    output = capsys.readouterr().out
    assert "sandbox_selected: none" in output
    assert "sandbox_resolved: none" in output
    assert "sandbox_engaged: no" in output
    assert "sandbox_explicit_unsandboxed: yes" in output


def test_assistant_and_measurement_session_metadata_record_resolution(
    tmp_path: Path,
) -> None:
    capabilities = _capabilities("Linux", set())
    resolution = resolve_sandbox_selection("none", capabilities)
    cfg = bind_sandbox_resolution(
        make_config(
            sandbox_backend="none",
            sandbox_bash=False,
            sandbox_required=False,
        ),
        resolution,
    )
    task = tmp_path / "task"
    task.mkdir()
    store = SessionStore(tmp_path / "assist")
    record = create_session(
        store,
        cwd=task,
        prompt_text="record the boundary",
        prompt_source="inline",
        model="test-model",
        config_paths=[],
        system_prompt_path=None,
        context_mode="full",
    )

    _record_sandbox_provenance(record, resolution)
    assistant_metadata = json.loads(
        (record.artifact_path / "session.json").read_text()
    )
    assert assistant_metadata["sandbox"]["selected"] == "none"
    assert assistant_metadata["sandbox"]["resolved"] == "none"
    assert assistant_metadata["sandbox"]["engaged"] is False
    assert "backend_executable" not in assistant_metadata["sandbox"]

    measurement_metadata = _build_run_metadata(
        run_dir=tmp_path / "measurement",
        cfg=cfg,
        args=SimpleNamespace(
            config=[], context="full", system_prompt=None,
        ),
        overrides={},
        started_at="2026-08-25T00:00:00+00:00",
    )
    assert measurement_metadata["sandbox"]["selected"] == "none"
    assert measurement_metadata["sandbox"]["resolved"] == "none"
    assert measurement_metadata["sandbox"]["engaged"] is False
    assert (
        measurement_metadata["sandbox"]["available"]
        == measurement_metadata["sandbox"]["installed"]
    )
    assert isinstance(measurement_metadata["sandbox"]["unavailable"], list)
    assert "backend_executable" not in measurement_metadata["sandbox"]


@pytest.mark.parametrize(
    ("resolved", "environment"),
    [
        ("bwrap", {"YUJ_CONTAINER": "ambient"}),
        ("none", {"YUJ_CONTAINER": "task-container"}),
        ("ambient", {}),
        ("docker-exec", {"YUJ_CONTAINER": "ambient"}),
        ("docker", {"YUJ_CONTAINER": "task-container"}),
    ],
)
def test_pinned_resolution_rejects_changed_legacy_container_environment(
    resolved: str,
    environment: dict[str, str],
) -> None:
    cfg = make_config(
        sandbox_backend=(
            "bwrap" if resolved in {"ambient", "docker-exec"} else resolved
        ),
        sandbox_resolved_backend=resolved,
        sandbox_bash=resolved != "none",
        sandbox_required=resolved != "none",
    )

    with pytest.raises(
        SandboxResolutionError,
        match="YUJ_CONTAINER changed after sandbox preflight",
    ):
        preflight_sandbox(cfg, environment=environment)


def test_pinned_legacy_container_resolution_accepts_unchanged_environment() -> None:
    cfg = make_config(
        sandbox_backend="bwrap",
        sandbox_resolved_backend="docker-exec",
        sandbox_legacy_container="task-container",
        sandbox_bash=True,
        sandbox_required=True,
    )

    resolution = preflight_sandbox(
        cfg,
        environment={"YUJ_CONTAINER": "task-container"},
    )

    assert resolution.resolved == "docker-exec"
    assert resolution.legacy_container == "task-container"


def test_pinned_legacy_container_rejects_a_different_container_id() -> None:
    cfg = make_config(
        sandbox_backend="bwrap",
        sandbox_resolved_backend="docker-exec",
        sandbox_legacy_container="original-container",
        sandbox_bash=True,
        sandbox_required=True,
    )

    with pytest.raises(
        SandboxResolutionError,
        match="YUJ_CONTAINER changed after sandbox preflight",
    ):
        preflight_sandbox(
            cfg,
            environment={"YUJ_CONTAINER": "replacement-container"},
        )


def test_none_keeps_permission_rules_and_hook_path_guards(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        sandbox_backend="none",
        sandbox_bash=False,
        sandbox_required=False,
        runtime_mode="assistant",
        permissions_rules={"bash": {"*": "deny"}},
        hooks_enabled=True,
        hooks={
            "pre_tool": [{
                "matcher": "bash",
                "command": [str(tmp_path / "task-owned-hook")],
            }],
        },
    )

    decision = PermissionPolicy.from_rule_tables(
        {}, cfg.permissions_rules
    ).evaluate(
        tool_name="bash",
        arguments={"cmd": "printf must-not-run"},
        runtime_mode="assistant",
    )
    assert decision.denied is True
    with pytest.raises(HookConfigurationError, match="inside the task cwd"):
        Session(cfg, MagicMock(), "system", "task", str(tmp_path))


def test_one_resolved_value_drives_foreground_background_and_lsp(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        sandbox_backend="none",
        sandbox_resolved_backend="none",
        sandbox_bash=False,
        sandbox_required=False,
    )

    kwargs = sandbox_execution_kwargs(cfg)
    assert kwargs["sandbox_backend"] == "none"
    assert kwargs["sandbox"] is False
    assert kwargs["sandbox_required"] is False
    assert "foreground" in dispatch(
        "bash",
        {"cmd": "printf foreground"},
        cwd=str(tmp_path),
        cfg=cfg,
    )
    assert build_background_sandbox_argv(
        "printf background",
        cwd=str(tmp_path),
        bwrap_bin=cfg.bwrap_bin,
        **kwargs,
    )[-2:] == ["-c", "printf background"]
    assert build_lsp_sandbox_argv(
        ("language-server", "--stdio"),
        cwd=str(tmp_path),
        bwrap_bin=cfg.bwrap_bin,
        **kwargs,
    ) == ["language-server", "--stdio"]


def test_resolved_container_executable_is_part_of_shared_execution_policy() -> None:
    cfg = make_config(
        sandbox_backend="podman",
        sandbox_resolved_backend="podman",
        sandbox_backend_executable="/opt/yuj/bin/podman",
        sandbox_container_runtime="podman",
        sandbox_container_image="sha256:" + ("a" * 64),
    )

    kwargs = sandbox_execution_kwargs(cfg)

    assert kwargs["sandbox_backend"] == "container"
    assert kwargs["container_runtime"] == "podman"
    assert kwargs["container_runtime_bin"] == "/opt/yuj/bin/podman"
    assert "sandbox_backend_executable" not in dump_config(cfg)
    assert "sandbox_legacy_container" not in dump_config(cfg)


def test_auto_cannot_be_pinned_to_unsandboxed_execution() -> None:
    cfg = make_config(
        sandbox_backend="auto",
        sandbox_resolved_backend="none",
        sandbox_bash=True,
        sandbox_required=True,
    )

    with pytest.raises(SandboxResolutionError, match="contradicts selected"):
        sandbox_execution_kwargs(cfg)


def test_setup_reports_an_explicit_none_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.local.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(config_path))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert assist_main([
        "setup",
        "--provider", "local",
        "--model", "local-model",
        "--sandbox", "none",
        "--force",
    ]) == 0

    assert '[sandbox]\nbackend = "none"\n' in config_path.read_text()
    output = capsys.readouterr().out
    assert "sandbox_available:" in output
    assert "sandbox_unavailable:" in output
    assert "sandbox_selected: none" in output
    assert "sandbox_resolved: none" in output


def test_setup_auto_requires_an_image_when_container_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(tmp_path / "config.local.toml"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "scripts.llm_assist.__main__.probe_sandbox_capabilities",
        lambda **_kwargs: _capabilities("Darwin", {"docker"}),
    )

    with pytest.raises(
        SystemExit,
        match="--sandbox-image is required.*auto resolves to docker",
    ):
        assist_main([
            "setup",
            "--provider", "local",
            "--model", "local-model",
            "--sandbox", "auto",
            "--force",
        ])


def test_doctor_reports_supported_installed_selected_and_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_machine_local(monkeypatch, tmp_path)
    overlay = tmp_path / "none.toml"
    overlay.write_text("[sandbox]\nbackend = \"none\"\n")
    client = MagicMock()
    client.health_check.return_value = [load_config(overlay).model]
    monkeypatch.setattr(
        "scripts.llm_assist.__main__._make_client",
        lambda *args, **kwargs: client,
    )

    assert assist_main(["doctor", "--config", str(overlay)]) == 0

    output = capsys.readouterr().out
    assert "sandbox_supported:" in output
    assert "sandbox_installed:" in output
    assert "sandbox_available:" in output
    assert "sandbox_unavailable:" in output
    assert "sandbox_selected: none" in output
    assert "sandbox_resolved: none" in output
