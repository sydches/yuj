import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist import __main__ as cli
from scripts.llm_assist.runner import run_session
from scripts.llm_assist.store import SessionStore
from scripts.llm_assist.trust import (
    WorkspaceTrustError,
    discover_workspace_behavior,
    require_trust_store_outside_workspace,
    save_workspace_trust,
    workspace_trust_state,
)
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness import solve_task


def _workspace_config(**overrides):
    cfg = load_config(resolve_runtime_extensions=False)
    values = {
        "project_docs_enabled": False,
        "imports_enabled": True,
        "skills_enabled": False,
        "injections_enabled": False,
        "stream_rules_enabled": False,
        "state_ignore_file_enabled": False,
        "hooks_enabled": False,
        "hooks": {},
        "compaction_hook": "",
        "lsp_enabled": False,
        "lsp_servers": {},
        "post_edit_checks": [],
        "formatter_enabled": False,
        "formatters": [],
    }
    values.update(overrides)
    return replace(cfg, **values)


def test_manifest_lists_each_repository_behavior_category(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "docs").mkdir()
    (workspace / "docs" / "rules.md").write_text("Imported rule.\n")
    (workspace / "AGENTS.md").write_text("Repository rule.\n@docs/rules.md\n")
    skill = workspace / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\nInstructions.\n"
    )
    (skill / "helper.sh").write_text("#!/bin/sh\nexit 0\n")
    injections = workspace / ".harness" / "injections"
    injections.mkdir(parents=True)
    (injections / "review.md").write_text("Review carefully.\n")
    stream_rules = workspace / ".harness" / "stream_rules"
    stream_rules.mkdir(parents=True)
    (stream_rules / "stop.md").write_text("Stop on this signal.\n")
    (workspace / ".yujignore").write_text("secret.txt\n")
    (workspace / "workspace_hook.py").write_text(
        "def compact(preparation):\n    return None\n"
    )
    overlay = workspace / "yuj.toml"
    overlay.write_text("[prompts]\nproject_docs_enabled = true\n")

    cfg = _workspace_config(
        project_docs_enabled=True,
        skills_enabled=True,
        skills_dirs=(".agents/skills",),
        injections_enabled=True,
        injections_dir=".harness/injections",
        stream_rules_enabled=True,
        stream_rules_dir=".harness/stream_rules",
        state_ignore_file_enabled=True,
        state_ignore_file_names=(".yujignore",),
        hooks_enabled=True,
        hooks={"session_start": {"command": ["/bin/true"]}},
        compaction_hook="workspace_hook:compact",
        lsp_enabled=True,
        lsp_servers={"python": {"command": ["pylsp"]}},
        post_edit_checks=[{"command": ["true"]}],
        formatter_enabled=True,
        formatters=[{
            "name": "example",
            "extensions": [".py"],
            "command": ["formatter", "{path}"],
        }],
    )

    manifest = discover_workspace_behavior(
        cfg,
        workspace=workspace,
        config_paths=[overlay],
    )

    assert set(manifest.categories) == {
        "compaction_hook",
        "configuration",
        "formatter_commands",
        "ignore_policy",
        "injections",
        "language_servers",
        "lifecycle_hooks",
        "post_edit_checks",
        "project_instructions",
        "skills",
        "stream_rules",
    }
    paths = {Path(item.path) for item in manifest.items}
    assert workspace / "AGENTS.md" in paths
    assert workspace / "docs" / "rules.md" in paths
    assert skill / "SKILL.md" in paths
    assert workspace / "workspace_hook.py" in paths
    assert overlay in paths


def test_workspace_trust_persists_until_revoked(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instructions = workspace / "AGENTS.md"
    instructions.write_text("First rule.\n")
    cfg = _workspace_config(project_docs_enabled=True)
    store = SessionStore(tmp_path / "state")

    first = discover_workspace_behavior(cfg, workspace=workspace)
    save_workspace_trust(store, first)
    assert workspace_trust_state(store, first) == "trusted"

    imported = workspace / "more.md"
    imported.write_text("One more rule.\n")
    instructions.write_text("Changed rule.\n@more.md\n")
    changed = discover_workspace_behavior(cfg, workspace=workspace)
    assert changed.digest != first.digest
    assert workspace_trust_state(store, changed) == "trusted"

    assert store.revoke_workspace_trust(workspace) is True
    assert workspace_trust_state(store, changed) == "untrusted"


def test_task_attachment_paths_are_workspace_behavior(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "src").mkdir()
    source = workspace / "src" / "demo.py"
    source.write_text("value = 1\n")
    cfg = _workspace_config()

    manifest = discover_workspace_behavior(
        cfg,
        workspace=workspace,
        task_attachment_paths=[Path("src"), source],
    )

    assert manifest.categories == ("task_attachments",)
    assert [(item.logical_path, item.kind) for item in manifest.items] == [
        ("project/src", "directory"),
        ("project/src/demo.py", "file"),
    ]


def test_noninteractive_cli_requires_trust_and_supports_status_and_revoke(
    tmp_path, monkeypatch, capsys
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Use the repository instructions.\n")
    overlay = workspace / "yuj.toml"
    overlay.write_text("[prompts]\nproject_docs_enabled = true\n")
    state = tmp_path / "state"
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(state))

    argv = [
        "--dry-run",
        "--cwd",
        str(workspace),
        "--config",
        str(overlay),
        "check startup",
    ]
    with patch.object(cli, "_is_interactive", return_value=False), patch.object(
        cli, "preflight_assistant_startup"
    ) as preflight:
        with pytest.raises(SystemExit, match="workspace trust is required"):
            cli.main(argv)
        preflight.assert_not_called()

    with patch.object(cli, "preflight_assistant_startup") as preflight, patch.object(
        cli, "render_startup_preflight", return_value="ready\n"
    ):
        assert cli.main(["--trust-workspace", *argv]) == 0
        assert preflight.call_count == 1
    capsys.readouterr()

    (workspace / "AGENTS.md").write_text("A later repository instruction.\n")
    with patch.object(cli, "preflight_assistant_startup"), patch.object(
        cli, "render_startup_preflight", return_value="ready\n"
    ):
        assert cli.main(argv) == 0

    assert cli.main(["trust", "status", "-C", str(workspace)]) == 0
    status = capsys.readouterr().out
    assert "workspace_trust: recorded" in status
    assert "project_instructions" in status
    assert "persistence: trusted until" in status

    assert cli.main(["trust", "revoke", "-C", str(workspace)]) == 0
    assert "changed: yes" in capsys.readouterr().out
    with patch.object(cli, "_is_interactive", return_value=False):
        with pytest.raises(SystemExit, match="workspace trust is required"):
            cli.main(argv)


def test_repository_compaction_hook_is_not_imported_before_trust(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = tmp_path / "imported.txt"
    module_name = "workspace_trust_hook"
    (workspace / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def compact(preparation):\n"
        "    return None\n"
    )
    overlay = workspace / "yuj.toml"
    overlay.write_text(
        f"[context]\ncompaction_hook = {f'{module_name}:compact'!r}\n"
    )
    monkeypatch.syspath_prepend(str(workspace))
    sys.modules.pop(module_name, None)

    cfg = cli._assistant_config_for_workspace_trust(
        [overlay], requested_model=None
    )
    manifest = discover_workspace_behavior(
        cfg,
        workspace=workspace,
        config_paths=[overlay],
    )
    store = SessionStore(tmp_path / "state")
    assert "compaction_hook" in manifest.categories
    assert not marker.exists()

    with patch.object(cli, "_is_interactive", return_value=False):
        with pytest.raises(SystemExit, match="workspace trust is required"):
            cli._gate_workspace_behavior(manifest, decision=None, store=store)
    assert not marker.exists()

    cli._gate_workspace_behavior(manifest, decision=True, store=store)
    assert not marker.exists()
    load_config(
        user_config=[overlay],
        overrides={"runtime_mode": "assistant", "max_sessions": 1},
    )
    assert marker.read_text() == "imported"
    sys.modules.pop(module_name, None)


def test_solver_startup_guard_runs_before_artifact_creation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    cfg = _workspace_config()

    def reject(_work_dir, _cfg, _system_prompt_file):
        raise WorkspaceTrustError("not trusted")

    with pytest.raises(WorkspaceTrustError, match="not trusted"):
        solve_task(
            workspace,
            cfg,
            object(),
            artifacts_dir=artifacts,
            startup_guard=reject,
        )
    assert not artifacts.exists()


def test_runner_rejects_repository_behavior_without_saved_trust(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Repository instruction.\n")
    overlay = workspace / "yuj.toml"
    overlay.write_text("[prompts]\nproject_docs_enabled = true\n")
    store = SessionStore(tmp_path / "state")
    record = store.create_session(
        cwd=workspace,
        model="qwen3-8b",
        prompt_text="Fix it",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[overlay],
    )

    with pytest.raises(WorkspaceTrustError, match="untrusted"):
        run_session(store, record, resume=False)
    assert store.get_session(record.session_id).status == "created"


def test_trust_store_must_be_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(WorkspaceTrustError, match="outside"):
        require_trust_store_outside_workspace(
            workspace / ".state", workspace
        )


def test_workspace_skill_symlink_fails_closed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = workspace / ".agents" / "skills"
    skills.mkdir(parents=True)
    external = tmp_path / "external-skill"
    external.mkdir()
    (external / "SKILL.md").write_text(
        "---\nname: external-skill\ndescription: External\n---\n"
    )
    (skills / "external-skill").symlink_to(external, target_is_directory=True)
    cfg = _workspace_config(
        skills_enabled=True,
        skills_dirs=(".agents/skills",),
    )

    with pytest.raises(WorkspaceTrustError, match="symbolic link"):
        discover_workspace_behavior(cfg, workspace=workspace)
