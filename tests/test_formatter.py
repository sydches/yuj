from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._tools.apply_patch import apply_patch_tool
from scripts.llm_solver.harness._tools.udiff import udiff_tool
from scripts.llm_solver.harness.savings import close_ledger, open_ledger
from scripts.llm_solver.harness.tools import write


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Formatter Test")
    _git(repo, "config", "user.email", "formatter@example.test")
    (repo / "app.py").write_text("value = 1\n")
    (repo / "other.py").write_text("other = 1\n")
    _git(repo, "add", "app.py", "other.py")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo


def _cfg(command: list[str], **overrides):
    values = {
        "formatter_enabled": True,
        "formatter_timeout": 5,
        "formatter_max_output_chars": 4000,
        "formatters": [{
            "name": "test-format",
            "extensions": [".py"],
            "command": command,
            "root_markers": [".git"],
        }],
        "sandbox_bash": False,
        "sandbox_required": False,
    }
    values.update(overrides)
    return make_config(**values)


def _python_formatter(source: str) -> list[str]:
    return [sys.executable, "-c", source, "{path}"]


def test_formatter_is_explicit_and_reports_attributed_changes(tmp_path: Path):
    repo = _repo(tmp_path)
    ledger_path = tmp_path / "savings.jsonl"
    cfg = _cfg(_python_formatter(
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.write_text(p.read_text().replace('  ', ' '))"
    ))

    open_ledger(ledger_path)
    try:
        result = write("app.py", "value  =  2\n", cwd=str(repo), cfg=cfg)
    finally:
        close_ledger()

    assert result.startswith("OK: wrote")
    assert '<formatter_run name="test-format" status="changed"' in result
    assert 'command_status="passed" exit_code="0"' in result
    assert "Pre-formatter SHA-256:" in result
    assert "Post-formatter SHA-256:" in result
    assert "Formatter changed paths:\n- app.py" in result
    assert (repo / "app.py").read_text() == "value = 2\n"
    assert "+value = 2" in _git(repo, "diff", "--", "app.py")

    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    event = next(row for row in events if row["bucket"] == "formatter_run")
    assert event["mechanism"] == "test-format"
    assert event["ctx"]["status"] == "changed"
    assert event["ctx"]["changed_paths"] == ["app.py"]
    assert len(event["ctx"]["before_sha256"]) == 64
    assert len(event["ctx"]["after_sha256"]) == 64


def test_disabled_and_unsupported_formatters_preserve_current_behavior(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    disabled = make_config(
        formatter_enabled=False,
        sandbox_bash=False,
        sandbox_required=False,
    )

    result = write("app.py", "value  =  2\n", cwd=str(repo), cfg=disabled)

    assert result == "OK: wrote 12 bytes to app.py"
    assert (repo / "app.py").read_text() == "value  =  2\n"

    cfg = _cfg(_python_formatter("raise SystemExit('must not run')"))
    unsupported = write("notes.txt", "plain\n", cwd=str(repo), cfg=cfg)
    assert unsupported == "OK: wrote 6 bytes to notes.txt"
    assert "formatter_run" not in unsupported


def test_nearest_root_marker_sets_command_directory_and_relative_path(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    package = repo / "packages/demo"
    source = package / "src/app.py"
    source.parent.mkdir(parents=True)
    (package / "pyproject.toml").write_text("[project]\nname='demo'\n")
    source.write_text("root = 'old'\n")
    _git(repo, "add", "packages")
    _git(repo, "commit", "--quiet", "-m", "add package")
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "p.write_text(\"root = 'demo'\\n\" if Path.cwd().name == 'demo' "
            "and str(p) == 'src/app.py' else \"wrong\\n\")"
        ),
        formatters=[{
            "name": "nested-format",
            "extensions": [".py"],
            "command": _python_formatter(
                "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                "p.write_text(\"root = 'demo'\\n\" if Path.cwd().name == 'demo' "
                "and str(p) == 'src/app.py' else \"wrong\\n\")"
            ),
            "root_markers": ["pyproject.toml"],
        }],
    )

    result = write(
        "packages/demo/src/app.py",
        "root = 'model'\n",
        cwd=str(repo),
        cfg=cfg,
    )

    assert source.read_text() == "root = 'demo'\n"
    assert "Project root: packages/demo" in result
    assert "- packages/demo/src/app.py" in result


def test_failure_and_timeout_leave_and_report_partial_effects(tmp_path: Path):
    repo = _repo(tmp_path)
    failing = _cfg(_python_formatter(
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text('formatter partial\\n'); print('bad format'); sys.exit(7)"
    ))

    failed = write("app.py", "model edit\n", cwd=str(repo), cfg=failing)

    assert failed.startswith("OK: wrote")
    assert 'status="failed" command_status="failed" exit_code="7"' in failed
    assert "Formatter failed. The model edit remains applied." in failed
    assert "- app.py" in failed
    assert "bad format" in failed
    assert (repo / "app.py").read_text() == "formatter partial\n"

    timeout = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys,time; p=Path(sys.argv[1]); "
            "p.write_text('timeout partial\\n'); time.sleep(3)"
        ),
        formatter_timeout=1,
    )
    timed_out = write("app.py", "second model edit\n", cwd=str(repo), cfg=timeout)
    assert 'status="timed_out" command_status="timed_out"' in timed_out
    assert 'timed_out="true"' in timed_out
    assert "command timed out after 1s" in timed_out
    assert "- app.py" in timed_out
    assert (repo / "app.py").read_text() == "timeout partial\n"


def test_formatter_reports_every_repository_visible_collateral_path(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    cfg = _cfg(_python_formatter(
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text('formatted target\\n'); "
        "Path('other.py').write_text('formatted collateral\\n')"
    ))

    result = write("app.py", "model edit\n", cwd=str(repo), cfg=cfg)

    assert "Formatter changed paths:\n- app.py\n- other.py" in result
    assert (repo / "other.py").read_text() == "formatted collateral\n"


def test_bash_deny_prevents_formatter_but_keeps_model_edit(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('must not run\\n')"
        ),
        permissions_rules={"bash": {"*": "deny"}},
    )

    result = write("app.py", "model edit\n", cwd=str(repo), cfg=cfg)

    assert '<formatter_run name="test-format" status="denied"' in result
    assert "resolved bash permission did not allow it" in result
    assert "Permission rule: *" in result
    assert (repo / "app.py").read_text() == "model edit\n"


def test_formatter_uses_the_resolved_command_environment(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import os,sys; "
            "Path(sys.argv[1]).write_text(os.environ['FORMAT_VALUE'] + '\\n')"
        ),
        sandbox_env_inherit="none",
        sandbox_env_set={"FORMAT_VALUE": "from-policy"},
    )

    result = write("app.py", "model edit\n", cwd=str(repo), cfg=cfg)

    assert 'status="changed" command_status="passed"' in result
    assert (repo / "app.py").read_text() == "from-policy\n"


def test_formatter_does_not_run_without_git_attribution(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = _python_formatter(
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).write_text('must not run\\n')"
    )
    cfg = _cfg(command, formatters=[{
        "name": "test-format",
        "extensions": [".py"],
        "command": command,
        "root_markers": [],
    }])

    result = write("app.py", "model edit\n", cwd=str(workspace), cfg=cfg)

    assert '<formatter_run name="test-format" status="not_run"' in result
    assert "Formatter attribution failed" in result
    assert "The formatter command did not run" in result
    assert (workspace / "app.py").read_text() == "model edit\n"


def test_formatter_supports_a_git_repository_before_its_first_commit(
    tmp_path: Path,
):
    repo = tmp_path / "unborn"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    cfg = _cfg(_python_formatter(
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text(p.read_text().replace('  ', ' '))"
    ))

    result = write("app.py", "value  =  2\n", cwd=str(repo), cfg=cfg)

    assert '<formatter_run name="test-format" status="changed"' in result
    assert "Formatter changed paths:\n- app.py" in result
    assert (repo / "app.py").read_text() == "value = 2\n"


def test_post_edit_block_happens_before_formatter(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1]).write_text('must not run\\n')"
        ),
        post_edit_check_enabled=True,
        post_edit_checks=[{
            "name": "reject",
            "trigger": "write",
            "when": "",
            "cmd": "false",
            "on_fail": "block",
        }],
    )

    result = write("app.py", "model edit\n", cwd=str(repo), cfg=cfg)

    assert result.startswith("ERROR: write blocked by post-edit check")
    assert "formatter_run" not in result
    assert (repo / "app.py").read_text() == "value = 1\n"


def test_formatter_output_is_bounded_with_full_hash_and_count(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter("print('x' * 2000)"),
        formatter_max_output_chars=256,
    )

    result = write("app.py", "model edit\n", cwd=str(repo), cfg=cfg)

    assert "Formatter output: chars=2001 sha256=" in result
    assert "truncated=true" in result
    assert "[formatter output bounded:" in result


def test_apply_patch_formats_each_supported_changed_file(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "p.write_text(p.read_text().replace('  ', ' '))"
        ),
        tools_edit_format="apply_patch",
    )
    patch = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@\n"
        "-value = 1\n"
        "+value  =  3\n"
        "*** End Patch\n"
    )

    result = apply_patch_tool(patch, cwd=str(repo), cfg=cfg)

    assert '<formatter_run name="test-format" status="changed"' in result
    assert (repo / "app.py").read_text() == "value = 3\n"


def test_udiff_reports_formatter_result_for_supported_file(tmp_path: Path):
    repo = _repo(tmp_path)
    cfg = _cfg(
        _python_formatter(
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "p.write_text(p.read_text().replace('  ', ' '))"
        ),
        tools_edit_format="udiff",
    )
    patch = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value  =  4\n"
    )

    result = udiff_tool(patch, cwd=str(repo), cfg=cfg)

    assert '<formatter_run name="test-format" status="changed"' in result
    assert (repo / "app.py").read_text() == "value = 4\n"


def test_formatter_config_is_strict_and_requires_explicit_declaration(
    tmp_path: Path,
):
    empty = tmp_path / "empty.toml"
    empty.write_text("[formatter]\nenabled = true\n")
    with pytest.raises(ValueError, match="requires at least one"):
        load_config(user_config=[empty])

    missing_path = tmp_path / "missing-path.toml"
    missing_path.write_text(
        "[formatter]\n"
        "enabled = true\n"
        "[[formatter.formatters]]\n"
        "name = 'ruff'\n"
        "extensions = ['.py']\n"
        "command = ['ruff', 'format']\n"
    )
    with pytest.raises(ValueError, match="exactly one.*path"):
        load_config(user_config=[missing_path])

    valid = tmp_path / "valid.toml"
    valid.write_text(
        "[formatter]\n"
        "enabled = true\n"
        "timeout = 12\n"
        "max_output_chars = 2048\n"
        "[[formatter.formatters]]\n"
        "name = 'ruff'\n"
        "extensions = ['.PY', '.py']\n"
        "root_markers = ['pyproject.toml']\n"
        "command = ['ruff', 'format', '--', '{path}']\n"
    )
    cfg = load_config(user_config=[valid])
    assert cfg.formatter_enabled is True
    assert cfg.formatter_timeout == 12
    assert cfg.formatter_max_output_chars == 2048
    assert cfg.formatters[0]["name"] == "ruff"
