"""Runtime acceptance tests for the repository model-view ignore file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import TurnResult, Usage

from _config_helpers import make_config


def _fixture_policy(root: Path):
    (root / ".yujignore").write_text(
        "secret.py\nprivate/\n!private/keep.py\n"
    )
    (root / "public.py").write_text(
        "MARKER = 'public'\n\ndef visible_symbol():\n    return 1\n"
    )
    (root / "secret.py").write_text(
        "MARKER = 'secret'\n\ndef hidden_symbol():\n    return 2\n"
    )
    private = root / "private"
    private.mkdir()
    (private / "drop.py").write_text("def dropped_symbol():\n    pass\n")
    (private / "keep.py").write_text("def kept_symbol():\n    pass\n")
    return load_ignore_policy(root)


def test_ignore_file_config_defaults_overlay_and_validation(tmp_path: Path) -> None:
    defaults = load_config()
    assert defaults.state_ignore_file_enabled is True
    assert defaults.state_ignore_file_names == (".yujignore",)

    overlay = tmp_path / "state.toml"
    overlay.write_text(
        "[state]\n"
        "ignore_file_enabled = false\n"
        'ignore_file_names = [".first", ".second"]\n'
    )
    configured = load_config(user_config=overlay)
    assert configured.state_ignore_file_enabled is False
    assert configured.state_ignore_file_names == (".first", ".second")

    for body, message in (
        ('ignore_file_enabled = "yes"', "ignore_file_enabled"),
        ('ignore_file_names = ".yujignore"', "array of strings"),
        ('ignore_file_names = ["../outside"]', "within the task root"),
    ):
        invalid = tmp_path / f"invalid-{len(message)}.toml"
        invalid.write_text(f"[state]\n{body}\n")
        with pytest.raises(ValueError, match=message):
            load_config(user_config=invalid)


def test_read_glob_grep_and_single_file_definitions_share_policy(
    tmp_path: Path,
) -> None:
    policy = _fixture_policy(tmp_path)
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=True,
        tools_glob_refuse_unscoped_recursive=False,
        tools_glob_max_listed_paths=0,
    )
    common = {"cwd": str(tmp_path), "cfg": cfg, "ignore_policy": policy}

    hidden_read = dispatch("read", {"path": "secret.py"}, **common)
    assert "file not found" in hidden_read
    assert "hidden_symbol" not in hidden_read
    assert "visible_symbol" in dispatch(
        "read", {"path": "public.py"}, **common
    )

    globbed = dispatch(
        "glob", {"pattern": "**/*.py", "path": "."}, **common
    )
    assert "public.py" in globbed
    assert "private/keep.py" in globbed
    assert "secret.py" not in globbed
    assert "private/drop.py" not in globbed
    nested_glob = dispatch(
        "glob", {"pattern": "*.py", "path": "private"}, **common
    )
    assert "private/keep.py" in nested_glob
    assert "private/drop.py" not in nested_glob

    grepped = dispatch(
        "grep", {"pattern": "MARKER", "path": "."}, **common
    )
    assert "public.py" in grepped
    assert "secret.py" not in grepped
    assert "'secret'" not in grepped

    hidden_outline = dispatch(
        "list_definitions", {"path": "secret.py"}, **common
    )
    assert 'error_kind="not_found"' in hidden_outline
    assert "hidden_symbol" not in hidden_outline


def test_repository_structural_search_excludes_ignored_sources(
    tmp_path: Path,
) -> None:
    policy = _fixture_policy(tmp_path)
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=True,
        grep_max_matches_per_page=50,
    )
    common = {"cwd": str(tmp_path), "cfg": cfg, "ignore_policy": policy}

    visible = dispatch(
        "list_definitions",
        {
            "path": ".",
            "repo_wide": True,
            "symbol": "visible_symbol",
            "kind": "def",
        },
        **common,
    )
    hidden = dispatch(
        "list_definitions",
        {
            "path": ".",
            "repo_wide": True,
            "symbol": "hidden_symbol",
            "kind": "def",
        },
        **common,
    )
    kept = dispatch(
        "list_definitions",
        {
            "path": ".",
            "repo_wide": True,
            "symbol": "kept_symbol",
            "kind": "def",
        },
        **common,
    )

    assert "public.py" in visible
    assert 'total="0"' in hidden
    assert "secret.py" not in hidden
    assert "private/keep.py" in kept


def test_bash_ls_cat_and_sandbox_masks_share_policy(tmp_path: Path) -> None:
    policy = _fixture_policy(tmp_path)
    cfg = make_config(sandbox_bash=True, sandbox_required=False)
    common = {"cwd": str(tmp_path), "cfg": cfg, "ignore_policy": policy}

    listing = dispatch("bash", {"cmd": "ls -la"}, **common)
    assert "public.py" in listing
    assert "secret.py" not in listing
    assert "private" in listing
    nested_listing = dispatch(
        "bash", {"cmd": "ls -1 private"}, **common
    )
    assert "keep.py" in nested_listing
    assert "drop.py" not in nested_listing

    hidden_cat = dispatch("bash", {"cmd": "cat secret.py"}, **common)
    assert "No such file or directory" in hidden_cat
    assert "hidden_symbol" not in hidden_cat

    with patch(
        "scripts.llm_solver.harness.tools._run_in_sandbox",
        return_value=("ok", 0, False),
    ) as runner:
        assert "ok" in dispatch("bash", {"cmd": "printf ok"}, **common)
    masks = runner.call_args.kwargs["unreadable_paths"]
    assert f"optional:{tmp_path / 'secret.py'}" in masks
    assert f"optional:{tmp_path / 'private' / 'drop.py'}" in masks
    assert f"optional:{tmp_path / 'private' / 'keep.py'}" not in masks


def test_session_start_hash_is_safe_and_immutable_across_sessions(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.harness.state_writer import project

    work = tmp_path / "task"
    work.mkdir()
    (work / ".git").mkdir()
    raw = b"secret.py\n"
    ignore_file = work / ".yujignore"
    ignore_file.write_bytes(raw)
    (work / "secret.py").write_text("SECRET")
    (work / "prompt.txt").write_text("finish")

    calls = 0

    def chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            ignore_file.write_text("changed.py\n")
        return TurnResult(
            content="continue",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )

    client = MagicMock()
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "continue",
    }
    cfg = make_config(
        max_sessions=2,
        max_turns=1,
        allow_implicit_done=False,
        state_writer_enabled=False,
    )

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(work, cfg, client) is False

    starts = [
        json.loads(line)
        for line in trace_path(work).read_text().splitlines()
        if json.loads(line).get("event") == "session_start"
    ]
    expected = hashlib.sha256(raw).hexdigest()
    assert len(starts) == 2
    assert {event["ignore_file_hash"] for event in starts} == {expected}
    assert all(event["ignore_file_names"] == [".yujignore"] for event in starts)
    assert "secret.py" not in json.dumps(starts)
    projected = project(starts, max_result_chars=2000)
    assert "ignore_file_hash" not in json.dumps(projected)
    metrics = json.loads((work / "metrics.json").read_text())
    assert str(work / "secret.py") not in json.dumps(
        metrics["provenance"]["config"]
    )
