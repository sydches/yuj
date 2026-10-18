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
from tests.test_task_files import bwrap, namespace_files


@pytest.mark.parametrize('text,enabled', [(None, True), ('# no rules\n', True), ('*\n', False)])
def test_empty_policy_tools_keep_native_results_without_ignore_only_stats(
    bwrap, tmp_path, text, enabled,
):
    from scripts.llm_solver.harness._guardrails.state import GuardrailState
    from scripts.llm_solver.harness._guardrails.test_inspection import _workspace_coverage
    from scripts.llm_solver.harness.lsp_support import LspManager
    from scripts.llm_solver.harness.operation_timing import operation_profile
    from scripts.llm_solver.harness.sandbox.ignore_policy import activate_ignore_policy
    from scripts.llm_solver.harness.task_path import activate_task_files

    source, files = namespace_files(bwrap, tmp_path)
    (source / 'src').mkdir()
    (source / 'src/code.py').write_text('def native_marker():\n    return 1\n')
    (Path(files.root) / 'src').mkdir()
    (Path(files.root) / 'src/code.py').write_text('HOST_SHADOW')
    if text is not None:
        (source / '.yujignore').write_text(text)
    cfg = make_config(sandbox_bash=True, sandbox_required=True,
                      tools_list_definitions_enabled=True)
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source, enabled=enabled)
        assert not policy.has_rules
        common = dict(cwd=str(source), cfg=cfg, ignore_policy=policy,
                      effective_env={'PATH': '/usr/bin:/bin'}, allow_login_shell=False)
        for name, args, expected, stats in (
            ('grep', {'pattern': 'native_marker', 'path': 'src'}, 'native_marker', 0),
            ('glob', {'pattern': '*.py', 'path': 'src'}, 'src/code.py', 0),
            ('list_definitions', {'path': 'src/code.py'}, 'native_marker', 1),
            ('bash', {'cmd': 'cat src/code.py'}, 'native_marker', 0),
            ('bash', {'cmd': 'head src/code.py'}, 'native_marker', 0),
            ('bash', {'cmd': 'ls src'}, 'code.py', 3 if enabled and text is not None else 0),
        ):
            with operation_profile() as profile:
                result = dispatch(name, args, **common)
            assert expected in result and 'HOST_SHADOW' not in result
            assert sum(row['calls'] for row in profile.take()
                       if row['operation'][-1] == 'files.stat') == stats
        with activate_ignore_policy(policy):
            manager = LspManager(cwd=str(source), servers=(), argv_builder=lambda *_: [],
                                 tool_enabled=True, task_files=files)
            with operation_profile() as profile:
                assert manager.query('symbols', path='src/code.py', ignore_policy=policy).status == 'unmatched'
            assert not any(row['operation'][-1] == 'files.stat' for row in profile.take())
            request = dict(workspace_cwd=str(source), workspace_namespace='task_execution',
                           workspace_task_view=files.binding)
            with operation_profile() as profile:
                coverage = _workspace_coverage(GuardrailState(), cfg, str(source), request, 'src/code.py')
            assert coverage[0] == 'file spelling'
            assert sum(row['calls'] for row in profile.take()
                       if row['operation'][-1] == 'files.stat') == 2


def test_native_tool_ignore_checks_observe_current_type_and_reject_escape(bwrap, tmp_path):
    from scripts.llm_solver.harness.task_path import activate_task_files

    source, files = namespace_files(bwrap, tmp_path)
    (source / '.yujignore').write_text('change.py/\n')
    target = source / 'change.py'
    target.write_text('def visible_marker():\n    pass\n')
    (source / 'escape.py').symlink_to(tmp_path / 'outside.py')
    (tmp_path / 'outside.py').write_text('OUTSIDE_MARKER')
    cfg = make_config(sandbox_bash=True, sandbox_required=True, tools_list_definitions_enabled=True)
    with activate_task_files(files, host_root=source):
        policy = load_ignore_policy(source)
        common = dict(cwd=str(source), cfg=cfg, ignore_policy=policy,
                      effective_env={'PATH': '/usr/bin:/bin'}, allow_login_shell=False)
        for name, args in (('grep', {'pattern': 'visible_marker', 'path': 'change.py'}),
                           ('list_definitions', {'path': 'change.py'}),
                           ('bash', {'cmd': 'cat change.py'}), ('bash', {'cmd': 'head change.py'})):
            assert 'visible_marker' in dispatch(name, args, **common)
        target.unlink()
        target.mkdir()
        (target / 'visible_marker.py').write_text('def visible_marker():\n    pass\n')
        for name, args in (('grep', {'pattern': 'visible_marker', 'path': 'change.py'}),
                           ('glob', {'pattern': '*', 'path': 'change.py'}),
                           ('list_definitions', {'path': 'change.py'}),
                           ('bash', {'cmd': 'cat change.py'}), ('bash', {'cmd': 'head change.py'}),
                           ('bash', {'cmd': 'ls change.py'})):
            assert 'visible_marker' not in dispatch(name, args, **common)
        for name, args in (('grep', {'pattern': 'OUTSIDE_MARKER', 'path': 'escape.py'}),
                           ('list_definitions', {'path': 'escape.py'}),
                           ('bash', {'cmd': 'cat escape.py'})):
            with pytest.raises(PermissionError, match='path escapes task root'):
                dispatch(name, args, **common)


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
    from scripts.llm_solver.harness.sandbox import bwrap_preflight
    ready, reason = bwrap_preflight('/usr/bin/bwrap')
    if not ready:
        pytest.skip(reason)
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
