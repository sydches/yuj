"""Structural pattern search and preview-bound editing contracts for #65."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver._permission_presets import PERMISSION_PRESET_SPECS
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._guardrails.extractors import _mutation_signature
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness._stream_rule_runtime import _tool_snapshot
from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness.approval_preview import build_approval_preview
from scripts.llm_solver.harness.injections import path_targets_for_tool
from scripts.llm_solver.harness.post_edit import PostEditResult
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.stale_guard import StaleFileGuard
from scripts.llm_solver.harness.structural_patterns import (
    search_structural_patterns,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.workspace_checkpoints import (
    tool_call_needs_checkpoint,
)

from _config_helpers import make_config


IDENTIFIER_QUERY = '((identifier) @match (#eq? @match "old_name"))'


def _cfg(**overrides):
    return make_config(
        tools_structural_enabled=True,
        tools_unified_envelope_enabled=False,
        **overrides,
    )


def _surface_names(cfg, *, max_tools: int = 99) -> tuple[str, ...]:
    client = SimpleNamespace(
        profile=SimpleNamespace(
            name="fixture",
            edit_format="exact",
            max_tools=max_tools,
            simplify_schemas=False,
        )
    )
    return tuple(
        item["function"]["name"]
        for item in apply_profile_to_schemas(get_tool_schemas(), cfg, client)
    )


def _preview_hash(result: str) -> str:
    assert result.startswith("OK: structural_preview ")
    metadata = json.loads(result.splitlines()[0].removeprefix(
        "OK: structural_preview "
    ))
    return metadata["preview_sha256"]


@pytest.mark.parametrize(
    ("language", "filename", "source"),
    [
        ("python", "sample.py", "old_name = 1\n"),
        ("javascript", "sample.js", "const old_name = 1;\n"),
        ("typescript", "sample.ts", "const old_name: number = 1;\n"),
        ("tsx", "sample.tsx", "const view = <div>{old_name}</div>;\n"),
        ("go", "sample.go", "package demo\nvar old_name = 1\n"),
        ("rust", "sample.rs", "fn main() { let old_name = 1; }\n"),
        ("java", "Sample.java", "class Sample { int old_name = 1; }\n"),
    ],
)
def test_shipped_languages_execute_real_tree_sitter_queries(
    tmp_path: Path,
    language: str,
    filename: str,
    source: str,
) -> None:
    (tmp_path / filename).write_text(source)

    result = search_structural_patterns(
        workspace=tmp_path,
        scope=tmp_path,
        language=language,
        query_source=IDENTIFIER_QUERY,
    )

    assert result.total == 1
    assert result.matches[0].path == filename
    assert result.matches[0].text == "old_name"
    assert result.matches[0].line >= 1
    assert result.matches[0].column >= 1


def test_search_is_deterministic_and_pages_stable_locations(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("old_name = 2\n")
    (tmp_path / "a.py").write_text("old_name = 1\n")
    cfg = _cfg(tools_structural_matches_per_page=1)
    arguments = {
        "path": ".",
        "language": "python",
        "query": IDENTIFIER_QUERY,
    }

    first = dispatch(
        "structural_search", arguments, cwd=str(tmp_path), cfg=cfg,
    )
    repeated = dispatch(
        "structural_search", arguments, cwd=str(tmp_path), cfg=cfg,
    )
    second = dispatch(
        "structural_search", {**arguments, "page": 2},
        cwd=str(tmp_path), cfg=cfg,
    )

    assert first == repeated
    assert '"path": "a.py"' in first
    assert '"next_page": 2' in first
    assert '"path": "b.py"' in second
    assert '"next_page": 0' in second


def test_preview_then_apply_changes_every_exact_location(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    original = "old_name = 1\nprint(old_name)\n"
    target.write_text(original)
    cfg = _cfg()
    preview_args = {
        "path": "sample.py",
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
    }

    preview = dispatch(
        "structural_search", preview_args, cwd=str(tmp_path), cfg=cfg,
    )

    assert target.read_text() == original
    assert '"state": "not_applied"' in preview
    assert preview.count('"path": "sample.py"') == 3
    assert "-old_name = 1" in preview
    assert "+new_name = 1" in preview
    result = dispatch(
        "structural_edit",
        {**preview_args, "expected_sha256": _preview_hash(preview)},
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert target.read_text() == "new_name = 1\nprint(new_name)\n"
    assert result.startswith("OK: structural_edit ")
    assert result.count('"path": "sample.py"') == 3
    assert hashlib.sha256(target.read_bytes()).hexdigest() in result


def test_capture_template_reorders_structural_children(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("answer = left + right\n")
    query = """
    (binary_operator
      left: (identifier) @left
      operator: "+"
      right: (identifier) @right) @match
    """
    cfg = _cfg()
    preview_args = {
        "path": "sample.py",
        "language": "python",
        "query": query,
        "replacement": "${right} + ${left}",
    }
    preview = dispatch(
        "structural_search", preview_args, cwd=str(tmp_path), cfg=cfg,
    )

    result = dispatch(
        "structural_edit",
        {**preview_args, "expected_sha256": _preview_hash(preview)},
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert result.startswith("OK: structural_edit ")
    assert target.read_text() == "answer = right + left\n"


def test_stale_preview_refuses_external_change(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("old_name = 1\n")
    cfg = _cfg()
    args = {
        "path": "sample.py",
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
    }
    preview = dispatch(
        "structural_search", args, cwd=str(tmp_path), cfg=cfg,
    )
    target.write_text("old_name = 2\n")

    result = dispatch(
        "structural_edit",
        {**args, "expected_sha256": _preview_hash(preview)},
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert '"error_kind": "stale_preview"' in result
    assert target.read_text() == "old_name = 2\n"


@pytest.mark.parametrize(
    ("overrides", "expected_kind"),
    [
        ({"query": "(not_a_real_node) @match"}, "invalid_pattern"),
        ({"query": "(identifier) @other"}, "invalid_pattern"),
        ({"query": '((identifier) @match (#eq? @match "missing"))'}, "no_match"),
        ({"language": "ruby"}, "unsupported_language"),
        ({"language": "javascript"}, "language_mismatch"),
        ({"replacement": "${missing}"}, "missing_capture"),
        ({"replacement": "("}, "invalid_replacement"),
        ({"query": "(module) @match\n(identifier) @match"}, "overlapping_matches"),
    ],
)
def test_invalid_unsupported_ambiguous_and_no_match_do_not_mutate(
    tmp_path: Path,
    overrides: dict[str, str],
    expected_kind: str,
) -> None:
    target = tmp_path / "sample.py"
    original = "old_name = 1\n"
    target.write_text(original)
    args = {
        "path": "sample.py",
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
        **overrides,
    }

    result = dispatch(
        "structural_search", args, cwd=str(tmp_path), cfg=_cfg(),
    )

    assert result.startswith("ERROR:")
    assert f'"error_kind": "{expected_kind}"' in result
    assert target.read_text() == original


def test_multiple_match_captures_are_ambiguous_and_do_not_mutate(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    original = "answer = left + right\n"
    target.write_text(original)
    query = """
    (binary_operator
      left: (identifier) @match
      right: (identifier) @match)
    """

    result = dispatch(
        "structural_search",
        {
            "path": "sample.py",
            "language": "python",
            "query": query,
            "replacement": "changed",
        },
        cwd=str(tmp_path),
        cfg=_cfg(),
    )

    assert '"error_kind": "ambiguous_match"' in result
    assert target.read_text() == original


def test_malformed_hidden_unreadable_and_outside_sources_are_not_changed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    malformed = workspace / "broken.py"
    malformed.write_text("def broken(:\n")
    hidden = workspace / "hidden.py"
    hidden.write_text("old_name = 1\n")
    unreadable = workspace / "secret.py"
    unreadable.write_text("old_name = 2\n")
    outside = tmp_path / "outside.py"
    outside.write_text("old_name = 3\n")
    (workspace / ".yujignore").write_text("hidden.py\n")
    policy = load_ignore_policy(workspace)
    cfg = _cfg(unreadable_paths=("secret.py",))
    common = {
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
    }

    malformed_result = dispatch(
        "structural_search", {**common, "path": "broken.py"},
        cwd=str(workspace), cfg=cfg, ignore_policy=policy,
    )
    hidden_result = dispatch(
        "structural_search", {**common, "path": "hidden.py"},
        cwd=str(workspace), cfg=cfg, ignore_policy=policy,
    )
    unreadable_result = dispatch(
        "structural_search", {**common, "path": "secret.py"},
        cwd=str(workspace), cfg=cfg, ignore_policy=policy,
    )
    outside_result = dispatch(
        "structural_search", {**common, "path": "../outside.py"},
        cwd=str(workspace), cfg=cfg, ignore_policy=policy,
    )

    assert '"error_kind": "parse_error"' in malformed_result
    assert '"error_kind": "not_found"' in hidden_result
    assert '"error_kind": "not_found"' in unreadable_result
    assert '"error_kind": "path_outside_cwd"' in outside_result
    assert hidden.read_text() == "old_name = 1\n"
    assert unreadable.read_text() == "old_name = 2\n"
    assert outside.read_text() == "old_name = 3\n"


def test_blocking_post_edit_check_restores_exact_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "sample.py"
    original = b"old_name = 1\r\n"
    target.write_bytes(original)
    cfg = _cfg()
    args = {
        "path": "sample.py",
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
    }
    preview = dispatch(
        "structural_search", args, cwd=str(tmp_path), cfg=cfg,
    )
    post_edit = importlib.import_module("scripts.llm_solver.harness.post_edit")
    calls: list[tuple[str, str]] = []

    def block(path: str, *, cwd: str, cfg, trigger: str) -> PostEditResult:
        calls.append((path, trigger))
        return PostEditResult("block", "\ncheck output", "syntax-check")

    monkeypatch.setattr(post_edit, "run_post_edit_checks", block)
    result = dispatch(
        "structural_edit",
        {**args, "expected_sha256": _preview_hash(preview)},
        cwd=str(tmp_path),
        cfg=cfg,
    )

    assert '"error_kind": "post_edit_blocked"' in result
    assert calls == [("sample.py", "edit")]
    assert target.read_bytes() == original


def test_config_surface_defaults_validation_and_profile_cap(tmp_path: Path) -> None:
    defaults = load_config()
    assert defaults.tools_structural_enabled is False
    assert defaults.tools_structural_max_files == 1000
    assert defaults.tools_structural_max_matches == 100
    assert defaults.tools_structural_matches_per_page == 25
    assert defaults.tools_structural_max_file_bytes == 4_194_304
    assert {"structural_search", "structural_edit"}.isdisjoint(
        _surface_names(make_config())
    )
    enabled = _cfg()
    assert {"structural_search", "structural_edit"} <= set(
        _surface_names(enabled, max_tools=8)
    )

    overlay = tmp_path / "structural.toml"
    overlay.write_text(
        "[tools]\nstructural_enabled = true\nstructural_max_files = 7\n"
        "structural_max_matches = 6\nstructural_matches_per_page = 5\n"
        "structural_max_file_bytes = 4096\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_structural_enabled is True
    assert configured.tools_structural_max_files == 7
    assert configured.tools_structural_max_matches == 6
    assert configured.tools_structural_matches_per_page == 5
    assert configured.tools_structural_max_file_bytes == 4096

    overlay.write_text('[tools]\nstructural_enabled = "yes"\n')
    with pytest.raises(ValueError, match="structural_enabled"):
        load_config(user_config=overlay)

    for setting in (
        "structural_max_files",
        "structural_max_matches",
        "structural_matches_per_page",
        "structural_max_file_bytes",
    ):
        overlay.write_text(f"[tools]\n{setting} = 0\n")
        with pytest.raises(ValueError, match=setting):
            load_config(user_config=overlay)

    target = tmp_path / "sample.py"
    target.write_text("old_name = 1\n")
    disabled = dispatch(
        "structural_search",
        {"path": "sample.py", "language": "python", "query": IDENTIFIER_QUERY},
        cwd=str(tmp_path),
        cfg=make_config(tools_unified_envelope_enabled=False),
    )
    assert '"error_kind": "disabled"' in disabled
    assert target.read_text() == "old_name = 1\n"


def test_approval_stale_permission_checkpoint_and_trace_controls(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("old_name = 1\n")
    cfg = _cfg()
    preview_args = {
        "path": "sample.py",
        "language": "python",
        "query": IDENTIFIER_QUERY,
        "replacement": "new_name",
    }
    preview = dispatch(
        "structural_search", preview_args, cwd=str(tmp_path), cfg=cfg,
    )
    edit_args = {
        **preview_args,
        "expected_sha256": _preview_hash(preview),
    }

    approval = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="structural_edit",
        tool_args=edit_args,
        cfg=cfg,
    )
    assert approval["status"] == "available"
    assert "+new_name = 1" in approval["content"]
    assert target.read_text() == "old_name = 1\n"

    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    blocked = dispatch(
        "structural_edit", edit_args, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    assert blocked == "ERROR: stale_file: read sample.py first"
    dispatch(
        "read", {"path": "sample.py"}, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    changed = dispatch(
        "structural_edit", edit_args, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    assert changed.startswith("OK: structural_edit ")
    assert guard.check_edit("sample.py").allowed is True

    assert PERMISSION_PRESET_SPECS["read-only"].tool_decisions[
        "structural_search"
    ] == "allow"
    assert "structural_edit" not in PERMISSION_PRESET_SPECS[
        "read-only"
    ].tool_decisions
    assert PERMISSION_PRESET_SPECS["allow-edits"].tool_decisions[
        "structural_edit"
    ] == "allow"
    assert action_metadata("structural_edit", edit_args) == {
        "write_like": True,
        "source_write_like": True,
        "source_write_paths": ["sample.py"],
    }
    assert tool_call_needs_checkpoint("structural_edit") is True

    signature = _mutation_signature("structural_edit", edit_args)
    changed_signature = _mutation_signature(
        "structural_edit", {**edit_args, "replacement": "other_name"}
    )
    snapshot, paths, ast_eligible = _tool_snapshot(
        json.dumps(edit_args), "structural_edit", tmp_path
    )
    targets = path_targets_for_tool(
        "structural_edit", edit_args, cwd=str(tmp_path)
    )
    assert signature[1] == "sample.py"
    assert signature[0] != changed_signature[0]
    assert snapshot == "new_name"
    assert paths == ("sample.py",)
    assert ast_eligible is False
    assert tuple(target.path for target in targets) == ("sample.py",)


def test_match_and_file_limits_fail_clearly_without_mutation(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    original = "old_name = old_name\n"
    target.write_text(original)
    limited_matches = dispatch(
        "structural_search",
        {
            "path": "sample.py",
            "language": "python",
            "query": IDENTIFIER_QUERY,
            "replacement": "new_name",
        },
        cwd=str(tmp_path),
        cfg=_cfg(tools_structural_max_matches=1),
    )
    assert '"error_kind": "match_limit"' in limited_matches
    assert target.read_text() == original

    (tmp_path / "other.py").write_text("old_name = 2\n")
    limited_files = dispatch(
        "structural_search",
        {"path": ".", "language": "python", "query": IDENTIFIER_QUERY},
        cwd=str(tmp_path),
        cfg=_cfg(tools_structural_max_files=1),
    )
    assert '"error_kind": "file_limit"' in limited_files
    assert target.read_text() == original
