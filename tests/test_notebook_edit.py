"""Source-only Jupyter notebook editing contracts for issue #64."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver._permission_presets import PERMISSION_PRESET_SPECS
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness._guardrails.extractors import (
    _mutation_signature,
)
from scripts.llm_solver.harness._stream_rule_runtime import _tool_snapshot
from scripts.llm_solver.harness._tools.notebook_edit import (
    notebook_edit,
    propose_notebook_edit,
)
from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness.approval_preview import build_approval_preview
from scripts.llm_solver.harness.post_edit import PostEditResult
from scripts.llm_solver.harness.injections import path_targets_for_tool
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.stale_guard import StaleFileGuard
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.workspace_checkpoints import (
    tool_call_needs_checkpoint,
)

from _config_helpers import make_config


def _notebook_text() -> str:
    return """{
  "metadata": {"kernelspec": {"name": "python3"}},
  "nbformat": 4,
  "custom": {"keep": [1, {"spacing": true}]},
  "cells": [
    {
      "cell_type": "markdown",
      "id": "intro",
      "metadata": {"tags": ["keep-me"]},
      "attachments": {"pixel.png": {"image/png": "AAAA"}},
      "source": [
        "# Héllo\\n",
        "Second α\\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "id": "calc",
      "metadata": {"collapsed": true},
      "outputs": [
        {"output_type": "stream", "name": "stdout", "text": ["old output\\n"]}
      ],
      "source": "print('old')\\n"
    }
  ],
  "nbformat_minor": 5
}
"""


def _source(cell: dict) -> str:
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def _cfg(**overrides):
    return make_config(tools_notebook_edit_enabled=True, **overrides)


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


def test_code_cell_string_source_changes_without_reserializing_notebook(
    tmp_path: Path,
) -> None:
    target = tmp_path / "analysis.ipynb"
    original = _notebook_text()
    target.write_text(original)
    before = json.loads(original)

    result = notebook_edit(
        "analysis.ipynb",
        "print('old')\n",
        "print('new 🚀')\n",
        cwd=str(tmp_path),
        cell_id="calc",
        cfg=_cfg(),
    )

    updated_text = target.read_text()
    updated = json.loads(updated_text)
    assert result == "OK: updated code cell id='calc' in analysis.ipynb"
    assert _source(updated["cells"][1]) == "print('new 🚀')\n"
    assert isinstance(updated["cells"][1]["source"], str)
    before["cells"][1]["source"] = updated["cells"][1]["source"]
    assert updated == before
    assert updated_text == original.replace(
        '"source": "print(\'old\')\\n"',
        '"source": "print(\'new 🚀\')\\n"',
        1,
    )


def test_markdown_array_source_preserves_every_byte_outside_selected_value(
    tmp_path: Path,
) -> None:
    target = tmp_path / "analysis.ipynb"
    original = _notebook_text().replace("\n", "\r\n")
    target.write_bytes(original.encode("utf-8"))
    new_source = "# Updated 🚀\nLine two\nLine three"
    proposal = propose_notebook_edit(
        original,
        old_source="# Héllo\nSecond α\n",
        new_source=new_source,
        cell_index=0,
    )

    result = notebook_edit(
        "analysis.ipynb",
        "# Héllo\nSecond α\n",
        new_source,
        cwd=str(tmp_path),
        cell_index=0,
        cfg=_cfg(),
    )

    updated_text = target.read_bytes().decode("utf-8")
    updated = json.loads(updated_text)
    assert result == "OK: updated markdown cell id='intro' in analysis.ipynb"
    assert _source(updated["cells"][0]) == new_source
    assert isinstance(updated["cells"][0]["source"], list)
    assert updated_text[: proposal.source_start] == original[: proposal.source_start]
    assert updated_text[proposal.source_start:].startswith(proposal.replacement)
    assert updated_text[proposal.source_start + len(proposal.replacement):] == (
        original[proposal.source_end:]
    )
    assert "\r\n" in updated_text
    assert "\n" not in updated_text.replace("\r\n", "")


def test_no_op_validates_but_does_not_write(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "analysis.ipynb"
    original = _notebook_text().encode("utf-8")
    target.write_bytes(original)
    module = importlib.import_module(
        "scripts.llm_solver.harness._tools.notebook_edit"
    )

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("no-op must not write")

    monkeypatch.setattr(module, "_atomic_write", unexpected_write)
    result = notebook_edit(
        "analysis.ipynb",
        "print('old')\n",
        "print('old')\n",
        cwd=str(tmp_path),
        cell_id="calc",
        cfg=_cfg(),
    )

    assert "no file change" in result
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    ("mutate_text", "kwargs", "message"),
    [
        (lambda text: text, {"cell_id": "missing"}, "was not found"),
        (lambda text: text, {"cell_id": "calc"}, "stale source"),
        (lambda text: text, {}, "exactly one"),
        (
            lambda text: text,
            {"cell_index": -1},
            "outside the notebook cell range",
        ),
        (
            lambda text: text,
            {"cell_id": "calc", "cell_index": 1},
            "exactly one",
        ),
        (
            lambda text: text.replace('"id": "calc"', '"id": "intro"'),
            {"cell_id": "intro"},
            "ambiguous duplicate cell id",
        ),
        (
            lambda text: text.replace(
                '"cell_type": "markdown"', '"cell_type": "raw"', 1
            ),
            {"cell_index": 0},
            "only code and markdown",
        ),
        (lambda _text: "{broken", {"cell_index": 0}, "not valid JSON"),
    ],
)
def test_invalid_missing_stale_and_ambiguous_requests_leave_bytes_unchanged(
    tmp_path: Path,
    mutate_text,
    kwargs: dict,
    message: str,
) -> None:
    target = tmp_path / "analysis.ipynb"
    original = mutate_text(_notebook_text())
    target.write_bytes(original.encode("utf-8"))
    old_source = (
        "not the current source"
        if message == "stale source"
        else (
            "# Héllo\nSecond α\n"
            if kwargs.get("cell_index") == 0
            else "print('old')\n"
        )
    )

    result = notebook_edit(
        "analysis.ipynb",
        old_source,
        "replacement\n",
        cwd=str(tmp_path),
        cfg=_cfg(),
        **kwargs,
    )

    assert result.startswith("ERROR:")
    assert message in result
    assert target.read_bytes() == original.encode("utf-8")


def test_path_boundary_and_extension_fail_without_touching_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside.ipynb"
    outside.write_text(_notebook_text())
    inside = workspace / "analysis.json"
    inside.write_text(_notebook_text())

    escaped = notebook_edit(
        "../outside.ipynb",
        "print('old')\n",
        "changed\n",
        cwd=str(workspace),
        cell_id="calc",
        cfg=_cfg(),
    )
    wrong_extension = notebook_edit(
        "analysis.json",
        "print('old')\n",
        "changed\n",
        cwd=str(workspace),
        cell_id="calc",
        cfg=_cfg(),
    )

    assert "path escapes cwd" in escaped
    assert ".ipynb path" in wrong_extension
    assert outside.read_text() == _notebook_text()
    assert inside.read_text() == _notebook_text()


def test_blocking_post_edit_check_restores_exact_original_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "analysis.ipynb"
    original = _notebook_text().replace("\n", "\r\n").encode("utf-8")
    target.write_bytes(original)
    post_edit = importlib.import_module("scripts.llm_solver.harness.post_edit")
    calls: list[tuple[str, str]] = []

    def block(path: str, *, cwd: str, cfg, trigger: str) -> PostEditResult:
        calls.append((path, trigger))
        return PostEditResult("block", "\ncheck output", "notebook-check")

    monkeypatch.setattr(post_edit, "run_post_edit_checks", block)
    result = notebook_edit(
        "analysis.ipynb",
        "print('old')\n",
        "broken\n",
        cwd=str(tmp_path),
        cell_id="calc",
        cfg=_cfg(),
    )

    assert result.startswith("ERROR: notebook edit blocked")
    assert calls == [("analysis.ipynb", "edit")]
    assert target.read_bytes() == original


def test_config_surface_is_opt_in_validated_and_survives_base_cap(
    tmp_path: Path,
) -> None:
    assert load_config().tools_notebook_edit_enabled is False
    overlay = tmp_path / "notebooks.toml"
    overlay.write_text("[tools]\nnotebook_edit_enabled = true\n")
    configured = load_config(user_config=overlay)
    assert configured.tools_notebook_edit_enabled is True
    assert "notebook_edit" not in _surface_names(make_config())
    enabled = make_config(tools_notebook_edit_enabled=True)
    assert "notebook_edit" in _surface_names(enabled)
    assert "notebook_edit" in _surface_names(enabled, max_tools=8)

    target = tmp_path / "analysis.ipynb"
    target.write_text(_notebook_text())
    disabled_result = dispatch(
        "notebook_edit",
        {
            "path": "analysis.ipynb",
            "cell_id": "calc",
            "old_source": "print('old')\n",
            "new_source": "print('new')\n",
        },
        cwd=str(tmp_path),
        cfg=make_config(),
    )
    assert "tool is disabled" in disabled_result
    assert target.read_text() == _notebook_text()

    overlay.write_text('[tools]\nnotebook_edit_enabled = "yes"\n')
    with pytest.raises(ValueError, match="notebook_edit_enabled"):
        load_config(user_config=overlay)


def test_approval_preview_is_exact_bounded_and_read_only(tmp_path: Path) -> None:
    target = tmp_path / "analysis.ipynb"
    original = _notebook_text()
    target.write_text(original)

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="notebook_edit",
        tool_args={
            "path": "analysis.ipynb",
            "cell_id": "calc",
            "old_source": "print('old')\n",
            "new_source": "print('new')\n",
        },
    )

    assert preview["status"] == "available"
    assert preview["format"] == "unified_diff"
    assert preview["paths"] == ["analysis.ipynb"]
    assert '-      "source": "print(\'old\')\\n"' in preview["content"]
    assert '+      "source": "print(\'new\')\\n"' in preview["content"]
    assert target.read_text() == original


def test_stale_guard_requires_read_then_observes_successful_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "analysis.ipynb"
    target.write_text(_notebook_text())
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    cfg = make_config(
        tools_notebook_edit_enabled=True,
        tools_unified_envelope_enabled=False,
    )
    arguments = {
        "path": "analysis.ipynb",
        "cell_id": "calc",
        "old_source": "print('old')\n",
        "new_source": "print('new')\n",
    }

    blocked = dispatch(
        "notebook_edit", arguments, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    assert blocked == "ERROR: stale_file: read analysis.ipynb first"
    assert target.read_text() == _notebook_text()
    dispatch(
        "read", {"path": "analysis.ipynb"}, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )
    changed = dispatch(
        "notebook_edit", arguments, cwd=str(tmp_path), cfg=cfg,
        stale_guard=guard,
    )

    assert changed.startswith("OK: updated code cell")
    assert guard.check_edit("analysis.ipynb").allowed is True


def test_permission_presets_and_action_metadata_treat_tool_as_an_edit() -> None:
    assert (
        PERMISSION_PRESET_SPECS["allow-edits"].tool_decisions["notebook_edit"]
        == "allow"
    )
    assert "notebook_edit" not in (
        PERMISSION_PRESET_SPECS["read-only"].tool_decisions
    )
    metadata = action_metadata(
        "notebook_edit", {"path": "analysis.ipynb", "new_source": "x"}
    )
    assert metadata == {
        "write_like": True,
        "source_write_like": True,
        "source_write_paths": ["analysis.ipynb"],
    }
    assert tool_call_needs_checkpoint("notebook_edit") is True


def test_repeat_stream_and_path_controls_use_notebook_edit_arguments(
    tmp_path: Path,
) -> None:
    arguments = {
        "path": "analysis.ipynb",
        "cell_id": "calc",
        "old_source": "old\n",
        "new_source": "new\n",
    }
    first_signature = _mutation_signature("notebook_edit", arguments)
    second_signature = _mutation_signature(
        "notebook_edit", {**arguments, "new_source": "different\n"}
    )
    snapshot, paths, ast_eligible = _tool_snapshot(
        json.dumps(arguments), "notebook_edit", tmp_path
    )
    targets = path_targets_for_tool(
        "notebook_edit", arguments, cwd=str(tmp_path)
    )

    assert first_signature[1] == "analysis.ipynb"
    assert first_signature[0] != second_signature[0]
    assert snapshot == "new\n"
    assert paths == ("analysis.ipynb",)
    assert ast_eligible is False
    assert tuple(target.path for target in targets) == ("analysis.ipynb",)
