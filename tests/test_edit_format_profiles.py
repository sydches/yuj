"""Acceptance coverage for inherited and overridden edit-format selection."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
    build_tool_surface,
)
from llm_solver.harness.schemas import get_tool_schemas
from llm_solver.harness.tools import dispatch
from llm_solver.server.profile_loader import load_profile


_EXPECTED_TOOL = {
    "exact": "edit",
    "apply_patch": "apply_patch",
    "udiff": "udiff",
    "whole": "write",
}
_EDIT_TOOLS = frozenset(_EXPECTED_TOOL.values())


@pytest.mark.parametrize("edit_format", tuple(_EXPECTED_TOOL))
def test_each_dialect_exposes_only_its_matching_tool(edit_format: str) -> None:
    cfg = make_config(
        tools_edit_format=edit_format,
        tools_apply_patch_enabled=False,
    )
    profile = SimpleNamespace(
        name="fixture",
        edit_format="whole",
        max_tools=99,
        simplify_schemas=False,
    )
    schemas = apply_profile_to_schemas(
        get_tool_schemas("minimal"), cfg, SimpleNamespace(profile=profile)
    )
    names = {schema["function"]["name"] for schema in schemas}

    assert names & _EDIT_TOOLS == {_EXPECTED_TOOL[edit_format]}


@pytest.mark.parametrize("edit_format", tuple(_EXPECTED_TOOL))
def test_shipped_base_profile_keeps_selected_dialect_under_tool_cap(
    edit_format: str,
) -> None:
    profile = load_profile("_base", PROJECT_ROOT / "profiles")
    schemas = apply_profile_to_schemas(
        get_tool_schemas("minimal"),
        make_config(tools_edit_format=edit_format),
        SimpleNamespace(profile=profile),
    )
    names = {schema["function"]["name"] for schema in schemas}

    assert names & _EDIT_TOOLS == {_EXPECTED_TOOL[edit_format]}


def test_selected_dialect_changes_model_facing_spec_text() -> None:
    descriptions = {}
    for edit_format, expected_tool in _EXPECTED_TOOL.items():
        cfg = make_config(tools_edit_format=edit_format)
        client = SimpleNamespace(
            profile=SimpleNamespace(
                name="fixture",
                edit_format="exact",
                max_tools=99,
                simplify_schemas=False,
            )
        )
        schemas = apply_profile_to_schemas(
            get_tool_schemas("minimal"), cfg, client
        )
        selected = next(
            schema for schema in schemas
            if schema["function"]["name"] == expected_tool
        )
        descriptions[edit_format] = selected["function"]["description"]

    assert "old_str" in descriptions["exact"]
    assert "*** Begin Patch" in descriptions["apply_patch"]
    assert "standard unified diff" in descriptions["udiff"]
    assert "overwrite a file" in descriptions["whole"]
    assert len(set(descriptions.values())) == 4


@pytest.mark.parametrize("edit_format", tuple(_EXPECTED_TOOL))
def test_deferred_surface_registers_and_activates_only_selected_dialect(
    edit_format: str,
) -> None:
    cfg = make_config(
        tools_edit_format=edit_format,
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "edit", "glob", "grep", "done"),
    )
    profile = SimpleNamespace(
        name="fixture",
        edit_format="exact",
        max_tools=99,
        simplify_schemas=False,
    )

    surface = build_tool_surface(cfg, SimpleNamespace(profile=profile))

    assert set(surface.registered_names) & _EDIT_TOOLS == {
        _EXPECTED_TOOL[edit_format]
    }
    assert set(surface.default_active_names) & _EDIT_TOOLS == {
        _EXPECTED_TOOL[edit_format]
    }


@pytest.mark.parametrize("edit_format", tuple(_EXPECTED_TOOL))
def test_each_selected_dialect_executes_its_production_handler(
    tmp_path: Path, edit_format: str,
) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old\n")
    calls = {
        "exact": (
            "edit",
            {"path": "value.txt", "old_str": "old", "new_str": "new"},
        ),
        "apply_patch": (
            "apply_patch",
            {
                "patch": (
                    "*** Begin Patch\n*** Update File: value.txt\n"
                    "@@\n-old\n+new\n*** End Patch"
                )
            },
        ),
        "udiff": (
            "udiff",
            {
                "patch": (
                    "--- a/value.txt\n+++ b/value.txt\n"
                    "@@ -1 +1 @@\n-old\n+new"
                )
            },
        ),
        "whole": (
            "write",
            {"path": "value.txt", "content": "new\n"},
        ),
    }
    tool_name, arguments = calls[edit_format]

    result = dispatch(
        tool_name,
        arguments,
        cwd=str(tmp_path),
        cfg=make_config(tools_edit_format=edit_format),
        active_tools=(tool_name,),
    )

    assert not result.startswith("ERROR:"), result
    assert target.read_text() == "new\n"


def test_profile_edit_format_inherits_and_child_overrides(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    base = profiles / "_base"
    inherited = profiles / "inherited"
    overridden = profiles / "overridden"
    base.mkdir(parents=True)
    inherited.mkdir()
    overridden.mkdir()
    (base / "profile.toml").write_text(
        """\
[profile]
name = "_base"
inherits = ""
edit_format = "exact"

[reasoning_levels.off]
chat_template_kwargs = { enable_thinking = false }
"""
    )
    (inherited / "profile.toml").write_text(
        """\
[profile]
name = "inherited"
inherits = "_base"
"""
    )
    (overridden / "profile.toml").write_text(
        """\
[profile]
name = "overridden"
inherits = "_base"
edit_format = "udiff"
"""
    )

    assert load_profile("inherited", profiles).edit_format == "exact"
    assert load_profile("overridden", profiles).edit_format == "udiff"


def test_profile_edit_format_rejects_unknown_value(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    bad = profiles / "bad"
    base = profiles / "_base"
    bad.mkdir(parents=True)
    base.mkdir()
    (base / "profile.toml").write_text(
        """\
[profile]
name = "_base"
inherits = ""
edit_format = "exact"

[reasoning_levels.off]
chat_template_kwargs = { enable_thinking = false }
"""
    )
    (bad / "profile.toml").write_text(
        '[profile]\nname = "bad"\ninherits = "_base"\nedit_format = "magic"\n'
    )

    with pytest.raises(ValueError, match="edit_format must be one of"):
        load_profile("bad", profiles)


def test_tools_edit_format_config_knob_and_legacy_selector(tmp_path: Path) -> None:
    overlay = tmp_path / "format.toml"
    overlay.write_text('[tools]\nedit_format = "whole"\n')
    assert load_config(overlay).tools_edit_format == "whole"

    invalid = tmp_path / "invalid.toml"
    invalid.write_text('[tools]\nedit_format = "magic"\n')
    with pytest.raises(ValueError, match="tools.edit_format"):
        load_config(invalid)

    legacy = make_config(tools_apply_patch_enabled=True)
    profile = SimpleNamespace(
        name="fixture", edit_format="exact", max_tools=99,
        simplify_schemas=False,
    )
    names = {
        schema["function"]["name"]
        for schema in apply_profile_to_schemas(
            get_tool_schemas("minimal"), legacy,
            SimpleNamespace(profile=profile),
        )
    }
    assert names & _EDIT_TOOLS == {"apply_patch"}

    canonical = make_config(
        tools_edit_format="whole",
        tools_apply_patch_enabled=True,
    )
    canonical_names = {
        schema["function"]["name"]
        for schema in apply_profile_to_schemas(
            get_tool_schemas("minimal"), canonical,
            SimpleNamespace(profile=profile),
        )
    }
    assert canonical_names & _EDIT_TOOLS == {"write"}
