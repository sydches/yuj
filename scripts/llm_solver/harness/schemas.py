"""Tool schemas — parameter shapes from TOML + per-mode descriptions from .txt files.

Parameter shapes live in ``profiles/_base/tool_schemas.toml`` and are
invariant across description modes. Per-mode prose lives under
``profiles/_base/tool_descriptions/<mode>/<tool>.txt``. A mode only changes
the description the model sees; it never changes tool names or argument shapes.

Mode is resolved in this order:

1. ``get_tool_schemas(mode=...)`` argument (from ``Config.tool_desc``)
2. ``[experiment] tool_desc`` in ``config.toml``
3. Hardcoded fallback: ``"minimal"``

There is no env-var toggle and no import-time side effect. Callers pass the
mode explicitly.
"""
from __future__ import annotations

from functools import lru_cache
import copy
from pathlib import Path

from .._shared.paths import project_root
from .._shared.toml_compat import load_toml
from .tool_specs import (
    CODE_MODE_SCHEMA_TOOL_NAMES,
    DECLARED_SCHEMA_TOOL_NAMES,
    EXEC_CELL_API_TOOL_NAMES,
    SCHEMA_TOOL_NAMES,
)


def _schemas_toml_path() -> Path:
    return project_root() / "profiles" / "_base" / "tool_schemas.toml"


def _descriptions_root() -> Path:
    return project_root() / "profiles" / "_base" / "tool_descriptions"


@lru_cache(maxsize=1)
def _load_tool_specs() -> list[dict]:
    """Load the mode-invariant tool parameter specs from TOML."""
    data = load_toml(_schemas_toml_path())
    tools = data.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(
            f"{_schemas_toml_path()} has no [[tools]] entries"
        )
    for spec in tools:
        for key in ("name", "required", "properties"):
            if key not in spec:
                raise ValueError(
                    f"Tool spec in {_schemas_toml_path()} missing '{key}': {spec!r}"
                )
    declared_names = tuple(spec["name"] for spec in tools)
    if declared_names != DECLARED_SCHEMA_TOOL_NAMES:
        raise ValueError(
            "Tool schema names/order drifted from harness.tool_specs: "
            f"toml={declared_names}, specs={DECLARED_SCHEMA_TOOL_NAMES}"
        )
    return tools


@lru_cache(maxsize=8)
def _load_descriptions(mode: str) -> dict[str, str]:
    """Load every ``<tool>.txt`` under the given mode directory."""
    mode_dir = _descriptions_root() / mode
    if not mode_dir.is_dir():
        available = sorted(
            p.name for p in _descriptions_root().iterdir() if p.is_dir()
        )
        raise ValueError(
            f"Unknown tool description mode '{mode}'. Available: {available}"
        )
    return {
        path.stem: path.read_text().rstrip("\n")
        for path in mode_dir.glob("*.txt")
    }


@lru_cache(maxsize=16)
def get_tool_schemas(
    mode: str = "minimal", *, code_mode: bool = False,
) -> list[dict]:
    """Build the OpenAI-style tool-schema list for the given description mode.

    Every tool declared in ``tool_schemas.toml`` must have a matching
    ``<tool>.txt`` file in the mode directory. A missing file raises
    :class:`FileNotFoundError` with the full path; there is no silent fallback.
    """
    specs = _load_tool_specs()
    descriptions = _load_descriptions(mode)

    by_name = {spec["name"]: spec for spec in specs}
    selected_names = (
        CODE_MODE_SCHEMA_TOOL_NAMES if code_mode else SCHEMA_TOOL_NAMES
    )
    schemas: list[dict] = []
    for name in selected_names:
        spec = by_name[name]
        if name not in descriptions:
            raise FileNotFoundError(
                f"No description file for tool '{name}' in mode '{mode}': "
                f"{_descriptions_root() / mode / f'{name}.txt'}"
            )
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": {
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                    "additionalProperties": False,
                },
            },
        })
    return schemas


@lru_cache(maxsize=8)
def get_exec_cell_function_schemas(mode: str = "minimal") -> list[dict]:
    """Return the exact function catalog injected into a Python cell.

    The catalog reuses the native tool shapes and descriptions so discovery
    cannot drift from dispatch.  ``bash(background=...)`` is deliberately not
    part of a synchronous cell: a cell has one bounded lifetime and returns
    only after all of its calls finish.
    """
    native = {
        schema["function"]["name"]: schema
        for schema in get_tool_schemas(mode, code_mode=False)
    }
    selected = [
        copy.deepcopy(native[name]) for name in EXEC_CELL_API_TOOL_NAMES
    ]
    bash_schema = next(
        schema for schema in selected
        if schema["function"]["name"] == "bash"
    )
    bash_schema["function"]["parameters"]["properties"].pop(
        "background", None
    )
    return selected


__all__ = ["get_exec_cell_function_schemas", "get_tool_schemas"]
