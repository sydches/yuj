"""Session-local deferred model-tool surface.

The handler registry remains complete.  This module owns only the schemas
that are visible to the model on a request and the additive activation state
that changes that visible set.  It does not execute tools.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .._shared.edit_formats import EDIT_FORMAT_TOOL_NAMES


LOADER_TOOL_NAME = "load_tools"
TOOL_LOADING_ERROR_VERSION = 1


class ToolLoadingError(ValueError):
    """A loader request cannot be applied atomically."""

    def __init__(self, message: str, *, requested: Sequence[str] = ()):
        self.requested = tuple(requested)
        super().__init__(message)


@dataclass(frozen=True)
class ToolActivation:
    """One successful additive loader operation."""

    requested: tuple[str, ...]
    activated: tuple[str, ...]
    already_active: tuple[str, ...]
    active_tools: tuple[str, ...]


class ToolSurface:
    """Registered schemas plus the request-visible active subset."""

    def __init__(
        self,
        registered_schemas: Sequence[dict],
        *,
        lazy_loading_enabled: bool,
        active_default: Iterable[str],
        max_active_tools: int | None = None,
    ) -> None:
        self.lazy_loading_enabled = bool(lazy_loading_enabled)
        if (
            max_active_tools is not None
            and (
                isinstance(max_active_tools, bool)
                or not isinstance(max_active_tools, int)
                or max_active_tools < 1
            )
        ):
            raise ValueError("max_active_tools must be a positive integer or None")
        self.max_active_tools = max_active_tools
        self._registered_schemas = tuple(registered_schemas)
        names = tuple(
            str(schema.get("function", {}).get("name", ""))
            for schema in self._registered_schemas
        )
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("registered tool schemas must have unique names")
        self._registered_names = names
        self._registered_name_set = frozenset(names)

        configured = tuple(dict.fromkeys(str(name) for name in active_default))
        registered_edit_tools = tuple(
            name for name in names if name in EDIT_FORMAT_TOOL_NAMES
        )
        if len(registered_edit_tools) == 1:
            selected_edit_tool = registered_edit_tools[0]
            configured = tuple(dict.fromkeys(
                selected_edit_tool if name in EDIT_FORMAT_TOOL_NAMES else name
                for name in configured
            ))
        if self.lazy_loading_enabled and LOADER_TOOL_NAME not in self._registered_name_set:
            raise ValueError(
                "lazy tool loading requires the load_tools schema and handler"
            )
        mandatory = {"done"}
        if "ask_user" in self._registered_name_set:
            mandatory.add("ask_user")
        if self.lazy_loading_enabled:
            mandatory.add(LOADER_TOOL_NAME)
        selected = set(configured) | mandatory
        self._default_active_names = tuple(
            name
            for name in self._registered_names
            if not self.lazy_loading_enabled or name in selected
        )
        if (
            self.max_active_tools is not None
            and len(self._default_active_names) > self.max_active_tools
        ):
            raise ValueError(
                "tools.active_default plus mandatory tools exceeds the model "
                f"profile active-tool limit {self.max_active_tools}"
            )
        self._active_name_set = set(self._default_active_names)

    @property
    def registered_schemas(self) -> tuple[dict, ...]:
        return self._registered_schemas

    @property
    def registered_names(self) -> tuple[str, ...]:
        return self._registered_names

    @property
    def default_active_names(self) -> tuple[str, ...]:
        return self._default_active_names

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self._registered_names if name in self._active_name_set
        )

    @property
    def active_schemas(self) -> list[dict]:
        active = self._active_name_set
        return [
            schema
            for schema in self._registered_schemas
            if schema["function"]["name"] in active
        ]

    def is_hidden(self, name: str, *, active_names: Iterable[str] | None = None) -> bool:
        """Return whether a registered name was absent from this request."""
        if not self.lazy_loading_enabled or name not in self._registered_name_set:
            return False
        visible = self._active_name_set if active_names is None else set(active_names)
        return name not in visible

    def activate(self, names: object) -> ToolActivation:
        """Atomically activate registered names, preserving registry order."""
        if not self.lazy_loading_enabled:
            raise ToolLoadingError("deferred tool loading is disabled")
        if not isinstance(names, (list, tuple)):
            raise ToolLoadingError("names must be an array of tool names")
        requested_list: list[str] = []
        for raw_name in names:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ToolLoadingError(
                    "every requested tool name must be a non-empty string"
                )
            name = raw_name.strip()
            if name not in requested_list:
                requested_list.append(name)
        if not requested_list:
            raise ToolLoadingError("names must contain at least one tool name")

        unavailable = [
            name for name in requested_list if name not in self._registered_name_set
        ]
        if unavailable:
            raise ToolLoadingError(
                "tools are unavailable in the current config/profile: "
                + ", ".join(unavailable),
                requested=requested_list,
            )

        before = set(self._active_name_set)
        newly_requested = [name for name in requested_list if name not in before]
        if (
            self.max_active_tools is not None
            and len(before) + len(newly_requested) > self.max_active_tools
        ):
            raise ToolLoadingError(
                "activation would exceed the model profile active-tool limit "
                f"{self.max_active_tools}; currently active {len(before)}, "
                f"newly requested {len(newly_requested)}",
                requested=requested_list,
            )
        self._active_name_set.update(requested_list)
        requested = tuple(requested_list)
        return ToolActivation(
            requested=requested,
            activated=tuple(
                name
                for name in self._registered_names
                if name in requested and name not in before
            ),
            already_active=tuple(
                name
                for name in self._registered_names
                if name in requested and name in before
            ),
            active_tools=self.active_names,
        )


def inactive_tool_error(tool_name: str) -> str:
    """Return the stable model-visible envelope for one hidden call."""
    payload = {
        "error": {
            "loader": LOADER_TOOL_NAME,
            "message": (
                f"Tool {tool_name!r} is registered but not active. "
                f"Call {LOADER_TOOL_NAME} with names=[{tool_name!r}] first."
            ),
            "tool": tool_name,
            "type": "tool_not_active",
            "version": TOOL_LOADING_ERROR_VERSION,
        }
    }
    return "ERROR: " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def loader_error(exc: ToolLoadingError, surface: ToolSurface) -> str:
    """Return a stable error envelope for an invalid loader request."""
    payload = {
        "error": {
            "active_tools": list(surface.active_names),
            "available_tools": list(surface.registered_names),
            "loader": LOADER_TOOL_NAME,
            "message": str(exc),
            "requested": list(exc.requested),
            "type": "tool_loading_reject",
            "version": TOOL_LOADING_ERROR_VERSION,
        }
    }
    return "ERROR: " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def loader_success(activation: ToolActivation) -> str:
    """Render one deterministic, small-model-readable activation result."""
    activated = ", ".join(activation.activated) or "none"
    already = ", ".join(activation.already_active) or "none"
    active = ", ".join(activation.active_tools)
    return (
        f"Activated tools: {activated}\n"
        f"Already active: {already}\n"
        f"Active tools now: {active}"
    )


def replace_tool_surface(
    previous: ToolSurface,
    registered_schemas: Sequence[dict],
    *,
    lazy_loading_enabled: bool,
    active_default: Iterable[str],
    max_active_tools: int | None = None,
) -> ToolSurface:
    """Rebuild a profile/config surface while retaining additive state."""
    updated = ToolSurface(
        registered_schemas,
        lazy_loading_enabled=lazy_loading_enabled,
        active_default=active_default,
        max_active_tools=max_active_tools,
    )
    if previous.lazy_loading_enabled and updated.lazy_loading_enabled:
        retained = [
            name
            for name in previous.active_names
            if name in updated.registered_names
        ]
        if retained:
            updated.activate(retained)
    return updated


def estimate_tool_block_tokens(
    schemas: Sequence[dict], tokenizer=None
) -> tuple[int, str]:
    """Count one initial request's schema block, with a declared method."""
    tools = list(schemas)
    if tokenizer is not None:
        probe = [{"role": "system", "content": ""}]
        with_tools = int(tokenizer.count(probe, tools=tools))
        without_tools = int(tokenizer.count(probe, tools=None))
        return max(0, with_tools - without_tools), "chat_template_delta"
    serialized = json.dumps(
        tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (len(serialized) + 3) // 4, "chars_div_4"


__all__ = [
    "LOADER_TOOL_NAME",
    "ToolActivation",
    "ToolLoadingError",
    "ToolSurface",
    "estimate_tool_block_tokens",
    "inactive_tool_error",
    "loader_error",
    "loader_success",
    "replace_tool_surface",
]
