"""Declarative per-tool allow/ask/deny policy.

Rules are compiled in declaration order and the last matching rule wins.
``bash`` rules inspect ``cmd``; file-oriented rules inspect ``path``.  The
match value is used only for policy evaluation and is never included in trace
fields or denial envelopes.

An ``ask`` decision is meaningful only in assistant mode with an approval
transport.  Measurement mode always resolves it to ``deny``.  If an assistant
caller has no approval transport, ``ask_fallback`` explicitly chooses
``allow`` or ``deny`` rather than silently guessing.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType


PERMISSION_DECISIONS = ("allow", "ask", "deny")
PERMISSION_ASK_FALLBACKS = ("deny", "allow")
PERMISSION_RUNTIME_MODES = ("measurement", "assistant")
PERMISSION_DENIED_ERROR_TYPE = "permission_denied"
PERMISSION_ENVELOPE_VERSION = 1
DEFAULT_PERMISSION_DECISION = "allow"
DEFAULT_PERMISSION_RULE = "<default>"

# One canonical argument surface per shipped tool.  Defaults mirror handler
# defaults so a rule sees the action that would execute, not a missing-key
# implementation detail.
PERMISSION_MATCH_FIELDS = MappingProxyType(
    {
        "apply_patch": ("patch", ""),
        "apply_subagent": ("task_id", ""),
        "udiff": ("patch", ""),
        "bash": ("cmd", ""),
        "bash_kill": ("proc_id", ""),
        "bash_poll": ("proc_id", ""),
        "checkpoint": ("goal", ""),
        "done": ("message", ""),
        "edit": ("path", ""),
        "exit_plan_mode": ("message", ""),
        "exec_cell": ("source", ""),
        "get_function_details": ("names", []),
        "glob": ("path", "."),
        "grep": ("path", "."),
        "list_definitions": ("path", ""),
        "load_tools": ("names", ()),
        "lsp": ("path", ""),
        "list_functions": ("arguments", ""),
        "notebook_edit": ("path", ""),
        "structural_edit": ("path", ""),
        "structural_search": ("path", ""),
        "subagent_changes": ("task_id", ""),
        "read": ("path", ""),
        "rewind": ("report", ""),
        "run_tests": ("path", ""),
        "task": ("agent", ""),
        "terminal_start": ("cmd", ""),
        "terminal_io": ("input", ""),
        "think": ("thought", ""),
        "write_todos": ("todos", ()),
        "write": ("path", ""),
    }
)


class PermissionPolicyError(ValueError):
    """A permission rule table or evaluation input is invalid."""


class PermissionDenied(RuntimeError):
    """A declarative permission rule denied a tool call."""

    def __init__(self, resolution: "PermissionResolution"):
        self.resolution = resolution
        super().__init__(resolution.denial_reason())

    def error_envelope(self) -> str:
        return self.resolution.denial_envelope()


class PermissionApprovalRequired(RuntimeError):
    """A declarative permission rule requires operator approval."""

    def __init__(self, resolution: "PermissionResolution"):
        self.resolution = resolution
        super().__init__(resolution.approval_reason())


@dataclass(frozen=True)
class PermissionRule:
    """One compiled rule in global declaration order."""

    tool_pattern: str
    argument_pattern: str
    decision: str
    ordinal: int
    layer: int

    def matches(self, tool_name: str, argument: str) -> bool:
        return _glob_matches(self.tool_pattern, tool_name) and _glob_matches(
            self.argument_pattern, argument
        )


@dataclass(frozen=True)
class PermissionResolution:
    """Configured and effective decision for one tool call."""

    tool: str
    argument_field: str
    configured_decision: str
    decision: str
    rule: str
    rule_tool: str
    rule_ordinal: int | None
    runtime_mode: str
    approval_available: bool
    fallback_reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def approval_required(self) -> bool:
        return self.decision == "ask"

    @property
    def denied(self) -> bool:
        return self.decision == "deny"

    def trace_fields(self) -> dict[str, str]:
        """Return exactly the required fields for a ``permission`` event."""
        return {
            "tool": self.tool,
            "rule": self.rule,
            "decision": self.decision,
        }

    def approval_reason(self) -> str:
        if not self.approval_required:
            raise ValueError("permission resolution does not require approval")
        return (
            f"permission rule {self.rule!r} requires operator approval "
            f"for tool {self.tool!r}"
        )

    def denial_reason(self) -> str:
        if not self.denied:
            raise ValueError("permission resolution is not denied")
        if self.configured_decision == "ask":
            return (
                f"permission rule {self.rule!r} requires approval, but "
                f"{self.fallback_reason or 'policy'} resolved it to deny"
            )
        return f"permission rule {self.rule!r} denies tool {self.tool!r}"

    def denial_envelope(self) -> str:
        """Render a value-free error-ladder-compatible denial."""
        payload = {
            "error": {
                "type": PERMISSION_DENIED_ERROR_TYPE,
                "version": PERMISSION_ENVELOPE_VERSION,
                "tool": self.tool,
                "rule": self.rule,
                "decision": self.decision,
                "message": self.denial_reason(),
            }
        }
        return "ERROR: " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


@dataclass(frozen=True)
class PermissionDispatchResult:
    """A handler result paired with the policy decision that allowed it."""

    result: str
    resolution: PermissionResolution


@dataclass(frozen=True)
class PermissionPolicy:
    """Immutable ordered permission rules with an explicit default."""

    rules: tuple[PermissionRule, ...]
    default_decision: str = DEFAULT_PERMISSION_DECISION

    @classmethod
    def from_rule_tables(
        cls,
        *rule_tables: object,
        default_decision: object = DEFAULT_PERMISSION_DECISION,
    ) -> "PermissionPolicy":
        """Compile base and override tables; later matching entries win.

        A table has the public TOML shape ``{tool = {glob = decision}}``.
        A tool value may be a decision string as shorthand for ``{"*":
        decision}``.  A whole-table decision string is a global catch-all.
        Supplying multiple tables appends later layers, which provides an
        agent/session override without mutating the base policy.
        """
        normalized_default = normalize_permission_decision(
            default_decision, path="permissions.default"
        )
        compiled: list[PermissionRule] = []
        ordinal = 0
        for layer, table in enumerate(rule_tables):
            if table is None:
                continue
            if isinstance(table, str):
                compiled.append(
                    PermissionRule(
                        "*",
                        "*",
                        normalize_permission_decision(
                            table, path=f"permissions.rules[{layer}]"
                        ),
                        ordinal,
                        layer,
                    )
                )
                ordinal += 1
                continue
            if not isinstance(table, Mapping):
                raise PermissionPolicyError(
                    f"permissions.rules[{layer}] must be a table or decision string"
                )
            for tool_pattern, entries in table.items():
                _validate_pattern(
                    tool_pattern,
                    path=f"permissions.rules[{layer}] tool key",
                    allow_empty=False,
                )
                if isinstance(entries, str):
                    entries = {"*": entries}
                if not isinstance(entries, Mapping):
                    raise PermissionPolicyError(
                        f"permissions.rules.{tool_pattern} must be a table "
                        "or decision string"
                    )
                for argument_pattern, raw_decision in entries.items():
                    _validate_pattern(
                        argument_pattern,
                        path=f"permissions.rules.{tool_pattern} pattern",
                        allow_empty=True,
                    )
                    decision = normalize_permission_decision(
                        raw_decision,
                        path=(
                            f"permissions.rules.{tool_pattern}."
                            f"{argument_pattern}"
                        ),
                    )
                    compiled.append(
                        PermissionRule(
                            tool_pattern,
                            argument_pattern,
                            decision,
                            ordinal,
                            layer,
                        )
                    )
                    ordinal += 1
        return cls(tuple(compiled), normalized_default)

    def evaluate(
        self,
        *,
        tool_name: object,
        arguments: object,
        runtime_mode: object,
        ask_fallback: object = "deny",
        approval_available: bool = True,
    ) -> PermissionResolution:
        if not isinstance(tool_name, str) or not tool_name:
            raise PermissionPolicyError("tool_name must be a non-empty string")
        mode = normalize_permission_runtime_mode(runtime_mode)
        fallback = normalize_ask_fallback(ask_fallback)
        argument_field, argument = permission_match_argument(tool_name, arguments)

        selected: PermissionRule | None = None
        for rule in self.rules:
            if rule.matches(tool_name, argument):
                selected = rule

        configured = (
            selected.decision if selected is not None else self.default_decision
        )
        effective = configured
        fallback_reason = None
        if configured == "ask":
            if mode == "measurement":
                effective = "deny"
                fallback_reason = "measurement mode"
            elif not approval_available:
                effective = fallback
                fallback_reason = "approval transport unavailable"

        return PermissionResolution(
            tool=tool_name,
            argument_field=argument_field,
            configured_decision=configured,
            decision=effective,
            rule=(
                selected.argument_pattern
                if selected is not None
                else DEFAULT_PERMISSION_RULE
            ),
            rule_tool=(selected.tool_pattern if selected is not None else ""),
            rule_ordinal=(selected.ordinal if selected is not None else None),
            runtime_mode=mode,
            approval_available=bool(approval_available),
            fallback_reason=fallback_reason,
        )


def normalize_permission_decision(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PermissionPolicyError(f"{path} must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in PERMISSION_DECISIONS:
        raise PermissionPolicyError(
            f"{path} must be one of: {', '.join(PERMISSION_DECISIONS)}"
        )
    return normalized


def normalize_ask_fallback(value: object) -> str:
    normalized = normalize_permission_decision(
        value, path="permissions.ask_fallback"
    )
    if normalized not in PERMISSION_ASK_FALLBACKS:
        raise PermissionPolicyError(
            "permissions.ask_fallback must be one of: "
            + ", ".join(PERMISSION_ASK_FALLBACKS)
        )
    return normalized


def normalize_permission_runtime_mode(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PermissionPolicyError("runtime_mode must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in PERMISSION_RUNTIME_MODES:
        raise PermissionPolicyError(
            "runtime_mode must be one of: "
            + ", ".join(PERMISSION_RUNTIME_MODES)
        )
    return normalized


def permission_match_argument(
    tool_name: str, arguments: object
) -> tuple[str, str]:
    """Return the canonical field name and value used by permission globs."""
    if not isinstance(arguments, Mapping):
        raise PermissionPolicyError("tool arguments must be a mapping")
    selector = PERMISSION_MATCH_FIELDS.get(tool_name)
    if selector is not None:
        field, default = selector
        value = arguments.get(field, default)
        if isinstance(value, str):
            return field, value
        # Schema validation normally intercepts this first.  A deterministic
        # JSON rendering keeps a direct policy caller fail-closed: catch-all
        # rules still match, while a malformed value cannot imitate a path or
        # command string accidentally.
        try:
            return field, json.dumps(
                value, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
        except (TypeError, ValueError) as exc:
            raise PermissionPolicyError(
                f"tool argument {field!r} is not JSON-compatible"
            ) from exc

    try:
        rendered = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PermissionPolicyError("tool arguments are not JSON-compatible") from exc
    return "arguments", rendered


def require_tool_permission(
    *,
    policy: PermissionPolicy,
    tool_name: object,
    arguments: object,
    runtime_mode: object,
    ask_fallback: object = "deny",
    approval_available: bool = True,
) -> PermissionResolution:
    """Return an allow decision or raise before any handler can execute."""
    resolution = policy.evaluate(
        tool_name=tool_name,
        arguments=arguments,
        runtime_mode=runtime_mode,
        ask_fallback=ask_fallback,
        approval_available=approval_available,
    )
    if resolution.approval_required:
        raise PermissionApprovalRequired(resolution)
    if resolution.denied:
        raise PermissionDenied(resolution)
    return resolution


def permission_guarded_dispatch(
    *,
    policy: PermissionPolicy,
    tool_name: object,
    arguments: object,
    runtime_mode: object,
    handler: Callable[[], str],
    ask_fallback: object = "deny",
    approval_available: bool = True,
) -> PermissionDispatchResult:
    """Invoke ``handler`` only after policy returns an effective allow."""
    resolution = require_tool_permission(
        policy=policy,
        tool_name=tool_name,
        arguments=arguments,
        runtime_mode=runtime_mode,
        ask_fallback=ask_fallback,
        approval_available=approval_available,
    )
    return PermissionDispatchResult(handler(), resolution)


def _validate_pattern(value: object, *, path: str, allow_empty: bool) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise PermissionPolicyError(f"{path} must be {qualifier}")


@lru_cache(maxsize=512)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile only the documented ``*`` and ``?`` wildcard language."""
    pieces: list[str] = ["\\A"]
    for character in pattern:
        if character == "*":
            pieces.append(".*")
        elif character == "?":
            pieces.append(".")
        else:
            pieces.append(re.escape(character))
    pieces.append("\\Z")
    return re.compile("".join(pieces), re.DOTALL)


def _glob_matches(pattern: str, value: str) -> bool:
    return _compile_glob(pattern).fullmatch(value) is not None


__all__ = [
    "DEFAULT_PERMISSION_DECISION",
    "DEFAULT_PERMISSION_RULE",
    "PERMISSION_ASK_FALLBACKS",
    "PERMISSION_DECISIONS",
    "PERMISSION_DENIED_ERROR_TYPE",
    "PERMISSION_ENVELOPE_VERSION",
    "PERMISSION_MATCH_FIELDS",
    "PERMISSION_RUNTIME_MODES",
    "PermissionApprovalRequired",
    "PermissionDenied",
    "PermissionDispatchResult",
    "PermissionPolicy",
    "PermissionPolicyError",
    "PermissionResolution",
    "PermissionRule",
    "normalize_ask_fallback",
    "normalize_permission_decision",
    "normalize_permission_runtime_mode",
    "permission_guarded_dispatch",
    "permission_match_argument",
    "require_tool_permission",
]
