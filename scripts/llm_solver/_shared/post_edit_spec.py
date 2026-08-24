"""Shared post-edit check spec parsing and validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_POST_EDIT_TRIGGERS: frozenset[str] = frozenset({
    "edit",
    "write",
    "apply_patch",
    "udiff",
})
VALID_POST_EDIT_ON_FAIL: frozenset[str] = frozenset({
    "append",
    "warn",
    "block",
})


@dataclass(frozen=True)
class PostEditCheckSpec:
    """Parsed post-edit check declaration."""

    name: str
    triggers: frozenset[str]
    when: str
    cmd: str
    on_fail: str


def _trigger_pieces(trigger: str) -> frozenset[str]:
    return frozenset(piece for piece in trigger.split("|") if piece)


def validate_post_edit_check_dict(spec: Any) -> None:
    """Config-load validation for raw post_edit_check tables."""
    if not isinstance(spec, dict):
        return
    trigger = spec.get("trigger", "")
    if not isinstance(trigger, str):
        raise ValueError(
            f"post_edit_check trigger must be a string, got "
            f"{type(trigger).__name__}: {spec!r}"
        )
    for piece in _trigger_pieces(trigger):
        if piece not in VALID_POST_EDIT_TRIGGERS:
            raise ValueError(
                f"post_edit_check {spec.get('name', '?')!r} has unknown "
                f"trigger {piece!r}; valid: {sorted(VALID_POST_EDIT_TRIGGERS)}"
            )
    on_fail = spec.get("on_fail", "")
    if on_fail and on_fail not in VALID_POST_EDIT_ON_FAIL:
        raise ValueError(
            f"post_edit_check {spec.get('name', '?')!r} has unknown "
            f"on_fail {on_fail!r}; valid: {sorted(VALID_POST_EDIT_ON_FAIL)}"
        )


def parse_post_edit_check_spec(spec: Any) -> PostEditCheckSpec:
    """Parse one raw post_edit_check table into a mechanical spec object."""
    if not isinstance(spec, dict):
        raise ValueError(
            f"post_edit_check entry must be a table, got {type(spec).__name__}"
        )
    for key in ("name", "trigger", "cmd", "on_fail"):
        if key not in spec:
            raise ValueError(
                f"post_edit_check entry missing required key {key!r}: {spec!r}"
            )

    trigger = spec["trigger"]
    if not isinstance(trigger, str):
        raise ValueError(
            f"post_edit_check trigger must be a string, got "
            f"{type(trigger).__name__}: {spec!r}"
        )
    triggers = _trigger_pieces(trigger)
    unknown = sorted(
        piece for piece in triggers if piece not in VALID_POST_EDIT_TRIGGERS
    )
    if unknown:
        raise ValueError(
            f"post_edit_check {spec.get('name', '?')!r} has unknown "
            f"trigger {unknown[0]!r}; valid: {sorted(VALID_POST_EDIT_TRIGGERS)}"
        )

    on_fail = spec["on_fail"]
    if on_fail not in VALID_POST_EDIT_ON_FAIL:
        raise ValueError(f"invalid on_fail {on_fail!r} in {spec!r}")

    cmd = spec["cmd"]
    if not isinstance(cmd, str):
        raise ValueError(
            f"post_edit_check {spec.get('name', '?')!r} cmd must be a string"
        )

    when = spec.get("when", "")
    if not isinstance(when, str):
        raise ValueError(
            f"post_edit_check {spec.get('name', '?')!r} when must be a string"
        )

    return PostEditCheckSpec(
        name=str(spec["name"]),
        triggers=triggers,
        when=when,
        cmd=cmd,
        on_fail=on_fail,
    )


__all__ = [
    "PostEditCheckSpec",
    "VALID_POST_EDIT_ON_FAIL",
    "VALID_POST_EDIT_TRIGGERS",
    "parse_post_edit_check_spec",
    "validate_post_edit_check_dict",
]
