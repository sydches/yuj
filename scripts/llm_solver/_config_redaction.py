"""Conservative value redaction shared by config metadata and inspection."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Iterable, Mapping as TypingMapping

from ._config_layers import ConfigValuePath, PathComponent


REDACTED_VALUE = "<redacted>"
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENSITIVE_WORDS = frozenset(
    {
        "authorization",
        "auth",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "token",
    }
)
_SENSITIVE_SUBTREE_NAMES = frozenset(
    {
        "extra_body",
        "extra_headers",
        "headers",
        "http_headers",
        "request_extra",
        "request_extras",
        "request_headers",
        "sandbox_env_set",
        "server_request_extra",
    }
)


def _normalized(component: PathComponent) -> str:
    if isinstance(component, int):
        return str(component)
    return "_".join(_WORD_RE.findall(component.lower()))


def _words(component: PathComponent) -> tuple[str, ...]:
    if isinstance(component, int):
        return ()
    return tuple(_WORD_RE.findall(component.lower()))


def _has_sequence(path: ConfigValuePath, sequence: tuple[str, ...]) -> bool:
    normalized = tuple(_normalized(component) for component in path)
    width = len(sequence)
    return any(
        normalized[index : index + width] == sequence
        for index in range(len(normalized) - width + 1)
    )


def _sensitive_subtree(path: ConfigValuePath) -> bool:
    names = {_normalized(component) for component in path}
    return bool(names & _SENSITIVE_SUBTREE_NAMES) or _has_sequence(
        path, ("server", "request_extra")
    ) or _has_sequence(path, ("sandbox", "env", "set"))


def _sensitive_leaf(path: ConfigValuePath) -> bool:
    if not path or isinstance(path[-1], int):
        return False
    words = _words(path[-1])
    if not words:
        return False
    if set(words) & _SENSITIVE_WORDS:
        return True
    compact = "".join(words)
    if "key" in words or (
        not compact.startswith("tokenizer")
        and compact.endswith(
            (
                "key",
                "token",
                "secret",
                "password",
                "passwd",
                "credential",
                "credentials",
            )
        )
    ):
        # Key-like paths are deliberately conservative. This covers api_key,
        # apiKey, x-api-key, private_key, accessToken, and future credential
        # spellings without treating tokenizer_id as a token or key.
        return True
    return False


def redact_config_value(
    value: object,
    *,
    path: Iterable[PathComponent] = (),
    environment_references: TypingMapping[ConfigValuePath, str] | None = None,
    force: bool = False,
) -> tuple[object, bool, tuple[str, ...]]:
    """Return a structure-preserving redacted value and explicit reasons."""
    current_path = tuple(path)
    references = environment_references or {}
    reasons: set[str] = set()
    environment_derived = current_path in references
    force = force or _sensitive_subtree(current_path) or _sensitive_leaf(current_path)
    if environment_derived:
        force = True
        reasons.add("environment-derived")

    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        redacted = False
        for key in sorted(value, key=str):
            child, child_redacted, child_reasons = redact_config_value(
                value[key],
                path=(*current_path, str(key)),
                environment_references=references,
                force=force,
            )
            output[str(key)] = child
            redacted = redacted or child_redacted
            reasons.update(child_reasons)
        return output, redacted, tuple(sorted(reasons))

    if isinstance(value, (list, tuple)):
        output_list: list[object] = []
        redacted = False
        for index, child_value in enumerate(value):
            child, child_redacted, child_reasons = redact_config_value(
                child_value,
                path=(*current_path, index),
                environment_references=references,
                force=force,
            )
            output_list.append(child)
            redacted = redacted or child_redacted
            reasons.update(child_reasons)
        return output_list, redacted, tuple(sorted(reasons))

    if force:
        if not environment_derived:
            reasons.add("sensitive-path")
        return REDACTED_VALUE, True, tuple(sorted(reasons))
    return value, False, ()


def sensitive_string_values(
    value: object,
    *,
    environment_references: TypingMapping[ConfigValuePath, str] | None = None,
) -> tuple[str, ...]:
    """Collect strings that the output policy would redact."""
    references = environment_references or {}
    values: set[str] = set()

    def visit(child: object, path: ConfigValuePath) -> None:
        if isinstance(child, Mapping):
            for key, grandchild in child.items():
                visit(grandchild, (*path, str(key)))
            return
        if isinstance(child, (list, tuple)):
            for index, grandchild in enumerate(child):
                visit(grandchild, (*path, index))
            return
        _safe, redacted, _reasons = redact_config_value(
            child,
            path=path,
            environment_references=references,
        )
        if redacted and isinstance(child, str) and child:
            values.add(child)

    visit(value, ())
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def environment_string_values(
    value: object,
    *,
    environment_references: TypingMapping[ConfigValuePath, str],
) -> tuple[str, ...]:
    """Collect the resolved string at each value-free environment reference."""
    values: set[str] = set()
    for path in environment_references:
        child = value
        for component in path:
            if isinstance(component, int) and isinstance(child, (list, tuple)):
                child = child[component]
            elif isinstance(component, str) and isinstance(child, Mapping):
                child = child[component]
            else:
                break
        else:
            if isinstance(child, str) and child:
                values.add(child)
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def redact_sensitive_text(
    message: object,
    *,
    sensitive_values: Iterable[str] = (),
    unquoted_values: Iterable[str] = (),
) -> str:
    """Remove sensitive values from an exception or diagnostic string."""
    text = str(message)
    representations: set[str] = set()
    for value in sensitive_values:
        if not value:
            continue
        representations.update(
            {
                repr(value),
                # JSON escaping covers errors that serialize a rejected value.
                json.dumps(value, ensure_ascii=False),
            }
        )
    # Environment-derived values can also appear unquoted in third-party
    # errors, so remove those exact strings as well. Literal sensitive values
    # use the quoted forms above to avoid corrupting unrelated range/path text.
    representations.update(value for value in unquoted_values if value)
    for representation in sorted(
        representations,
        key=lambda item: (-len(item), item),
    ):
        text = text.replace(representation, REDACTED_VALUE)
    return text


__all__ = [
    "REDACTED_VALUE",
    "environment_string_values",
    "redact_config_value",
    "redact_sensitive_text",
    "sensitive_string_values",
]
