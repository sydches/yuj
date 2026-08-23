"""Parse and validate repository-owned mid-stream rule files."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .._shared.toml_compat import tomllib


_FRONTMATTER_FENCE = "+++"
_VALID_INTERRUPT_MODES = frozenset({"never", "prose-only", "tool-only", "always"})
_VALID_REPEAT_MODES = frozenset({"once", "after-gap"})
_VALID_KEYS = frozenset({
    "name", "condition", "astCondition", "scope", "globs",
    "interruptMode", "repeatMode", "repeatGap",
})
_TOOL_SCOPE_RE = re.compile(
    r"^tool(?::(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)(?:\((?P<glob>.+)\))?)?$"
)
_METAVAR_RE = re.compile(r"(?<!\$)\$([A-Z][A-Z0-9_]*)")
_META_PREFIX = "__yuj_stream_meta_"


class StreamRuleError(ValueError):
    """A rule file or structural matcher cannot be used safely."""


@dataclass(frozen=True, slots=True)
class StreamRuleScope:
    source: str
    tool_name: str = ""
    path_glob: str = ""

    @property
    def label(self) -> str:
        if self.source != "tool" or not self.tool_name:
            return self.source
        suffix = f"({self.path_glob})" if self.path_glob else ""
        return f"tool:{self.tool_name}{suffix}"


@dataclass(frozen=True, slots=True)
class StreamRule:
    name: str
    conditions: tuple[re.Pattern[str], ...]
    condition_sources: tuple[str, ...]
    ast_conditions: tuple[str, ...]
    scopes: tuple[StreamRuleScope, ...]
    globs: tuple[str, ...]
    interrupt_mode: str
    repeat_mode: str
    repeat_gap: int | None
    body: str
    source_path: str


@dataclass(frozen=True, slots=True)
class LoadedStreamRules:
    rules: tuple[StreamRule, ...]
    files: tuple[dict[str, object], ...]


def _string_list(value: object, *, field: str, source_path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        values = tuple(value)
    else:
        raise StreamRuleError(
            f"{source_path}: {field} must be a string or array of strings"
        )
    cleaned = tuple(item.strip() for item in values)
    if any(not item for item in cleaned):
        raise StreamRuleError(f"{source_path}: {field} entries must be non-empty")
    return cleaned


def _split_scope_string(value: str) -> tuple[str, ...]:
    """Split comma-separated scope shorthand without splitting inside ``(...)``."""
    out: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise StreamRuleError(f"invalid scope {value!r}: unmatched ')'")
        elif char == "," and depth == 0:
            out.append(value[start:index].strip())
            start = index + 1
    if depth:
        raise StreamRuleError(f"invalid scope {value!r}: unmatched '('")
    out.append(value[start:].strip())
    return tuple(out)


def _parse_scopes(value: object, *, source_path: str) -> tuple[StreamRuleScope, ...]:
    if value is None:
        tokens = ("text", "tool")
    elif isinstance(value, str):
        tokens = _split_scope_string(value)
    elif isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        tokens = tuple(str(item).strip() for item in value)
    else:
        raise StreamRuleError(
            f"{source_path}: scope must be a comma-separated string or array of strings"
        )
    if not tokens or any(not token for token in tokens):
        raise StreamRuleError(f"{source_path}: scope must contain at least one token")

    scopes: list[StreamRuleScope] = []
    for token in tokens:
        if token in {"text", "thinking"}:
            scope = StreamRuleScope(token)
        else:
            normalized = "tool" if token == "toolcall" else token
            match = _TOOL_SCOPE_RE.fullmatch(normalized)
            if match is None:
                raise StreamRuleError(
                    f"{source_path}: invalid scope token {token!r}; expected "
                    "text, thinking, tool, or tool:<name>(<glob>)"
                )
            scope = StreamRuleScope(
                "tool", match.group("name") or "", match.group("glob") or ""
            )
        if scope not in scopes:
            scopes.append(scope)
    return tuple(scopes)


def _frontmatter(text: str, *, source_path: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
        raise StreamRuleError(
            f"{source_path}: missing opening {_FRONTMATTER_FENCE!r} "
            "TOML frontmatter fence"
        )
    try:
        end = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONTMATTER_FENCE
        )
    except StopIteration as exc:
        raise StreamRuleError(
            f"{source_path}: missing closing {_FRONTMATTER_FENCE!r} "
            "TOML frontmatter fence"
        ) from exc
    source = "\n".join(lines[1:end]).strip()
    body = "\n".join(lines[end + 1:]).strip()
    try:
        parsed = tomllib.loads(source)
    except Exception as exc:
        raise StreamRuleError(
            f"{source_path}: invalid TOML frontmatter: {exc}"
        ) from exc
    if not isinstance(parsed, dict):  # pragma: no cover - tomllib always maps
        raise StreamRuleError(f"{source_path}: frontmatter must be a TOML table")
    return parsed, body


def parse_stream_rule(
    text: str,
    *,
    source_path: str,
    default_name: str = "",
) -> StreamRule:
    """Parse and validate one stream-rule Markdown file."""
    fm, body = _frontmatter(text, source_path=source_path)
    unknown = sorted(set(fm) - _VALID_KEYS)
    if unknown:
        raise StreamRuleError(
            f"{source_path}: unknown stream-rule frontmatter key(s): "
            f"{', '.join(unknown)}"
        )
    raw_name = fm.get("name", default_name)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise StreamRuleError(f"{source_path}: name must be a non-empty string")
    name = raw_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise StreamRuleError(
            f"{source_path}: name {name!r} may contain only letters, digits, "
            "'.', '_', and '-'"
        )
    if not body:
        raise StreamRuleError(f"{source_path}: rule body must be non-empty")

    condition_sources = (
        _string_list(fm["condition"], field="condition", source_path=source_path)
        if "condition" in fm else ()
    )
    ast_conditions = (
        _string_list(
            fm["astCondition"], field="astCondition", source_path=source_path
        )
        if "astCondition" in fm else ()
    )
    if not condition_sources and not ast_conditions:
        raise StreamRuleError(
            f"{source_path}: at least one of condition or astCondition is required"
        )
    compiled: list[re.Pattern[str]] = []
    for condition in condition_sources:
        try:
            compiled.append(re.compile(condition))
        except re.error as exc:
            raise StreamRuleError(
                f"{source_path}: invalid condition regex {condition!r}: {exc}"
            ) from exc

    scopes = _parse_scopes(fm.get("scope"), source_path=source_path)
    globs = (
        _string_list(fm["globs"], field="globs", source_path=source_path)
        if "globs" in fm else ()
    )
    interrupt_mode = fm.get("interruptMode", "always")
    if not isinstance(interrupt_mode, str) or interrupt_mode not in _VALID_INTERRUPT_MODES:
        raise StreamRuleError(
            f"{source_path}: interruptMode must be one of "
            f"{', '.join(sorted(_VALID_INTERRUPT_MODES))}"
        )
    repeat_mode = fm.get("repeatMode", "once")
    if not isinstance(repeat_mode, str) or repeat_mode not in _VALID_REPEAT_MODES:
        raise StreamRuleError(
            f"{source_path}: repeatMode must be 'once' or 'after-gap'"
        )
    repeat_gap = fm.get("repeatGap")
    if repeat_gap is not None and (
        isinstance(repeat_gap, bool)
        or not isinstance(repeat_gap, int)
        or repeat_gap < 1
    ):
        raise StreamRuleError(f"{source_path}: repeatGap must be an integer >= 1")
    if repeat_gap is not None and repeat_mode != "after-gap":
        raise StreamRuleError(
            f"{source_path}: repeatGap is only valid with repeatMode='after-gap'"
        )
    if ast_conditions:
        tool_scopes = [scope for scope in scopes if scope.source == "tool"]
        reachable = any(
            not scope.tool_name or scope.tool_name in {"edit", "write"}
            for scope in tool_scopes
        )
        if not reachable and not condition_sources:
            raise StreamRuleError(
                f"{source_path}: astCondition is reachable only on edit/write tool scopes"
            )
        for pattern in ast_conditions:
            substituted = _METAVAR_RE.sub(
                lambda match: _META_PREFIX + match.group(1), pattern
            )
            if "$" in substituted:
                raise StreamRuleError(
                    f"{source_path}: astCondition {pattern!r} contains an unsupported "
                    "metavariable; use $NAME placeholders"
                )

    return StreamRule(
        name=name,
        conditions=tuple(compiled),
        condition_sources=condition_sources,
        ast_conditions=ast_conditions,
        scopes=scopes,
        globs=globs,
        interrupt_mode=interrupt_mode,
        repeat_mode=repeat_mode,
        repeat_gap=repeat_gap,
        body=body,
        source_path=source_path,
    )


def load_stream_rules(
    dir_path: Path,
    *,
    display_dir: str = "",
    allowed_root: Path | None = None,
) -> LoadedStreamRules:
    """Load every ``*.md`` rule in deterministic filename order."""
    directory = Path(dir_path)
    root = Path(allowed_root).resolve(strict=False) if allowed_root else None
    if root is not None:
        try:
            directory.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise StreamRuleError(
                f"{display_dir or directory}: stream-rule directory escapes "
                "the task repository"
            ) from exc
    if not directory.is_dir():
        return LoadedStreamRules((), ())
    rules: list[StreamRule] = []
    files: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.md")):
        prefix = display_dir.strip("/") or directory.name
        source = f"{prefix}/{path.name}"
        if root is not None:
            try:
                path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise StreamRuleError(
                    f"{source}: stream-rule file escapes the task repository"
                ) from exc
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise StreamRuleError(
                f"{source}: stream-rule file must be valid UTF-8: {exc}"
            ) from exc
        except OSError as exc:
            raise StreamRuleError(f"{source}: cannot read stream-rule file: {exc}") from exc
        rule = parse_stream_rule(
            text,
            source_path=source,
            default_name=path.stem,
        )
        if rule.name in seen:
            raise StreamRuleError(
                f"{source}: duplicate stream-rule name {rule.name!r}; "
                f"first declared by {seen[rule.name]}"
            )
        seen[rule.name] = source
        rules.append(rule)
        files.append({
            "path": source,
            "name": rule.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return LoadedStreamRules(tuple(rules), tuple(files))


__all__ = [
    "LoadedStreamRules",
    "StreamRule",
    "StreamRuleError",
    "StreamRuleScope",
    "load_stream_rules",
    "parse_stream_rule",
]
