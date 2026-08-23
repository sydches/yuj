"""Compound-aware bash-to-dedicated-tool redirect matching.

Rules remain TOML-owned.  This module loads ``[[redirect]]`` entries, splits
top-level shell compounds without breaking quoted operators, gates each rule
on the active tool set, and renders a unified error result.  It never rewrites
or executes a command.
"""
from __future__ import annotations

import html
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Pattern


log = logging.getLogger(__name__)
_READ_SIDE_TOOLS = frozenset({"read", "grep", "glob"})
_OPERATORS = ("&&", "||", "|&", ";", "|", "&", "\n")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)


@dataclass(frozen=True)
class RedirectRule:
    name: str
    pattern: Pattern[str]
    tool: str
    message: str
    fragment_aware: bool = True

    @property
    def read_side(self) -> bool:
        return self.tool in _READ_SIDE_TOOLS


@dataclass(frozen=True)
class ShellFragment:
    text: str
    index: int
    operator_before: str = ""
    operator_after: str = ""

    @property
    def stdin_from_pipe(self) -> bool:
        return self.operator_before in {"|", "|&"}


@dataclass(frozen=True)
class RedirectDecision:
    rule_name: str
    tool: str
    message: str
    fragment: str
    fragment_index: int | None

    def trace_fields(self) -> dict[str, object]:
        return {
            "rule": self.rule_name,
            "tool": self.tool,
            "fragment_index": self.fragment_index,
        }


def _compile_flags(value: object) -> int:
    flags = 0
    raw = "i" if value is None else str(value)
    for char in raw:
        try:
            flags |= {
                "i": re.IGNORECASE,
                "m": re.MULTILINE,
                "s": re.DOTALL,
                "x": re.VERBOSE,
            }[char.lower()]
        except KeyError:
            raise ValueError(f"unsupported regex flag {char!r}") from None
    return flags


def parse_redirect_rules(entries: Iterable[Mapping[str, object]]) -> list[RedirectRule]:
    """Compile redirect mappings, skipping malformed entries with a warning."""
    rules: list[RedirectRule] = []
    for ordinal, entry in enumerate(entries, start=1):
        try:
            name = str(entry.get("name") or f"redirect_{ordinal}")
            tool = str(entry["tool"]).strip()
            message = str(entry["message"]).strip()
            pattern_text = str(entry["pattern"])
            if not tool or not message:
                raise ValueError("tool and message must be non-empty")
            rules.append(RedirectRule(
                name=name,
                pattern=re.compile(pattern_text, _compile_flags(entry.get("flags"))),
                tool=tool,
                message=message,
                fragment_aware=bool(entry.get("fragment_aware", True)),
            ))
        except (KeyError, TypeError, ValueError, re.error) as exc:
            log.warning("redirect rule %d skipped: %s", ordinal, exc)
    return rules


def load_redirect_rules(path: Path | None = None) -> list[RedirectRule]:
    """Load ``[[redirect]]`` from the canonical bash forbidden-rule file."""
    from .._shared.toml_compat import tomllib

    if path is None:
        path = Path(__file__).parents[1] / "bash_quirks" / "forbidden.toml"
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    return parse_redirect_rules(data.get("redirect", []))


def split_shell_fragments(command: str) -> tuple[ShellFragment, ...]:
    """Split on unquoted top-level ``&& || ; | |& &`` and newlines.

    Parenthesized command substitutions/subshells and braced groups remain in
    their surrounding fragment.  Empty fragments are omitted while operators
    remain attached as before/after metadata on their neighbors.
    """
    pieces: list[tuple[str, str]] = []
    quote = ""
    escaped = False
    paren_depth = 0
    brace_depth = 0
    start = 0
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == "(":
            paren_depth += 1
            index += 1
            continue
        if char == ")" and paren_depth:
            paren_depth -= 1
            index += 1
            continue
        if char == "{":
            brace_depth += 1
            index += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            index += 1
            continue
        if paren_depth or brace_depth:
            index += 1
            continue

        operator = ""
        for candidate in _OPERATORS:
            if command.startswith(candidate, index):
                operator = candidate
                break
        if not operator:
            index += 1
            continue
        pieces.append((command[start:index], operator))
        index += len(operator)
        start = index
    pieces.append((command[start:], ""))

    fragments: list[ShellFragment] = []
    operator_before = ""
    for text, operator_after in pieces:
        stripped = text.strip()
        if stripped:
            fragments.append(ShellFragment(
                text=stripped,
                index=len(fragments),
                operator_before=operator_before,
                operator_after=operator_after,
            ))
            operator_before = operator_after
        elif operator_after:
            operator_before = operator_after
    return tuple(fragments)


def strip_leading_assignments(fragment: str) -> str:
    """Remove top-level ``NAME=value`` words while preserving command text."""
    position = 0
    length = len(fragment)
    while True:
        while position < length and fragment[position].isspace():
            position += 1
        word_start = position
        quote = ""
        escaped = False
        while position < length:
            char = fragment[position]
            if escaped:
                escaped = False
            elif char == "\\" and quote != "'":
                escaped = True
            elif quote:
                if char == quote:
                    quote = ""
            elif char in "'\"":
                quote = char
            elif char.isspace():
                break
            elif char in ";&|<>\n":
                return fragment[word_start:].lstrip()
            position += 1
        raw_word = fragment[word_start:position]
        if not raw_word:
            return ""
        try:
            parsed = shlex.split(raw_word, posix=True)
        except ValueError:
            return fragment[word_start:].lstrip()
        if len(parsed) != 1 or not _ASSIGNMENT_RE.fullmatch(parsed[0]):
            return fragment[word_start:].lstrip()


def _argv(fragment: str) -> list[str]:
    try:
        return shlex.split(strip_leading_assignments(fragment), posix=True)
    except ValueError:
        return []


def _is_aggregate(fragment: str) -> bool:
    argv = _argv(fragment)
    if not argv:
        return False
    verb = argv[0].rsplit("/", 1)[-1]
    args = argv[1:]
    if verb == "wc":
        return True
    if verb not in {"grep", "egrep", "fgrep", "rg", "ag", "ack"}:
        return False
    long_flags = {
        "--count", "--count-matches", "--files-with-matches",
        "--files-without-match", "--quiet", "--stats", "--json",
    }
    if any(arg in long_flags for arg in args):
        return True
    for arg in args:
        if re.fullmatch(r"-[A-Za-z]+", arg) and any(
            char in arg[1:] for char in "clLq"
        ):
            return True
    return False


def _aggregate_pipeline_indices(
    fragments: tuple[ShellFragment, ...],
) -> frozenset[int]:
    """Return all stages in pipelines whose consumer is aggregate-only."""
    exempt: set[int] = set()
    group: list[ShellFragment] = []
    for fragment in fragments:
        if not group:
            group = [fragment]
        elif fragment.operator_before in {"|", "|&"}:
            group.append(fragment)
        else:
            if len(group) > 1 and any(_is_aggregate(item.text) for item in group):
                exempt.update(item.index for item in group)
            group = [fragment]
        if fragment.operator_after not in {"|", "|&"}:
            if len(group) > 1 and any(_is_aggregate(item.text) for item in group):
                exempt.update(item.index for item in group)
            group = []
    if len(group) > 1 and any(_is_aggregate(item.text) for item in group):
        exempt.update(item.index for item in group)
    return frozenset(exempt)


def find_redirect(
    command: str,
    rules: Iterable[RedirectRule],
    *,
    active_tools: Iterable[str],
    read_side_enabled: bool,
) -> RedirectDecision | None:
    """Return the first applicable full-command or fragment rule match."""
    active = frozenset(active_tools)
    usable = tuple(
        rule for rule in rules
        if rule.tool in active and (read_side_enabled or not rule.read_side)
    )
    if not usable:
        return None
    fragments = split_shell_fragments(command)
    aggregate_exempt = _aggregate_pipeline_indices(fragments)
    any_aggregate = any(
        fragment.index in aggregate_exempt or _is_aggregate(fragment.text)
        for fragment in fragments
    )

    normalized_full = strip_leading_assignments(command)
    for rule in usable:
        if rule.read_side and any_aggregate:
            continue
        if rule.pattern.search(normalized_full):
            return RedirectDecision(
                rule_name=rule.name,
                tool=rule.tool,
                message=rule.message,
                fragment=normalized_full,
                fragment_index=None,
            )

    for fragment in fragments:
        normalized = strip_leading_assignments(fragment.text)
        for rule in usable:
            if not rule.fragment_aware:
                continue
            if rule.read_side and (
                fragment.stdin_from_pipe
                or fragment.index in aggregate_exempt
                or _is_aggregate(fragment.text)
            ):
                continue
            if rule.pattern.search(normalized):
                return RedirectDecision(
                    rule_name=rule.name,
                    tool=rule.tool,
                    message=rule.message,
                    fragment=normalized,
                    fragment_index=fragment.index,
                )
    return None


def render_redirect_error(
    decision: RedirectDecision,
    *,
    max_chars: int | None = None,
) -> str:
    """Render a capped, valid error envelope counted by the error ladder."""
    tool = html.escape(decision.tool, quote=True)
    opening = (
        f'<tool_result tool_name="bash" status="error" '
        f'error_kind="redirect_rule" redirect_tool="{tool}" v="1">\n'
    )
    closing = "\n</tool_result>"
    body_text = f"Blocked: {decision.message}"
    body = html.escape(body_text, quote=False)
    result = opening + body + closing
    if max_chars is None or len(result) <= max_chars:
        return result

    marker = "... [redirect message truncated]"
    budget = max_chars - len(opening) - len(closing)
    if budget <= 0:
        return result[:max(0, max_chars)]
    if budget <= len(marker):
        body = marker[:budget]
    else:
        prefix_budget = budget - len(marker)
        low, high = 0, len(body_text)
        while low < high:
            middle = (low + high + 1) // 2
            escaped = html.escape(body_text[:middle], quote=False)
            if len(escaped) <= prefix_budget:
                low = middle
            else:
                high = middle - 1
        body = html.escape(body_text[:low], quote=False) + marker
    return opening + body + closing


__all__ = [
    "RedirectDecision", "RedirectRule", "ShellFragment", "find_redirect",
    "load_redirect_rules", "parse_redirect_rules", "render_redirect_error",
    "split_shell_fragments", "strip_leading_assignments",
]
