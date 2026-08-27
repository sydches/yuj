"""Per-session matching and injection state for mid-stream rules."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
from html import escape as xml_escape
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ._stream_rule_loader import (
    StreamRule,
    StreamRuleError,
    _METAVAR_RE,
    _META_PREFIX,
)


_LANGUAGE_BY_SUFFIX = {
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
}
_GRAMMARS = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
}


def _normalize_path(value: str, cwd: Path) -> str:
    value = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve(strict=False).relative_to(
                cwd.resolve(strict=False)
            ).as_posix()
        except ValueError:
            return ""
    normalized = path.as_posix()
    return normalized[2:] if normalized.startswith("./") else normalized


def _path_matches(path: str, pattern: str) -> bool:
    normalized = path.replace("\\", "/")
    candidates = [pattern]
    # Repository globs allow ``**/`` to consume zero path segments, so
    # ``**/*.py`` includes both ``root.py`` and ``src/root.py``.
    if pattern.startswith("**/"):
        candidates.append(pattern[3:])
    return any(
        fnmatchcase(normalized, candidate)
        or fnmatchcase(Path(normalized).name, candidate)
        for candidate in candidates
    )


def _extract_paths(arguments: Mapping[str, object], cwd: Path) -> tuple[str, ...]:
    out: list[str] = []

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key.lower() in {
            "path", "file", "filename", "file_path", "filepath"
        }:
            normalized = _normalize_path(value, cwd)
            if normalized and normalized not in out:
                out.append(normalized)

    visit(arguments)
    return tuple(out)


def _tool_snapshot(
    raw_arguments: str,
    tool_name: str,
    cwd: Path,
) -> tuple[str, tuple[str, ...], bool]:
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return raw_arguments, (), False
    if not isinstance(arguments, Mapping):
        return raw_arguments, (), False
    paths = _extract_paths(arguments, cwd)
    primary = {
        "bash": "cmd",
        "edit": "new_str",
        "notebook_edit": "new_source",
        "structural_edit": "replacement",
        "write": "content",
        "apply_patch": "patch",
    }.get(tool_name)
    value = arguments.get(primary) if primary else None
    if isinstance(value, str):
        return value, paths, tool_name in {"edit", "write"}
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False), paths, False


@lru_cache(maxsize=16)
def _parser_for(language: str):
    try:
        module_name, function_name = _GRAMMARS[language]
        module = importlib.import_module(module_name)
        from tree_sitter import Language, Parser
        grammar = Language(getattr(module, function_name)())
        try:
            parser = Parser(grammar)
        except TypeError:  # tree-sitter < 0.25
            parser = Parser()
            if hasattr(parser, "set_language"):
                parser.set_language(grammar)
            else:
                parser.language = grammar
        return parser
    except (KeyError, ImportError, AttributeError) as exc:
        raise StreamRuleError(
            f"stream-rule structural backend unavailable for {language!r}; "
            "reinstall Yuj with its tree-sitter dependencies"
        ) from exc


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _match_ast_node(pattern_node, candidate_node, pattern: bytes, candidate: bytes,
                    bindings: dict[str, str]) -> bool:
    pattern_text = _node_text(pattern_node, pattern)
    if pattern_node.type == "identifier" and pattern_text.startswith(_META_PREFIX):
        name = pattern_text[len(_META_PREFIX):]
        value = _node_text(candidate_node, candidate)
        prior = bindings.get(name)
        if prior is not None:
            return prior == value
        bindings[name] = value
        return True
    if pattern_node.type != candidate_node.type:
        return False
    pattern_children = list(pattern_node.children)
    candidate_children = list(candidate_node.children)
    if len(pattern_children) != len(candidate_children):
        return False
    if not pattern_children:
        return pattern_text == _node_text(candidate_node, candidate)
    return all(
        _match_ast_node(p_child, c_child, pattern, candidate, bindings)
        for p_child, c_child in zip(pattern_children, candidate_children)
    )


def _walk_nodes(node):
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


@lru_cache(maxsize=256)
def _compiled_ast_pattern(language: str, source_pattern: str):
    parser = _parser_for(language)
    substituted = _METAVAR_RE.sub(
        lambda match: _META_PREFIX + match.group(1), source_pattern
    )
    raw = substituted.encode("utf-8")
    tree = parser.parse(raw)
    root = tree.root_node
    if root.has_error:
        raise StreamRuleError(
            f"invalid astCondition {source_pattern!r} for {language}: parse error"
        )
    node = root.named_children[0] if len(root.named_children) == 1 else root
    return raw, node


def _ast_offset(snapshot: str, path: str, patterns: Sequence[str]) -> int | None:
    language = _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())
    if language is None:
        return None
    raw = snapshot.encode("utf-8")
    tree = _parser_for(language).parse(raw)
    for source_pattern in patterns:
        pattern, pattern_node = _compiled_ast_pattern(language, source_pattern)
        for candidate_node in _walk_nodes(tree.root_node):
            if _match_ast_node(pattern_node, candidate_node, pattern, raw, {}):
                return len(raw[:candidate_node.start_byte].decode("utf-8", errors="ignore"))
    return None


def _record_body(record: Mapping[str, object]) -> str:
    return str(record.get("body") or "").rstrip()


def _record_attrs(record: Mapping[str, object]) -> str:
    attrs = [
        'reason="rule_violation"',
        f'rule="{xml_escape(str(record.get("rule") or ""), quote=True)}"',
    ]
    path = str(record.get("path") or "")
    if path:
        attrs.append(f'path="{xml_escape(path, quote=True)}"')
    return " ".join(attrs)


def format_interrupt_fragment(record: Mapping[str, object]) -> str:
    attrs = _record_attrs(record)
    return (
        f"<injected-fragment {attrs}>\n{_record_body(record)}\n"
        "</injected-fragment>"
    )


def format_tool_reminder(record: Mapping[str, object]) -> str:
    attrs = _record_attrs(record)
    return (
        f"<system-reminder {attrs}>\n{_record_body(record)}\n"
        "</system-reminder>"
    )


class StreamRuleRuntime:
    """Match rules and retain repeat/pending state for one Session."""

    def __init__(self, rules: Iterable[StreamRule], *, repeat_gap: int, cwd: Path):
        self.rules = tuple(rules)
        self.repeat_gap = int(repeat_gap)
        self.cwd = Path(cwd)
        self._last_injected_turn: dict[str, int] = {}
        self._buffers: dict[tuple[str, int], str] = {}
        self._pending_names: set[str] = set()
        self._triggered: list[dict[str, object]] = []
        self._pending_prose: list[dict[str, object]] = []
        self._pending_tool_index: dict[int, list[dict[str, object]]] = {}
        self._pending_tool_id: dict[str, list[dict[str, object]]] = {}

    def begin_attempt(self) -> None:
        self._buffers.clear()
        self._pending_names.clear()
        self._triggered.clear()
        # These queues can contain never-interrupt matches from an earlier
        # chunk of an attempt that a later rule aborts. No tool/result from
        # that discarded attempt exists, so those reminders must not leak
        # into the retry.
        self._pending_prose.clear()
        self._pending_tool_index.clear()

    def _eligible(self, rule: StreamRule, turn: int) -> bool:
        if rule.name in self._pending_names:
            return False
        prior = self._last_injected_turn.get(rule.name)
        if prior is None:
            return True
        if rule.repeat_mode == "once":
            return False
        gap = rule.repeat_gap if rule.repeat_gap is not None else self.repeat_gap
        return turn - prior >= gap

    @staticmethod
    def _interrupts(rule: StreamRule, source: str) -> bool:
        if rule.interrupt_mode == "never":
            return False
        if rule.interrupt_mode == "always":
            return True
        prose = source in {"text", "thinking"}
        return prose if rule.interrupt_mode == "prose-only" else not prose

    @staticmethod
    def _scope(
        rule: StreamRule,
        *,
        source: str,
        tool_name: str,
        paths: Sequence[str],
    ) -> tuple[str, str] | None:
        for scope in rule.scopes:
            if scope.source != source:
                continue
            if source != "tool":
                return scope.label, ""
            if scope.tool_name and scope.tool_name != tool_name:
                continue
            if scope.path_glob:
                path = next(
                    (candidate for candidate in paths
                     if _path_matches(candidate, scope.path_glob)),
                    "",
                )
                if not path:
                    continue
                return scope.label, path
            return scope.label, paths[0] if paths else ""
        return None

    @staticmethod
    def _global_path(rule: StreamRule, paths: Sequence[str]) -> str | None:
        if not rule.globs:
            return paths[0] if paths else ""
        return next(
            (path for path in paths
             if any(_path_matches(path, pattern) for pattern in rule.globs)),
            None,
        )

    @staticmethod
    def _regex_offset(rule: StreamRule, snapshot: str) -> int | None:
        starts = [
            match.start()
            for condition in rule.conditions
            if (match := condition.search(snapshot)) is not None
        ]
        return min(starts) if starts else None

    def observe(self, delta, *, turn: int, force_non_interrupt: bool = False) -> None:
        """Consume one transport delta; raise on an interrupt-worthy batch."""
        source = str(delta.source)
        tool_index = int(getattr(delta, "tool_index", -1))
        tool_name = str(getattr(delta, "tool_name", "") or "")
        if source == "tool":
            raw_arguments = str(getattr(delta, "tool_arguments", "") or "")
            snapshot, paths, structural = _tool_snapshot(
                raw_arguments, tool_name, self.cwd
            )
            self._buffers[(source, tool_index)] = snapshot
        else:
            key = (source, -1)
            snapshot = self._buffers.get(key, "") + str(delta.delta or "")
            self._buffers[key] = snapshot
            paths = ()
            structural = False

        batch: list[dict[str, object]] = []
        for rule in self.rules:
            if not self._eligible(rule, turn):
                continue
            scoped = self._scope(
                rule, source=source, tool_name=tool_name, paths=paths
            )
            if scoped is None:
                continue
            scope_label, scoped_path = scoped
            global_path = self._global_path(rule, paths)
            if global_path is None:
                continue
            path = scoped_path or global_path or ""
            offset = self._regex_offset(rule, snapshot)
            if (
                offset is None
                and structural
                and path
                and rule.ast_conditions
            ):
                offset = _ast_offset(snapshot, path, rule.ast_conditions)
            if offset is None:
                continue
            interrupt = (
                False if force_non_interrupt else self._interrupts(rule, source)
            )
            record: dict[str, object] = {
                "rule": rule.name,
                "scope": scope_label,
                "offset": int(offset),
                "path": path,
                "tool_name": tool_name,
                "tool_index": tool_index,
                "interrupt": interrupt,
                "body": rule.body,
            }
            batch.append(record)
            self._pending_names.add(rule.name)

        if not batch:
            return
        self._triggered.extend(batch)
        if any(bool(record["interrupt"]) for record in batch):
            from ..server._streaming import StreamRuleInterrupt
            raise StreamRuleInterrupt(tuple(batch))
        for record in batch:
            if source == "tool":
                self._pending_tool_index.setdefault(tool_index, []).append(record)
            else:
                self._pending_prose.append(record)

    def accept_response(
        self,
        result,
        *,
        turn: int,
        streamed: bool,
        replay: bool,
    ) -> tuple[dict, ...]:
        """Finalize non-interrupt matches and bind tool indices to call IDs."""
        if not streamed and not replay:
            if result.content:
                self.observe(
                    _SnapshotDelta("text", str(result.content)),
                    turn=turn,
                    force_non_interrupt=True,
                )
            for index, tool_call in enumerate(result.tool_calls):
                self.observe(
                    _SnapshotDelta(
                        "tool",
                        "",
                        tool_index=index,
                        tool_name=tool_call.name,
                        tool_arguments=json.dumps(
                            tool_call.arguments, ensure_ascii=False
                        ),
                    ),
                    turn=turn,
                    force_non_interrupt=True,
                )
        for index, records in tuple(self._pending_tool_index.items()):
            if 0 <= index < len(result.tool_calls):
                self._pending_tool_id.setdefault(
                    result.tool_calls[index].id, []
                ).extend(records)
            del self._pending_tool_index[index]
        triggered = tuple(self._triggered)
        self._triggered.clear()
        return triggered

    def mark_injected(self, records: Iterable[Mapping[str, object]], *, turn: int) -> None:
        names = {str(record.get("rule") or "") for record in records}
        for name in names:
            if name:
                self._last_injected_turn[name] = int(turn)
                self._pending_names.discard(name)
        self._pending_prose = [
            record for record in self._pending_prose
            if str(record.get("rule") or "") not in names
        ]
        for key in tuple(self._pending_tool_id):
            kept = [
                record for record in self._pending_tool_id[key]
                if str(record.get("rule") or "") not in names
            ]
            if kept:
                self._pending_tool_id[key] = kept
            else:
                del self._pending_tool_id[key]

    def take_prose_injections(self, *, turn: int) -> tuple[dict, ...]:
        records = tuple(self._pending_prose)
        self._pending_prose.clear()
        self.mark_injected(records, turn=turn)
        return records

    def decorate_tool_result(
        self,
        tool_call_id: str,
        result: str,
        *,
        turn: int,
    ) -> tuple[str, tuple[dict, ...]]:
        records = tuple(self._pending_tool_id.pop(tool_call_id, ()))
        if not records:
            return result, ()
        self.mark_injected(records, turn=turn)
        reminder = "\n\n".join(format_tool_reminder(record) for record in records)
        return f"{reminder}\n\n{result}", records


@dataclass(frozen=True, slots=True)
class _SnapshotDelta:
    source: str
    delta: str
    tool_index: int = -1
    tool_name: str = ""
    tool_arguments: str = ""


__all__ = [
    "StreamRuleRuntime",
    "format_interrupt_fragment",
    "format_tool_reminder",
]
