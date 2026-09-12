"""Per-session matching and injection state for mid-stream rules."""
from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from html import escape as xml_escape
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ._stream_rule_ast import _ast_offset
from ._stream_rule_loader import (
    StreamRule,
)


def _normalize_path(value: str, cwd: Path) -> str:
    value = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute():
        try:
            from .task_path import bound_task_path
            target = bound_task_path(cwd, value)
            if target is not None:
                return target.relative_to(target.files.root).as_posix()
            return path.resolve(strict=False).relative_to(
                cwd.resolve(strict=False)
            ).as_posix()
        except (OSError, ValueError):
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
        # Every rule below sees this same snapshot. Reuse only its source parse;
        # a later delta, including incomplete syntax, must get a fresh tree.
        parsed_by_language: dict = {}
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
                offset = _ast_offset(snapshot, path, rule.ast_conditions,
                                     parsed_by_language=parsed_by_language)
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
