"""CompoundSelectiveContext — keep compound's evidence classes, trim bulk.

This is a sibling to ``compound``. It preserves the same state source and
section layout, but compresses more selectively than ``focused_compound``:

- keep a bounded tail of unresolved evidence in full
- keep a bounded tail of resolved evidence as stubs
- preserve action diversity inside trace/resolved-evidence budgets
- reserve a small anchor slice for older discovery breadcrumbs
- use a smaller trace budget
- use a smaller rolling tool-result window

The intent is to preserve the discovery breadcrumbs that ``compound`` used
on harder tasks without ferrying every resolved payload verbatim.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ..context import chars_div_4
from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from .compound_context import CompoundContext
from ._metadata import (
    COMPOUND_SELECTIVE_BUDGET_CONFIG_ATTRS,
    COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS,
    COMPOUND_SECTION_LABELS,
    COMPOUND_SECTION_ORDER,
    ContextModeMetadata,
)


class CompoundSelectiveContext(CompoundContext):
    """Compound renderer with selective evidence compression."""

    def __init__(
        self,
        cwd: str,
        original_prompt: str,
        *,
        trace_lines: int,
        evidence_lines: int,
        inference_lines: int,
        recent_tool_results_chars: int,
        trace_stub_chars: int,
        min_turns: int,
        suffix: str,
        selective_trace_lines: int = 0,
        selective_unresolved_evidence_lines: int = 0,
        selective_resolved_evidence_lines: int = 0,
        selective_resolved_evidence_stub_chars: int = 0,
        selective_recent_tool_results_chars: int = 0,
        selective_trace_action_repeat_cap: int = 0,
        selective_resolved_action_repeat_cap: int = 0,
        selective_trace_anchor_lines: int = 0,
        selective_resolved_anchor_lines: int = 0,
        selective_trace_source_anchor_lines: int = 0,
        selective_trace_test_anchor_lines: int = 0,
        selective_resolved_source_anchor_lines: int = 0,
        selective_resolved_test_anchor_lines: int = 0,
        ignore_state: bool = False,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(
            cwd=cwd,
            original_prompt=original_prompt,
            trace_lines=trace_lines,
            evidence_lines=evidence_lines,
            inference_lines=inference_lines,
            recent_tool_results_chars=recent_tool_results_chars,
            trace_stub_chars=trace_stub_chars,
            min_turns=min_turns,
            suffix=suffix,
            ignore_state=ignore_state,
            token_estimator=token_estimator,
        )
        self._selective_trace_lines = selective_trace_lines or trace_lines
        self._selective_unresolved_evidence_lines = (
            selective_unresolved_evidence_lines or evidence_lines
        )
        self._selective_resolved_evidence_lines = selective_resolved_evidence_lines
        self._selective_resolved_evidence_stub_chars = (
            selective_resolved_evidence_stub_chars or trace_stub_chars
        )
        self._selective_recent_tool_results_chars = (
            selective_recent_tool_results_chars or recent_tool_results_chars
        )
        self._selective_trace_action_repeat_cap = selective_trace_action_repeat_cap
        self._selective_resolved_action_repeat_cap = selective_resolved_action_repeat_cap
        self._selective_trace_anchor_lines = selective_trace_anchor_lines
        self._selective_resolved_anchor_lines = selective_resolved_anchor_lines
        self._selective_trace_source_anchor_lines = selective_trace_source_anchor_lines
        self._selective_trace_test_anchor_lines = selective_trace_test_anchor_lines
        self._selective_resolved_source_anchor_lines = selective_resolved_source_anchor_lines
        self._selective_resolved_test_anchor_lines = selective_resolved_test_anchor_lines

    @staticmethod
    def _item_action_key(item) -> str:
        if isinstance(item, dict):
            return str(item.get("action", ""))
        return str(item)

    @classmethod
    def _item_action_text(cls, item) -> str:
        if isinstance(item, dict):
            return str(item.get("action", ""))
        return str(item)

    @staticmethod
    def _extract_quoted_arg(action: str, names: tuple[str, ...]) -> str:
        for name in names:
            marker = f"{name}="
            start = action.find(marker)
            if start < 0:
                continue
            value_start = start + len(marker)
            if value_start >= len(action):
                continue
            quote = action[value_start]
            if quote not in {"'", '"'}:
                continue
            chars: list[str] = []
            escaped = False
            for char in action[value_start + 1:]:
                if escaped:
                    chars.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    chars.append(char)
                    escaped = True
                    continue
                if char == quote:
                    return "".join(chars)
                chars.append(char)
            if chars:
                return "".join(chars)
        return ""

    @classmethod
    def _extract_action_path(cls, item) -> str:
        action = cls._item_action_text(item)
        return cls._extract_quoted_arg(action, ("path", "file_path"))

    @classmethod
    def _extract_action_cmd(cls, item) -> str:
        action = cls._item_action_text(item)
        return cls._extract_quoted_arg(action, ("cmd",))

    @staticmethod
    def _looks_like_test_signal(text: str) -> bool:
        lowered = text.lower()
        return bool(text) and (
            "tests/" in lowered
            or lowered.startswith("tests/")
            or "test_" in lowered
            or "/test_" in lowered
            or lowered.endswith("_test.py")
        )

    @staticmethod
    def _is_concrete_source_path(path: str) -> bool:
        if not path:
            return False
        candidate = path.split("::", 1)[0].rstrip(",")
        if candidate in {".", "..", "/"} or candidate.endswith("/"):
            return False
        return "." in candidate.rsplit("/", 1)[-1]

    @classmethod
    def _anchor_bucket(cls, item) -> str:
        action = cls._item_action_text(item)
        path = cls._extract_action_path(item)
        if path:
            if cls._looks_like_test_signal(path):
                return "test"
            if action.startswith("read(") and cls._is_concrete_source_path(path):
                return "source"
        cmd = cls._extract_action_cmd(item)
        if cmd:
            if _TEST_COMMAND_RE.search(cmd):
                return "test"
            if cls._looks_like_test_signal(cmd):
                return "test"
        if cls._looks_like_test_signal(action):
            return "test"
        return ""

    @staticmethod
    def _select_tail_with_repeat_cap(
        items: list,
        *,
        limit: int,
        repeat_cap: int,
        key_fn,
    ) -> list:
        if limit <= 0:
            return []
        if repeat_cap <= 0:
            return items[-limit:]
        selected: list = []
        seen_counts: dict[str, int] = {}
        for item in reversed(items):
            key = key_fn(item)
            count = seen_counts.get(key, 0)
            if count >= repeat_cap:
                continue
            seen_counts[key] = count + 1
            selected.append(item)
            if len(selected) >= limit:
                break
        selected.reverse()
        return selected

    @classmethod
    def _select_anchored_tail(
        cls,
        items: list,
        *,
        limit: int,
        repeat_cap: int,
        anchor_lines: int,
        source_anchor_lines: int,
        test_anchor_lines: int,
        key_fn,
    ) -> list:
        if limit <= 0:
            return []
        if anchor_lines <= 0 and source_anchor_lines <= 0 and test_anchor_lines <= 0:
            return cls._select_tail_with_repeat_cap(
                items, limit=limit, repeat_cap=repeat_cap, key_fn=key_fn
            )
        selected: list[tuple[int, object]] = []
        seen_counts: dict[str, int] = {}
        anchor_budget = min(anchor_lines, limit)
        cap = repeat_cap if repeat_cap > 0 else None
        chosen_indices: set[int] = set()

        def maybe_add(idx: int, item: object) -> bool:
            if idx in chosen_indices:
                return False
            key = key_fn(item)
            count = seen_counts.get(key, 0)
            if cap is not None and count >= cap:
                return False
            seen_counts[key] = count + 1
            selected.append((idx, item))
            chosen_indices.add(idx)
            return True

        source_budget = min(source_anchor_lines, limit)
        source_added = 0
        if source_budget > 0:
            for idx, item in enumerate(items):
                if source_added >= source_budget:
                    break
                if cls._anchor_bucket(item) != "source":
                    continue
                if maybe_add(idx, item):
                    source_added += 1

        test_budget = min(test_anchor_lines, max(0, limit - len(selected)))
        test_added = 0
        if test_budget > 0:
            for idx, item in enumerate(items):
                if test_added >= test_budget or len(selected) >= limit:
                    break
                if cls._anchor_bucket(item) != "test":
                    continue
                if maybe_add(idx, item):
                    test_added += 1

        for idx, item in enumerate(items):
            if len(selected) >= min(limit, anchor_budget + source_added + test_added):
                break
            maybe_add(idx, item)

        for idx in range(len(items) - 1, -1, -1):
            if idx in chosen_indices:
                continue
            item = items[idx]
            if maybe_add(idx, item) and len(selected) >= limit:
                break

        selected.sort(key=lambda pair: pair[0])
        return [item for _, item in selected]

    def _render_resolved_evidence_item(self, item) -> str:
        if isinstance(item, dict):
            result = str(item.get("result", ""))
            if (
                self._selective_resolved_evidence_stub_chars > 0
                and len(result) > self._selective_resolved_evidence_stub_chars
            ):
                result = (
                    result[: self._selective_resolved_evidence_stub_chars - 3]
                    + "..."
                )
            result = result.replace("\n", " ")
            return f"step {item['step']}: {item['action']} → {result}"
        text = str(item)
        if (
            self._selective_resolved_evidence_stub_chars > 0
            and len(text) > self._selective_resolved_evidence_stub_chars
        ):
            return text[: self._selective_resolved_evidence_stub_chars - 3] + "..."
        return text

    def _split_evidence_selective(self, evidence_list) -> tuple[list[str], list[str]]:
        fails: list[str] = []
        passes: list[str] = []
        for item in evidence_list:
            if self._is_failing(item):
                fails.append(self._render_evidence_item(item))
            else:
                passes.append(item)
        selected_passes = self._select_anchored_tail(
            passes,
            limit=self._selective_resolved_evidence_lines,
            repeat_cap=self._selective_resolved_action_repeat_cap,
            anchor_lines=self._selective_resolved_anchor_lines,
            source_anchor_lines=self._selective_resolved_source_anchor_lines,
            test_anchor_lines=self._selective_resolved_test_anchor_lines,
            key_fn=self._item_action_key,
        )
        return (
            fails[-self._selective_unresolved_evidence_lines:],
            [self._render_resolved_evidence_item(item) for item in selected_passes],
        )

    def _format_selective_tool_results(self) -> str:
        if (
            self._selective_recent_tool_results_chars
            == self._recent_tool_results_chars
        ):
            return self._format_tool_results()
        original = self._recent_tool_results_chars
        self._recent_tool_results_chars = self._selective_recent_tool_results_chars
        try:
            return self._format_tool_results()
        finally:
            self._recent_tool_results_chars = original

    def _build_from_solver(self, solver_dir: Path) -> list[dict]:
        files = self._get_solver_files(solver_dir)

        if self._raw_state_cache is None:
            state_path = solver_dir / "state.json"
            try:
                self._raw_state_cache = json.loads(state_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                self._raw_state_cache = {}
        raw_state = self._raw_state_cache
        raw_trace = raw_state.get("trace", []) if isinstance(raw_state, dict) else []
        raw_evidence = raw_state.get("evidence", []) if isinstance(raw_state, dict) else []

        parts = [f"Task: {self._original_prompt}"]

        if files["state"]:
            parts.append(f"=== State ===\n{files['state']}")

        gate = self._last_failing_entry(raw_evidence)
        if gate:
            parts.append(f"=== Gate (blocking) ===\n{gate}")

        selected_trace = self._select_anchored_tail(
            raw_trace,
            limit=self._selective_trace_lines,
            repeat_cap=self._selective_trace_action_repeat_cap,
            anchor_lines=self._selective_trace_anchor_lines,
            source_anchor_lines=self._selective_trace_source_anchor_lines,
            test_anchor_lines=self._selective_trace_test_anchor_lines,
            key_fn=self._item_action_key,
        )
        trace_rendered = self._format_trace(selected_trace, len(selected_trace))
        if trace_rendered:
            parts.append(f"=== Trace ===\n{trace_rendered}")

        fails, passes = self._split_evidence_selective(raw_evidence)
        if fails or passes:
            ev_lines = []
            if fails:
                ev_lines.append("-- unresolved --")
                ev_lines.extend(fails)
            if passes:
                ev_lines.append("-- resolved --")
                ev_lines.extend(passes)
            parts.append("=== Evidence ===\n" + "\n".join(ev_lines))

        tool_results = self._format_selective_tool_results()
        if tool_results:
            parts.append(tool_results)

        parts.extend(self._injected_fragments)

        suffix = self._render_state_suffix(files)
        if suffix:
            parts.append(suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "compound_selective"
CONTEXT_CLASS = CompoundSelectiveContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=10,
    message_shape="two-message selective compound projection after min_turns_before_context",
    state_source=".solver/state.json",
    source_type="trace_state",
    normal_prompt_sources=(
        ".solver/state.json",
        "in_memory_recent_tool_results",
        "live_workspace_files_from_state_trace_on_session_resume",
    ),
    section_order=COMPOUND_SECTION_ORDER,
    section_labels=COMPOUND_SECTION_LABELS,
    file_freshness="snapshot",
    injection_support="buried_in_projection",
    state_ignored_when_context_ignore_state=True,
    budget_config_attrs=COMPOUND_SELECTIVE_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS,
)
