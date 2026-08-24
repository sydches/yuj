"""FocusedCompoundContext — compound semantics with a smaller renderer surface.

Keeps the same source of truth as CompoundContext (`.solver/state.json`) and
the same cross-session semantics, but narrows what is rendered:

- smaller trace budget
- fail-focused evidence by default
- smaller rolling tool-result window

This is a sibling to `compound`, not a mutation of it. The baseline stays
stable; this class is the experimental surface for "compound, but less".
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ..context import chars_div_4
from .compound_context import CompoundContext
from ._metadata import (
    FOCUSED_COMPOUND_BUDGET_CONFIG_ATTRS,
    FOCUSED_COMPOUND_CONSTRUCTOR_CONFIG_ATTRS,
    COMPOUND_SECTION_LABELS,
    COMPOUND_SECTION_ORDER,
    ContextModeMetadata,
)


class FocusedCompoundContext(CompoundContext):
    """Compound renderer with config-driven compression knobs."""

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
        focused_trace_lines: int = 0,
        focused_evidence_lines: int = 0,
        focused_recent_tool_results_chars: int = 0,
        focused_include_resolved_evidence: bool = False,
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
        self._focused_trace_lines = focused_trace_lines or trace_lines
        self._focused_evidence_lines = focused_evidence_lines or evidence_lines
        self._focused_recent_tool_results_chars = (
            focused_recent_tool_results_chars or recent_tool_results_chars
        )
        self._focused_include_resolved_evidence = focused_include_resolved_evidence

    def _split_evidence_for_focus(self, evidence_list) -> tuple[list[str], list[str]]:
        fails: list[str] = []
        passes: list[str] = []
        for item in evidence_list:
            text = self._render_evidence_item(item)
            if self._is_failing(item):
                fails.append(text)
            elif self._focused_include_resolved_evidence:
                passes.append(text)
        return (
            fails[-self._focused_evidence_lines:],
            passes[-self._focused_evidence_lines:],
        )

    def _format_focused_tool_results(self) -> str:
        if self._focused_recent_tool_results_chars == self._recent_tool_results_chars:
            return self._format_tool_results()
        original = self._recent_tool_results_chars
        self._recent_tool_results_chars = self._focused_recent_tool_results_chars
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

        trace_rendered = self._format_trace(raw_trace, self._focused_trace_lines)
        if trace_rendered:
            parts.append(f"=== Trace ===\n{trace_rendered}")

        fails, passes = self._split_evidence_for_focus(raw_evidence)
        if fails or passes:
            ev_lines = []
            if fails:
                ev_lines.append("-- unresolved --")
                ev_lines.extend(fails)
            if passes:
                ev_lines.append("-- resolved --")
                ev_lines.extend(passes)
            parts.append("=== Evidence ===\n" + "\n".join(ev_lines))

        tool_results = self._format_focused_tool_results()
        if tool_results:
            parts.append(tool_results)

        suffix = self._render_state_suffix(files)
        if suffix:
            parts.append(suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "focused_compound"
CONTEXT_CLASS = FocusedCompoundContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=9,
    message_shape="two-message focused compound projection after min_turns_before_context",
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
    budget_config_attrs=FOCUSED_COMPOUND_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=FOCUSED_COMPOUND_CONSTRUCTOR_CONFIG_ATTRS,
)
