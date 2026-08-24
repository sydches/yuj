"""CompoundContext — narrative memory + persistent state, merged.

Combines the reasoning-preservation of YujTranscript with the cross-session
persistence of SolverStateContext. Reads state from `.solver/state.json`
(harness-written, cross-session) AND renders each trace entry with the
model's own pre-tool reasoning when available.

Design principles:

  - Source of truth: `.solver/state.json`, projected from `.trace.jsonl` by
    state_writer. Survives session boundaries. Model never maintains it.
  - Narrative trace: each entry includes reasoning (model's pre-tool
    sentence) in addition to action/result/next. Reasoning is deduplicated
    by (session, turn) — multi-tool turns render it once.
  - Content-blind: the harness never inspects tool results for task-format
    patterns. The evidence `verdict` field is OK/FAIL, derived by
    `classify_outcome` from harness-generated exit-code markers only.
    Fail-first ordering and the gate pointer both key off that field, not
    off content parsing.
  - Evidence ordering: FAIL entries first (the Friction signal), then pass
    entries if budget remains. Gate pointer = most recent FAIL, rendered at
    the top of the user message so the model sees what's blocking before
    scanning the full trace.
  - Rolling tool-result window: identical to parent. Raw tool output only —
    reasoning text never enters the rolling window.

Supersedes YujTranscript (narrative-only) and the base SolverStateContext
(stub-only trace) by taking the better half of each.
"""
from __future__ import annotations

import json
from pathlib import Path

from .solver_state_context import SolverStateContext
from ._metadata import (
    STATEFUL_BUDGET_CONFIG_ATTRS,
    STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
    COMPOUND_SECTION_LABELS,
    COMPOUND_SECTION_ORDER,
    ContextModeMetadata,
)


class CompoundContext(SolverStateContext):
    """Narrative + persistent context manager.

    Inherits file-reading, cache invalidation, rolling tool-result
    window, and 2-message prompt shape from SolverStateContext. Only
    the trace rendering and user-message assembly are overridden.
    """

    def _format_trace(self, trace, max_entries: int) -> str:
        """Render trace with reasoning-per-turn and stubbed results.

        Format per entry:
            Tn [reasoning ...]
               → action → stub_result

        Reasoning is printed ONLY when it differs from the previous
        entry's reasoning (same-turn multi-tool calls share reasoning;
        rendering it once per turn is enough). Result is stubbed to
        trace_stub_chars — the full payload lives in the rolling
        tool-result window and must not be duplicated here.
        """
        if not isinstance(trace, list) or not trace:
            return ""
        tail = trace[-max_entries:]
        lines: list[str] = []
        prev_reasoning = None
        for entry in tail:
            if not isinstance(entry, dict):
                lines.append(f"? | {entry}")
                continue
            step = entry.get("step", "?")
            turn = entry.get("turn")
            reasoning = (entry.get("reasoning") or "").strip()
            action = entry.get("action", "")
            result = entry.get("result", "")
            nxt = entry.get("next", "")

            stub_result = (
                result[: self._trace_stub_chars - 3] + "..."
                if len(result) > self._trace_stub_chars
                else result
            )
            stub_result = stub_result.replace("\n", " ")

            header = f"T{turn}" if turn is not None else f"step{step}"
            if reasoning and reasoning != prev_reasoning:
                short = reasoning.replace("\n", " ").strip()
                if len(short) > 300:
                    short = short[:297] + "..."
                lines.append(f"{header} [{short}]")
                prev_reasoning = reasoning
            lines.append(f"    → {action} → {stub_result}{' → ' + nxt if nxt else ''}")
        return "\n".join(lines)

    @staticmethod
    def _render_evidence_item(item) -> str:
        """Render an evidence entry as a human-readable line."""
        if isinstance(item, dict):
            return f"step {item['step']}: {item['action']} → {item.get('result', '')}"
        return str(item)

    @staticmethod
    def _is_failing(item) -> bool:
        """True if this evidence entry represents a failure.

        Reads only the structured `verdict` field, which was computed
        content-blindly by classify_outcome at projection time. No
        string scanning of the result text — the harness never reads
        task output for verdict clues.
        """
        if isinstance(item, dict):
            return item.get("verdict", "").startswith("FAIL")
        return False

    def _split_evidence_fail_first(self, evidence_list) -> tuple[list[str], list[str]]:
        """Split evidence into (fails, passes) preserving tail ordering."""
        fails: list[str] = []
        passes: list[str] = []
        for item in evidence_list[-self._evidence_lines:]:
            text = self._render_evidence_item(item)
            if self._is_failing(item):
                fails.append(text)
            else:
                passes.append(text)
        return fails, passes

    def _last_failing_entry(self, evidence_list) -> str | None:
        """Return the most recent failing evidence entry, or None."""
        for item in reversed(evidence_list):
            if self._is_failing(item):
                return self._render_evidence_item(item)
        return None

    def _build_from_solver(self, solver_dir: Path) -> list[dict]:
        """Build a 2-message prompt with yuj-vocab sections + Gate pointer.

        Section order reflects the commandments:
          1. Task (original problem statement)
          2. State (BareMetal — current position)
          3. Gate (Friction — what's blocking right now)
          4. Trace (Compromise — descent path with reasoning)
          5. Evidence (Ratchet — fail-first, then pass)
          6. Tool results (rolling window, raw outputs, no reasoning)
          7. Continuation suffix (from cfg.state_context_suffix)
        """
        files = self._get_solver_files(solver_dir)

        # Raw state.json needed for the reasoning-aware trace renderer and
        # the fail-first evidence split. Cached on the parent's
        # _raw_state_cache so consecutive get_messages() calls within a
        # turn hit the parse once. Invalidated in add_tool_result.
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

        # Gate (blocking) — most recent failing evidence entry. Shown early
        # so the model sees what's blocking before scanning the full trace.
        gate = self._last_failing_entry(raw_evidence)
        if gate:
            parts.append(f"=== Gate (blocking) ===\n{gate}")

        # Trace — reasoning-aware renderer.
        trace_rendered = self._format_trace(raw_trace, self._trace_lines)
        if trace_rendered:
            parts.append(f"=== Trace ===\n{trace_rendered}")

        # Evidence — fail-first, then pass. Failures show what's blocking,
        # passes show what's been fixed. Verdict is content-blind.
        fails, passes = self._split_evidence_fail_first(raw_evidence)
        if fails or passes:
            ev_lines = []
            if fails:
                ev_lines.append("-- unresolved --")
                ev_lines.extend(fails)
            if passes:
                ev_lines.append("-- resolved --")
                ev_lines.extend(passes)
            parts.append("=== Evidence ===\n" + "\n".join(ev_lines))

        # Rolling tool-result window — raw tool outputs only. Never
        # includes reasoning text: that belongs in the Trace section
        # above. Keeping these separate avoids the duplication bug
        # noted in solver_state_context.py:186.
        tool_results = self._format_tool_results()
        if tool_results:
            parts.append(tool_results)

        suffix = self._render_state_suffix(files)
        if suffix:
            parts.append(suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "compound"
CONTEXT_CLASS = CompoundContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=8,
    message_shape="two-message compound projection after min_turns_before_context",
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
    budget_config_attrs=STATEFUL_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
)
