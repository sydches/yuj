"""YujTranscript — CompactTranscript with yuj protocol structure.

Extends CompactTranscript to map the four commandments onto harness-built
state.  The harness materializes what the commandments describe — the model
doesn't maintain .solver/ files.

Mapping:
  Ratchet (evidence)    → progress log entries with OK/FAIL outcomes
  Ratchet (inference)   → model reasoning extracted from assistant content
  Compromise (trace)    → turn sequence showing descent path
  Friction (gates)      → FAIL entries are unresolved gates
  BareMetal (state)     → current turn, last action, last error

The synthesized user message uses protocol vocabulary so the model
can reason within the yuj framework without file maintenance overhead.
"""
from __future__ import annotations

from collections.abc import Callable

from ._metadata import COMPACT_CONSTRUCTOR_CONFIG_ATTRS, ContextModeMetadata
from .compact_transcript import CompactTranscript, TurnEntry, _classify_outcome, chars_div_4


class YujTranscript(CompactTranscript):
    """CompactTranscript with yuj-structured progress sections.

    Same bounded 2-message format.  Same harness-built entries.
    Different labels — uses protocol vocabulary (evidence, gates, trace)
    so the commandments and the context reinforce each other.

    Evidence here is derived from the content-blind classify_outcome:
    entries where the harness's own exit-code markers indicated FAIL.
    No task-format parsing.
    """

    def __init__(
        self,
        original_prompt: str,
        *,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(
            original_prompt=original_prompt,
            recent_results_chars=recent_results_chars,
            trace_reasoning_chars=trace_reasoning_chars,
            min_turns=min_turns,
            args_summary_chars=args_summary_chars,
            token_estimator=token_estimator,
        )

    def _build_compact(self) -> list[dict]:
        """Build 2-message prompt with yuj-structured sections."""
        parts = [f"Task: {self._original_prompt}"]

        if self._turn_entries:
            # Split entries into evidence (pass/fail verdicts) and trace (action sequence)
            evidence_lines = []
            trace_lines = []
            last_fail = None

            for e in self._turn_entries:
                reason = e.reasoning.replace("\n", " ").strip()
                if len(reason) > self._trace_reasoning_chars:
                    reason = reason[:self._trace_reasoning_chars - 3] + "..."

                # Trace: every action
                if reason:
                    trace_lines.append(
                        f"T{e.turn}: \"{reason}\" → {e.tool_name}({e.args_summary})"
                    )
                else:
                    trace_lines.append(
                        f"T{e.turn}: {e.tool_name}({e.args_summary})"
                    )

                # Evidence: only gate verdicts (outcomes)
                if e.outcome != "OK":
                    evidence_lines.append(f"T{e.turn}: {e.tool_name} → {e.outcome}")
                    last_fail = e

            # Compromise trace — descent path
            parts.append("=== Trace ===\n" + "\n".join(trace_lines))

            # Ratchet evidence — gate verdicts (failures are the signal)
            if evidence_lines:
                parts.append("=== Evidence (unresolved) ===\n" + "\n".join(evidence_lines))

            # Friction — current blocking gate
            if last_fail:
                parts.append(
                    f"=== Gate (blocking) ===\n"
                    f"{last_fail.tool_name}({last_fail.args_summary}) → {last_fail.outcome}"
                )

        # BareMetal state — current position
        state_parts = [f"Turn: {self._turn_count}"]
        if self._turn_entries:
            last = self._turn_entries[-1]
            state_parts.append(f"Last action: {last.tool_name}({last.args_summary}) {last.outcome}")
        parts.append("=== State ===\n" + " | ".join(state_parts))

        # Rolling tool-result window, char-budgeted newest-first.
        # Same logic as the base CompactTranscript — walk from most recent
        # backward, keep entries until the budget is exhausted, drop older
        # ones permanently. Keeps code reads visible across the 2-5 turns
        # it takes to go from "I read the file" to "I edit the file".
        if self._recent_results:
            kept_rev: list[str] = []
            chars_used = 0
            for content in reversed(self._recent_results):
                if chars_used + len(content) > self._recent_results_chars and kept_rev:
                    break
                kept_rev.append(content)
                chars_used += len(content)
            while len(self._recent_results) > len(kept_rev):
                self._recent_results.popleft()
            results = "\n---\n".join(reversed(kept_rev))
            parts.append(
                f"Last {len(kept_rev)} tool results (full, newest last):\n{results}"
            )

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "yuj"
CONTEXT_CLASS = YujTranscript
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=4,
    state_source="append_only_messages",
    source_type="append_log",
    normal_prompt_sources=(
        "in_memory_append_log",
        "in_memory_turn_entries",
        "in_memory_recent_tool_results",
    ),
    file_freshness="snapshot",
    injection_support="buried_in_projection",
    constructor_config_attrs=COMPACT_CONSTRUCTOR_CONFIG_ATTRS,
)
