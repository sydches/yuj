"""CompactTranscript — bounded context via harness-built summary.

Each turn, records the model's reasoning (assistant content) and tool
outcome (name, args, success/error).  get_messages() builds a 2-message
prompt: system + synthesized user containing the original task, a
compressed progress log, and a char-budgeted window of recent full
tool results.

No model cooperation required.  No disk I/O.  No protocol dependency.
The class keeps recent results in full and summarizes older turns.
"""
from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..context import ContextManager, chars_div_4
from ._metadata import COMPACT_CONSTRUCTOR_CONFIG_ATTRS, ContextModeMetadata


# File-targeting tools whose results are eligible for read-dedup. Mutation
# tools (write/edit/str_replace/create) are also tracked here because a
# mutation on path P invalidates any earlier read of P (the file's bytes
# may have changed) — without that check we'd elide a stale read whose
# content actually still matters.
_FILE_TARGETING_TOOLS = frozenset({"read", "write", "edit", "str_replace", "create"})
_MUTATION_TOOLS = frozenset({"write", "edit", "str_replace", "create"})

# Argument-shape patterns for path extraction. Args are usually a JSON
# blob string ('{"path": "src/foo.py", "limit": 80}') but the older
# schema sometimes uses key=val syntax. Accept both.
_PATH_KEY_RE = re.compile(r'(?:^|[\s,{"])(?:path|file_path|target)\s*[:=]\s*["\']?([^"\',}\s]+)')


def _compact_byte_count(n: int) -> str:
    """Render a byte count as a short human string: 847b / 2.3k / 47k.

    Used by the sized-stub band in the progress log. Rounds to one
    decimal in the 1k-10k range for readability ("2.3k" beats "2300b"
    for a one-glance magnitude check), then drops the decimal for
    larger values where the next significant digit is more useful.
    """
    if n < 1000:
        return f"{n}b"
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    return f"{n // 1000}k"


def _extract_path_from_args(tool_name: str, args_summary: str) -> str | None:
    """Return the file path the tool targeted, or None when not applicable.

    Used by the file-read dedup pre-pass in CompactTranscript._build_compact.
    Two strategies:

    1. JSON parse — args_summary is typically the raw OpenAI-tool args
       JSON ('{"path": "...", ...}'). cheap, succeeds in the common case.
    2. Regex fallback — when the args were already truncated for trace
       (mid-string ellipsis) or when the tool used the legacy key=val
       shape, the JSON parse fails. The regex catches `path=`, `path:`,
       `file_path=`, and `target=` variants.

    Returns None when:
      - tool_name isn't in _FILE_TARGETING_TOOLS
      - neither extractor finds a path
      - the extracted path is empty / clearly malformed
    """
    if tool_name not in _FILE_TARGETING_TOOLS:
        return None
    s = args_summary or ""
    if not s:
        return None
    try:
        d = json.loads(s)
        for k in ("path", "file_path", "target"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    m = _PATH_KEY_RE.search(s)
    if m:
        p = m.group(1).strip()
        return p or None
    return None


@dataclass
class TurnEntry:
    """One turn's compressed record."""
    turn: int
    reasoning: str  # assistant content (verbatim, typically 1-2 sentences)
    tool_name: str
    args_summary: str
    outcome: str  # "OK" or "FAIL" — content-blind, from exit-code markers only
    # Output-size annotation for the Hermes-style sized stub band. Lets the
    # progress log show the MAGNITUDE of a result that fell out of the
    # char-budgeted recent-results window — not just the verb. Chars and
    # lines computed at add_tool_result time from the raw result text;
    # zero values mean "not measured" (e.g. earlier projection with no
    # size info).
    result_chars: int = 0
    result_lines: int = 0


# Canonical outcome classifier — single source of truth in _shared.classification.
# Re-exported here as _classify_outcome so yuj_transcript.py's import is unchanged.
from ..._shared.classification import classify_outcome as _classify_outcome


class CompactTranscript(ContextManager):
    """Bounded context built from turn-level summaries.

    After min_turns, every prompt is exactly 2 messages:
      system: static prompt (server-cached)
      user: task + progress log + last N full tool results

    The progress log preserves:
      - Model's reasoning per turn (semantic signal)
      - Tool call + outcome per turn (structural signal, content-blind)
    Full tool result payloads are kept only for the last N turns.

    All numeric tunables are required kwargs — no module-level shadow
    defaults. The harness wires them from config.toml through Config.
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
        super().__init__(token_estimator)
        self._original_prompt = original_prompt
        self._trace_reasoning_chars = trace_reasoning_chars
        self._min_turns = min_turns
        self._args_summary_chars = args_summary_chars
        self._system_content: str = ""
        self._turn_entries: list[TurnEntry] = []
        # Unbounded deque — trimmed by char budget in _build_compact, not
        # by entry count. The `recent_results` kwarg is kept for backward
        # compatibility but unused (previously `deque(maxlen=3)`).
        self._recent_results: deque[str] = deque()
        # Per-result metadata aligned 1:1 with _recent_results. Each entry
        # is a dict with at least `tool_name` and (when extractable from
        # args_summary) `path`, used by the file-read dedup pre-pass in
        # _build_compact. When the assistant message is unavailable at
        # add_tool_result time, we still append a placeholder dict so
        # the alignment invariant (len(_recent_results) ==
        # len(_recent_results_meta)) holds.
        self._recent_results_meta: deque[dict] = deque()
        self._recent_results_chars: int = recent_results_chars
        self._all_messages: list[dict] = []  # raw log (fallback for turns 0-1)
        self._turn_count: int = 0
        self._last_assistant_msg: dict | None = None  # buffer between add_assistant and first add_tool_result
        self._prev_assistant_msg: dict | None = None  # retained for multi-tool lookups within same turn
        # Per-turn message + token caches. get_messages rebuilds a compact
        # representation that is identical within a turn; estimate_tokens
        # scans that representation. Caching eliminates the second build
        # when estimate_tokens is called after get_messages within the
        # same turn, and avoids a fresh compact-build if both are called
        # multiple times without a mutation in between. Invalidated by
        # every add_* method.
        self._msg_cache: list[dict] | None = None
        self._tok_cache: int | None = None

    def add_system(self, content: str) -> None:
        self._system_content = content
        self._all_messages.append({"role": "system", "content": content})
        self._msg_cache = None
        self._tok_cache = None

    def add_user(self, content: str) -> None:
        self._all_messages.append({"role": "user", "content": content})
        self._msg_cache = None
        self._tok_cache = None

    def add_assistant(self, message: dict) -> None:
        self._all_messages.append(message)
        self._last_assistant_msg = message
        self._turn_count += 1
        self._msg_cache = None
        self._tok_cache = None

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        self._all_messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "content": content,
        })
        self._recent_results.append(content)
        self._msg_cache = None
        self._tok_cache = None

        # Build turn entry from the assistant message + this result.
        # First tool result for a turn gets the reasoning; subsequent ones
        # (multi-tool calls) get empty reasoning but still record the outcome.
        assistant_msg = self._last_assistant_msg or self._prev_assistant_msg
        if assistant_msg is not None:
            reasoning = ""
            if self._last_assistant_msg is not None:
                reasoning = self._last_assistant_msg.get("content") or ""
                self._prev_assistant_msg = self._last_assistant_msg
                self._last_assistant_msg = None
            extracted_name, args_summary = self._extract_tool_info(
                assistant_msg, tool_call_id,
            )
            self._turn_entries.append(TurnEntry(
                turn=self._turn_count,
                reasoning=reasoning,
                tool_name=extracted_name,
                args_summary=args_summary,
                outcome=_classify_outcome(content),
                # Cheap measurements taken once at write time — the
                # progress log can then surface size in O(1) per turn.
                result_chars=len(content),
                result_lines=content.count("\n") + (0 if content.endswith("\n") else 1) if content else 0,
            ))
            # Record metadata used by the dedup pre-pass. Path extraction
            # only fires for read/write/edit/str_replace/create — the
            # tools that target a single file. For other tools (bash,
            # glob, grep) path is None and the entry is treated as
            # never-deduplicable.
            self._recent_results_meta.append({
                "tool_name": extracted_name,
                "path": _extract_path_from_args(extracted_name, args_summary),
            })
        else:
            # Assistant message unavailable — keep alignment with a
            # blank metadata entry so _recent_results_meta and
            # _recent_results stay 1:1 indexable.
            self._recent_results_meta.append({"tool_name": tool_name, "path": None})

    def get_messages(self) -> list[dict]:
        if self._msg_cache is not None:
            return self._msg_cache
        if self._turn_count < self._min_turns:
            # Fallback path returns the raw-log reference; cache holds the
            # same reference so a subsequent estimate_tokens sees identical
            # data. Mutation goes through add_* which invalidates.
            self._msg_cache = self._all_messages
        else:
            self._msg_cache = self._build_compact()
            # Token accounting: the projection replaces the full append log
            # with a compact summary. Record the exact delta vs. what a
            # FullTranscript would have emitted for the same turn state.
            from ..savings import get_ledger
            full_chars = sum(len(str(m)) for m in self._all_messages)
            actual_chars = sum(len(str(m)) for m in self._msg_cache)
            get_ledger().record(
                bucket="context_projection",
                layer="context_strategy",
                mechanism="compact_transcript",
                input_chars=full_chars,
                output_chars=actual_chars,
                measure_type="exact",
                ctx={"turn_count": self._turn_count,
                     "messages": len(self._msg_cache)},
            )
        return self._msg_cache

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self.get_messages())
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._all_messages)

    def replace_all_messages(self, new_messages: list[dict]) -> bool:
        """Persist the harness's compacted message list as the new append log.

        Called by Session._maybe_compact_messages once a digest has been
        rendered + stitched with the latest assistant/tool pair. Without
        this, the very next get_messages() would re-render from the
        un-compacted log and the prompt would snap back past the gate.
        """
        self._all_messages = list(new_messages)
        # The parallel result/meta deques are projection state derived
        # from add_tool_result calls. After compaction those calls will
        # NOT replay (the digest has folded them) so any subsequent
        # _build_compact must not surface the now-stale results. Empty
        # both deques to maintain the alignment invariant.
        self._recent_results.clear()
        self._recent_results_meta.clear()
        self._msg_cache = None
        self._tok_cache = None
        return True

    # ── Internal ──────────────────────────────────────────

    def _extract_tool_info(self, assistant_msg: dict, tool_call_id: str) -> tuple[str, str]:
        """Extract tool name and args summary from an assistant message by tool_call_id."""
        tool_calls = assistant_msg.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("id") == tool_call_id:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                # Summarize args: truncate long values
                if isinstance(args, str) and len(args) > self._args_summary_chars:
                    args = args[:self._args_summary_chars - 3] + "..."
                return name, args
        return "?", ""

    def _build_compact(self) -> list[dict]:
        """Build 2-message prompt from turn entries + recent results."""
        parts = [f"Task: {self._original_prompt}"]

        # Progress log. Each turn line includes a size annotation so the
        # model can see the magnitude of every tool result
        # — not just the verb. Format: `[N L, M chars]` for results
        # over a threshold. Suppressed for empty / tiny results
        # (≤ 200 chars) where the annotation would be wasted bytes
        # (touch / mv / chmod / "OK" responses). Shown on EVERY eligible
        # turn including ones whose full body still appears verbatim
        # below — slight redundancy, but the cutoff would need to know
        # about the per-budget eviction that runs later in this method,
        # and over-showing is cheap.
        if self._turn_entries:
            lines = []
            for e in self._turn_entries:
                reason = e.reasoning.replace("\n", " ").strip()
                if len(reason) > self._trace_reasoning_chars:
                    reason = reason[:self._trace_reasoning_chars - 3] + "..."
                size_ann = ""
                if e.result_chars > 200:
                    size_ann = f" [{e.result_lines}L, {_compact_byte_count(e.result_chars)}]"
                if reason:
                    lines.append(
                        f"- T{e.turn}: \"{reason}\" → {e.tool_name}({e.args_summary}) {e.outcome}{size_ann}"
                    )
                else:
                    lines.append(
                        f"- T{e.turn}: {e.tool_name}({e.args_summary}) {e.outcome}{size_ann}"
                    )
            parts.append("Progress:\n" + "\n".join(lines))

        # File-read dedup pre-pass (Cline `attemptFileReadOptimizationCore`).
        # Walk from
        # newest to oldest tracking the (path → newest_index) of every
        # `read` we see; for each older read of the same path, mark it
        # for elision IF no intervening mutation on that same path has
        # happened in between (the mutation invalidates the cached
        # bytes). Replace the elided entry with a one-line stub so the
        # progress log still shows the read happened.
        #
        # Without this pass: a session that re-reads the same 5k-line
        # file 4 times consumes 4× the budget for the same content.
        # With this pass: only the latest copy survives the budget; the
        # older 3 collapse to ~80 chars each.
        if self._recent_results:
            n = len(self._recent_results)
            results_list = list(self._recent_results)
            meta_list = list(self._recent_results_meta)
            # First, scan front-to-back to find the latest read index per
            # path AND the latest mutation index per path. A read at
            # index i is elidable iff there is a later read at j > i on
            # the same path AND no mutation k with i < k < j on that path.
            latest_read_idx: dict[str, int] = {}
            mutation_indices: dict[str, list[int]] = {}
            elidable: set[int] = set()
            for i, meta in enumerate(meta_list):
                tn = (meta.get("tool_name") or "")
                p = meta.get("path")
                if not p:
                    continue
                if tn == "read":
                    if p in latest_read_idx:
                        prev_i = latest_read_idx[p]
                        muts = mutation_indices.get(p, [])
                        if not any(prev_i < k < i for k in muts):
                            elidable.add(prev_i)
                    latest_read_idx[p] = i
                if tn in _MUTATION_TOOLS:
                    mutation_indices.setdefault(p, []).append(i)

            elided_count = 0
            elided_chars_saved = 0
            if elidable:
                for i in elidable:
                    p = meta_list[i].get("path") or "?"
                    later_idx = latest_read_idx.get(p, i)
                    later_turn = self._turn_entries[-(n - later_idx)].turn if (n - later_idx) <= len(self._turn_entries) else later_idx
                    stub = f"[duplicate read of {p} elided — see entry from T{later_turn}]"
                    elided_chars_saved += max(0, len(results_list[i]) - len(stub))
                    results_list[i] = stub
                    elided_count += 1
                # Record the savings exactly as a context_strategy mechanism.
                if elided_chars_saved > 0:
                    from ..savings import get_ledger
                    get_ledger().record(
                        bucket="context_projection",
                        layer="context_strategy",
                        mechanism="compact_transcript_read_dedup",
                        input_chars=elided_chars_saved + sum(len(r) for r in results_list),
                        output_chars=sum(len(r) for r in results_list),
                        measure_type="exact",
                        ctx={"elided_reads": elided_count},
                    )

            # Rolling tool-result window, char-budgeted newest-first.
            # Walk from the most recent result backward, keeping entries
            # until the char budget is exhausted; drop older ones
            # permanently so the deque stays bounded across long runs.
            kept_rev: list[str] = []
            chars_used = 0
            for content in reversed(results_list):
                if chars_used + len(content) > self._recent_results_chars and kept_rev:
                    break
                kept_rev.append(content)
                chars_used += len(content)
            # Drop the same number of oldest entries from BOTH parallel
            # deques so the alignment invariant holds across turns.
            while len(self._recent_results) > len(kept_rev):
                self._recent_results.popleft()
                if self._recent_results_meta:
                    self._recent_results_meta.popleft()
            results = "\n---\n".join(reversed(kept_rev))
            parts.append(
                f"Last {len(kept_rev)} tool results (full, newest last):\n{results}"
            )

        parts.append(f"Turn: {self._turn_count}")

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "compact"
CONTEXT_CLASS = CompactTranscript
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=1,
    state_source="append_only_messages",
    source_type="append_log",
    normal_prompt_sources=(
        "in_memory_append_log",
        "in_memory_turn_entries",
        "in_memory_recent_tool_results",
    ),
    file_freshness="snapshot+dedup",
    injection_support="buried_in_projection",
    constructor_config_attrs=COMPACT_CONSTRUCTOR_CONFIG_ATTRS,
)
