"""SolverStateContext — build messages from .solver/state.json, not raw history.

Every turn, constructs a two-message prompt:
  [system] system prompt (static, cached)
  [user]   .solver/ state + recent tool results (bounded, partially cached)

The model never sees conversation history. .solver/state.json IS the memory.
Recent tool results are preserved in a rolling char-budget window so that
code reads remain visible across the 2-3 turns it takes to go from "read
the file" to "edit the file" — otherwise the model would see a 200-char
stub of its own read by the next decision turn.
"""
import json
from collections import deque
from collections.abc import Callable
from pathlib import Path

from ..context import ContextManager, chars_div_4


from ._solver_state_dedup import apply_dedup
from ._solver_state_format import format_list, format_state, format_trace
# Re-exports so existing imports `from .solver_state_context import _TEST_PREFIXES`
# / `_classify_cmd` / `_dedup_message` / `_extract_error_snippet` keep working.
from ._solver_state_helpers import (
    _READ_PREFIXES, _SEARCH_PREFIXES, _TEST_PREFIXES,
    _classify_cmd, _dedup_message, _extract_error_snippet,
)
from ._solver_state_io import prepopulate_from_trace as _prepopulate_from_trace
from ._metadata import (
    STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
    STATEFUL_SECTION_LABELS,
    STATEFUL_SECTION_ORDER,
    ContextModeMetadata,
)


class SolverStateContext(ContextManager):
    """Builds messages from .solver/ state each turn.

    After .solver/ exists, every prompt is exactly two messages:
      system: static prompt (always cached)
      user: .solver/ state + last tool results (partially cached)

    Falls back to append-only behavior until .solver/ files exist (turns 0-1).

    All numeric tunables are required kwargs — no module-level shadow
    defaults. The harness wires them from config.toml through Config.
    """

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
        ignore_state: bool = False,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._cwd = Path(cwd)
        self._original_prompt = original_prompt
        self._trace_lines = trace_lines
        self._evidence_lines = evidence_lines
        self._inference_lines = inference_lines
        self._recent_tool_results_chars = recent_tool_results_chars
        self._trace_stub_chars = trace_stub_chars
        self._min_turns = min_turns
        self._suffix = suffix
        self._ignore_state = ignore_state

        # Internal state
        self._system_content: str = ""
        self._all_messages: list[dict] = []  # raw append log (fallback only)
        # Rolling window of recent tool results (newest append). No per-turn
        # reset — a code read at turn 3 must still be visible at turn 5 when
        # the model decides to edit. Bounded by self._recent_tool_results_chars
        # in _format_tool_results, not by entry count.
        self._recent_tool_results: deque[dict] = deque()
        self._turn_count: int = 0
        self._file_cache: dict[str, str] | None = None  # cached .solver/ file contents
        # Raw-parse cache shared with subclasses that re-read state.json
        # (CompoundContext does this for reasoning-aware trace + fail-
        # first evidence rendering). Invalidated in lockstep with
        # _file_cache — a new tool result means state.json has been
        # regenerated and the parse is stale.
        self._raw_state_cache: dict | None = None
        # Escalating dedup: tracks how many times each unique output has
        # been deduplicated. Keyed by hash(content). Escalation:
        #   1st dedup (2nd attempt): behavioral warning
        #   2nd+ dedup (3rd+ attempt): hard block
        # Cleared on successful write/edit (code change invalidates the
        # assumption that repeated commands produce identical output).
        self._dedup_counts: dict[int, int] = {}
        self._dedup_epoch: int = 0
        # Per-turn message + token caches. _build_from_solver rebuilds the
        # full user-message payload (reads state.json, formats trace,
        # splits evidence). Both get_messages and estimate_tokens are
        # called once per turn, so the cache pays back as soon as
        # estimate_tokens runs after get_messages within the same turn.
        # Every add_* invalidates both. The file-level _file_cache still
        # exists and is invalidated in add_tool_result; the message cache
        # is a superset layered on top.
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
        self._msg_cache = None
        self._tok_cache = None
        # Do NOT clear _recent_tool_results here. The original design cleared
        # them on every new turn, which meant: at turn N the model could only
        # see the tool result from turn N-1. A code read at turn 3 was
        # invisible by turn 5, so the model could never make an edit decision
        # grounded in a file it had read 2+ turns earlier. Now the rolling
        # window is trimmed only by char budget in _format_tool_results.
        self._turn_count += 1

    def reset_dedup_counts(self) -> None:
        """Clear dedup escalation state.

        Called by the session loop after a successful write/edit — a code
        change invalidates the assumption that repeated commands will produce
        identical output.
        """
        self._dedup_counts.clear()
        # Increment epoch so both dedup tiers skip pre-edit entries.
        # Replaces the old cmd_sig stripping approach — epoch handles
        # both cmd-sig (tier 1) and content (tier 2) in one shot.
        self._dedup_epoch += 1

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        original_chars = len(content)
        content, dedup_fired, dedup_tier = apply_dedup(
            content,
            tool_name=tool_name,
            cmd_signature=cmd_signature,
            recent_tool_results=self._recent_tool_results,
            dedup_counts=self._dedup_counts,
            dedup_epoch=self._dedup_epoch,
            turn_count=self._turn_count,
        )

        # Token accounting: record exact dedup savings when either tier fires.
        if dedup_fired and original_chars != len(content):
            from ..savings import get_ledger
            get_ledger().record(
                bucket="dedup",
                layer="context_strategy",
                mechanism=dedup_tier,
                input_chars=original_chars,
                output_chars=len(content),
                measure_type="exact",
                ctx={"tool_name": tool_name, "gate_blocked": gate_blocked},
            )

        # Gate-blocked entries get epoch -1 so dedup tiers never match
        # against them — the tool was never executed, the content is a
        # gate message, not real output.
        entry_epoch = -1 if gate_blocked else self._dedup_epoch
        msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content, "_cmd_sig": cmd_signature, "_epoch": entry_epoch}
        self._all_messages.append(msg)
        self._recent_tool_results.append(msg)
        self._file_cache = None  # tool execution may have written to .solver/state.json
        self._raw_state_cache = None
        self._msg_cache = None
        self._tok_cache = None

    def get_messages(self) -> list[dict]:
        """Build messages from .solver/state.json if available, else fall back to full list."""
        if self._msg_cache is not None:
            return self._msg_cache
        solver_dir = self._cwd / ".solver"
        if (
            self._ignore_state
            or not (solver_dir / "state.json").is_file()
            or self._turn_count < self._min_turns
        ):
            self._msg_cache = self._all_messages
        else:
            self._msg_cache = self._build_from_solver(solver_dir)
            # Token accounting: solver-state projection vs. full append log.
            from ..savings import get_ledger
            full_chars = sum(len(str(m)) for m in self._all_messages)
            actual_chars = sum(len(str(m)) for m in self._msg_cache)
            get_ledger().record(
                bucket="context_projection",
                layer="context_strategy",
                mechanism=type(self).__name__,
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
        rendered + stitched with the latest assistant/tool pair.
        """
        self._all_messages = list(new_messages)
        self._msg_cache = None
        self._tok_cache = None
        return True

    # ── Internal ──────────────────────────────────────────

    _EMPTY_SECTIONS = {
        "state": "",
        "trace": "",
        "evidence": "",
    }

    def _get_solver_files(self, solver_dir: Path) -> dict[str, str]:
        """Read .solver/state.json and format each section as text.

        Cache is invalidated by add_tool_result() — the only point where
        tool execution may have written to .solver/state.json. Between tool
        results, multiple get_messages()/estimate_tokens() calls reuse the
        same read.

        Missing or empty file → empty sections (expected on first turn).
        Malformed JSON → JSONDecodeError surfaces. The model wrote garbage;
        the failure is evidence per the protocol.
        """
        if self._file_cache is not None:
            return self._file_cache

        state_path = solver_dir / "state.json"
        if not state_path.is_file():
            self._file_cache = dict(self._EMPTY_SECTIONS)
            return self._file_cache

        raw = state_path.read_text().strip()
        if not raw:
            self._file_cache = dict(self._EMPTY_SECTIONS)
            return self._file_cache

        data = json.loads(raw)
        self._file_cache = {
            "state": format_state(data.get("state")),
            "trace": format_trace(data.get("trace", []), self._trace_lines, self._trace_stub_chars),
            "evidence": format_list(data.get("evidence", []), self._evidence_lines),
        }
        return self._file_cache

    def _format_state(self, state) -> str:
        return format_state(state)

    def _format_trace(self, trace, max_entries: int) -> str:
        return format_trace(trace, max_entries, self._trace_stub_chars)

    def _format_list(self, items, max_items: int) -> str:
        return format_list(items, max_items)

    def prepopulate_from_trace(self) -> int:
        return _prepopulate_from_trace(
            self._cwd, self._recent_tool_results, self._recent_tool_results_chars,
        )

    def _format_tool_results(self) -> str:
        """Format recent tool results for injection into user message.

        Walks the rolling window newest-first, accumulating full contents
        until recent_tool_results_chars is exhausted. Older results drop
        out. The result is rendered oldest-to-newest so the model reads
        it in chronological order.

        Also trims the deque as a side-effect: anything that didn't fit
        in this turn's window is evicted permanently so the memory
        footprint stays bounded across long runs.
        """
        if not self._recent_tool_results:
            return ""
        # Walk newest to oldest, keep within char budget.
        kept_rev: list[dict] = []
        chars_used = 0
        for tr in reversed(self._recent_tool_results):
            content = tr.get("content") or ""
            if chars_used + len(content) > self._recent_tool_results_chars and kept_rev:
                break
            kept_rev.append(tr)
            chars_used += len(content)
        # Evict anything beyond what we kept so the deque doesn't grow
        # without bound over a long session.
        while len(self._recent_tool_results) > len(kept_rev):
            self._recent_tool_results.popleft()
        parts = [tr["content"] for tr in reversed(kept_rev)]
        results = "\n---\n".join(parts)
        label = (
            f"=== Tool results (last {len(kept_rev)}, newest last) ==="
            if len(kept_rev) > 1
            else "=== Tool result from your last action ==="
        )
        return f"{label}\n{results}"

    def _build_from_solver(self, solver_dir: Path) -> list[dict]:
        """Build a two-message prompt: system + user.

        User message contains .solver/state.json sections + last tool results.
        No conversation history. No assistant messages. Bounded size.
        """
        files = self._get_solver_files(solver_dir)

        # Build the context summary
        parts = [f"Task: {self._original_prompt}"]

        if files["state"]:
            parts.append(f"=== Current state ===\n{files['state']}")
        if files["trace"]:
            parts.append(f"=== Progress trace (recent) ===\n{files['trace']}")
        if files["evidence"]:
            parts.append(f"=== Evidence ===\n{files['evidence']}")

        # Inject last tool results — the one-turn blind spot
        tool_results = self._format_tool_results()
        if tool_results:
            parts.append(tool_results)

        if self._suffix:
            parts.append(self._suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "stateful"
CONTEXT_CLASS = SolverStateContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=7,
    message_shape="two-message state projection after min_turns_before_context",
    state_source=".solver/state.json",
    source_type="trace_state",
    normal_prompt_sources=(
        ".solver/state.json",
        "in_memory_recent_tool_results",
        "live_workspace_files_from_state_trace_on_session_resume",
    ),
    section_order=STATEFUL_SECTION_ORDER,
    section_labels=STATEFUL_SECTION_LABELS,
    file_freshness="snapshot",
    injection_support="buried_in_projection",
    state_ignored_when_context_ignore_state=True,
    constructor_config_attrs=STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
)
