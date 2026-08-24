"""Shared concise/yconcise baseline on top of WorkingSet.

The first concise/yconcise attempt overfit to memory compression and grew
multiple ad hoc prompt assemblers. This module replaces that with one
shared, explicit contract:

- keep only the state the model can act on now
- surface one blocking output body, not many competing payloads
- show the current file working set, not a rolling transcript of reads
- bound the prompt as a whole, not section-by-section with overlapping caps

Both concise variants share the same ingestion and rendering machinery.
They differ only in section labels and whether they consult
``.solver/state.json`` for state/trace/evidence continuity.

Per-concern logic lives in sibling modules (``_working_set_baseline_*``):
focus, recovery, evidence, state. This file owns the public API
(``add_*``, ``get_messages``, ``replace_all_messages``…), the
ingestion-time bookkeeping, the section-budget orchestrator (``_build``),
and thin delegate methods that subclasses can still override.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..context import ContextManager, chars_div_4
from ..checkpoint_rewind import preserve_rewind_reports
from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from ..._shared.classification import classify_outcome as _classify_outcome
from ..edit_operations import edit_operations
from ..tool_specs import ACTION_WRITE_LIKE_TOOL_NAMES
from ._working_set import WorkingSet, GateSlot
from ._working_set_baseline_helpers import (  # noqa: F401
    _clean_reasoning, _cmd_display, _cmd_text, _extract_action_target,
    _extract_focus_target_from_command, _extract_test_target_from_action,
    _extract_test_target_from_command, _fit_blocks, _fit_lines,
    _is_inspection_action, _looks_like_path_token, _looks_like_test_path,
    _path_from_read_cmd, _pick_path, _truncate_text,
)
# Per-concern logic — delegated to via thin methods so subclasses can still override.
from . import _working_set_baseline_evidence as _E
from . import _working_set_baseline_focus as _F
from . import _working_set_baseline_recovery as _R
from . import _working_set_baseline_state as _S


_PATH_KEYS = ("path", "file_path")
_READ_TOOLS = frozenset({"read"})
_WRITE_TOOLS = ACTION_WRITE_LIKE_TOOL_NAMES
_BASH_READ_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_ACTION_PATH_RE = re.compile(
    r"(?:path|file_path)='([^']+)'|"
    r"(?:path|file_path)=\"([^\"]+)\"|"
    r"\"(?:path|file_path)\"\s*:\s*\"([^\"]+)\""
)
_ACTION_CMD_RE = re.compile(
    r"cmd='([^']+)'|cmd=\"([^\"]+)\"|\"cmd\"\s*:\s*\"([^\"]+)\""
)
_INSPECT_CMD_PREFIXES = ("ls", "find", "grep", "rg", "fd", "tree", "cat", "head", "tail")
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)


@dataclass(frozen=True)
class TurnEntry:
    turn: int
    reasoning: str
    tool_name: str
    args_summary: str
    outcome: str


@dataclass(frozen=True)
class TraceRecord:
    turn: int
    reasoning: str
    action: str
    outcome: str


@dataclass(frozen=True)
class EvidenceRecord:
    turn: int
    action: str
    verdict: str
    content: str
    first_turn: int | None = None
    repeat_count: int = 0


@dataclass(frozen=True)
class SectionSpec:
    title: str
    weight: int
    renderer: Callable[[int], str]


class WorkingSetBaselineContext(ContextManager):
    """Shared action-oriented baseline for concise/yconcise modes."""

    def __init__(
        self,
        *,
        cwd: str,
        original_prompt: str,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        trace_lines: int | None = None,
        evidence_lines: int | None = None,
        suffix: str = "",
        use_solver_state: bool = False,
        style: str = "generic",
        contract: str = "baseline",
        inspect_repeat_threshold: int = 0,
        recovery_same_target_threshold: int = 0,
        recovery_verify_repeat_threshold: int = 0,
        slot_max_candidates: int = 1,
        slot_inline_files: int = 1,
        savings_mechanism: str,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._cwd = Path(cwd)
        self._original_prompt = original_prompt
        self._char_budget = recent_results_chars
        self._trace_reasoning_chars = trace_reasoning_chars
        self._min_turns = min_turns
        self._args_summary_chars = args_summary_chars
        self._trace_lines = trace_lines
        self._evidence_lines = evidence_lines
        self._suffix = suffix
        self._use_solver_state = use_solver_state
        self._style = style
        self._contract = contract
        self._inspect_repeat_threshold = inspect_repeat_threshold
        self._recovery_same_target_threshold = recovery_same_target_threshold
        self._recovery_verify_repeat_threshold = recovery_verify_repeat_threshold
        self._slot_max_candidates = max(1, int(slot_max_candidates or 1))
        self._slot_inline_files = max(1, int(slot_inline_files or 1))
        self._savings_mechanism = savings_mechanism

        self._system_content: str = ""
        self._all_messages: list[dict] = []
        self._turn_entries: list[TurnEntry] = []
        self._ws = WorkingSet(cwd=self._cwd)
        self._turn_count = 0
        self._last_assistant_msg: dict | None = None
        self._prev_assistant_msg: dict | None = None

        self._msg_cache: list[dict] | None = None
        self._tok_cache: int | None = None
        self._raw_state_cache: dict | None = None

    # -- ingestion -------------------------------------------------

    def add_system(self, content: str) -> None:
        self._system_content = content
        self._all_messages.append({"role": "system", "content": content})
        self._invalidate()

    def add_user(self, content: str) -> None:
        self._all_messages.append({"role": "user", "content": content})
        self._invalidate()

    def add_assistant(self, message: dict) -> None:
        self._all_messages.append(message)
        self._last_assistant_msg = message
        self._turn_count += 1
        self._invalidate()

    def reset_dedup_counts(self) -> None:
        """No-op for interface parity with SolverStateContext."""
        return

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str = "",
        cmd_signature: str = "",
        gate_blocked: bool = False,
    ) -> None:
        self._all_messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "content": content,
        })
        self._invalidate()

        assistant_msg = self._last_assistant_msg or self._prev_assistant_msg
        tool_args: dict = {}
        reasoning = ""
        args_summary = ""
        resolved_name = tool_name
        if assistant_msg is not None:
            if self._last_assistant_msg is not None:
                reasoning = self._last_assistant_msg.get("content") or ""
                self._prev_assistant_msg = self._last_assistant_msg
                self._last_assistant_msg = None
            resolved_name, args_summary, tool_args = self._extract_tool_info(
                assistant_msg, tool_call_id,
            )
            if not tool_name:
                tool_name = resolved_name

        outcome = "BLOCKED" if gate_blocked else _classify_outcome(content)

        if gate_blocked:
            pass
        elif tool_name in _READ_TOOLS:
            path = _pick_path(tool_args)
            if path and outcome == "OK":
                self._ws.record_read(path, content, self._turn_count)
            else:
                self._ws.record_artifact(tool_name, args_summary, content,
                                         self._turn_count)
        elif tool_name in _WRITE_TOOLS:
            operations = edit_operations(tool_name, tool_args)
            if operations and outcome == "OK":
                for kind, path in operations:
                    if kind == "delete":
                        self._ws.forget_file(path)
                    else:
                        self._ws.record_mutation(path, self._turn_count)
            else:
                self._ws.record_artifact(tool_name, args_summary, content,
                                         self._turn_count)
        elif tool_name == "bash":
            effective_sig = cmd_signature
            cmd_text = tool_args.get("cmd") if isinstance(tool_args, dict) else None
            if not effective_sig and isinstance(cmd_text, str) and cmd_text:
                effective_sig = json.dumps({"cmd": cmd_text}, sort_keys=True)
            cmd_display = _cmd_display(effective_sig, args_summary)
            cmd_str = _cmd_text(effective_sig)
            bash_path = _path_from_read_cmd(cmd_str)
            if bash_path and outcome == "OK":
                self._ws.record_read(bash_path, content, self._turn_count)
            elif effective_sig:
                self._ws.record_gate(effective_sig, cmd_display, content,
                                     self._turn_count, outcome)
            else:
                self._ws.record_artifact(tool_name or "?", args_summary, content,
                                         self._turn_count)
        else:
            self._ws.record_artifact(tool_name or "?", args_summary, content,
                                     self._turn_count)

        self._turn_entries.append(TurnEntry(
            turn=self._turn_count,
            reasoning=reasoning,
            tool_name=tool_name or resolved_name or "?",
            args_summary=args_summary,
            outcome=outcome,
        ))

    # -- projection -----------------------------------------------

    def get_messages(self) -> list[dict]:
        if self._msg_cache is not None:
            return self._msg_cache
        if self._turn_count < self._min_turns:
            self._msg_cache = preserve_rewind_reports(
                self._all_messages, self._all_messages
            )
            return self._msg_cache
        self._msg_cache = preserve_rewind_reports(
            self._build(), self._all_messages
        )
        from ..savings import get_ledger
        full_chars = sum(len(str(m)) for m in self._all_messages)
        actual_chars = sum(len(str(m)) for m in self._msg_cache)
        get_ledger().record(
            bucket="context_projection",
            layer="context_strategy",
            mechanism=self._savings_mechanism,
            input_chars=full_chars,
            output_chars=actual_chars,
            measure_type="exact",
            ctx={"turn_count": self._turn_count, "messages": len(self._msg_cache)},
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
        self._invalidate()
        return True

    def snapshot_messages(self) -> list[dict]:
        """Snapshot the raw append log rather than the working-set view."""
        return copy.deepcopy(self._all_messages)

    def rewind_messages(self, new_messages: list[dict]) -> bool:
        """Rebuild the working set and turn records from a retained prefix."""
        retained = copy.deepcopy(list(new_messages))
        self._system_content = ""
        self._all_messages = []
        self._turn_entries.clear()
        self._ws = WorkingSet(cwd=self._cwd)
        self._turn_count = 0
        self._last_assistant_msg = None
        self._prev_assistant_msg = None
        self._invalidate()
        for message in retained:
            role = message.get("role")
            if role == "system":
                self.add_system(str(message.get("content") or ""))
            elif role == "user":
                self.add_user(str(message.get("content") or ""))
            elif role == "assistant":
                self.add_assistant(message)
            elif role == "tool":
                self.add_tool_result(
                    str(message.get("tool_call_id") or ""),
                    str(message.get("content") or ""),
                )
            else:
                self._all_messages.append(message)
        self._invalidate()
        return True

    def prepopulate_from_trace(self) -> int:
        state_path = self._cwd / ".solver" / "state.json"
        return self._ws.seed_from_state_trace(state_path, turn=0)

    # -- internal -------------------------------------------------

    def _invalidate(self) -> None:
        self._msg_cache = None
        self._tok_cache = None
        self._raw_state_cache = None

    def _extract_tool_info(
        self,
        assistant_msg: dict,
        tool_call_id: str,
    ) -> tuple[str, str, dict]:
        for tc in assistant_msg.get("tool_calls", []):
            if tc.get("id") != tool_call_id:
                continue
            func = tc.get("function", {})
            name = func.get("name", "?")
            raw = func.get("arguments", "")
            args_summary = raw if isinstance(raw, str) else json.dumps(raw)
            if len(args_summary) > self._args_summary_chars:
                args_summary = args_summary[: self._args_summary_chars - 3] + "..."
            parsed: dict = {}
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    parsed = json.loads(raw)
                except (ValueError, TypeError):
                    parsed = {}
            return name, args_summary, parsed
        return "?", "", {}

    def _load_state_json(self) -> dict:
        if self._raw_state_cache is not None:
            return self._raw_state_cache
        state_path = self._cwd / ".solver" / "state.json"
        if not state_path.is_file():
            self._raw_state_cache = {}
            return self._raw_state_cache
        try:
            self._raw_state_cache = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            self._raw_state_cache = {}
        return self._raw_state_cache

    def _has_solver_state(self) -> bool:
        return self._use_solver_state and bool(self._load_state_json())

    # -- delegating methods (per-concern logic in sibling modules) ---
    # Kept as one-liners so subclass overrides still work; logic lives in
    # _working_set_baseline_{state,focus,recovery,evidence}.py.

    def _state_text(self, max_chars: int) -> str: return _S.state_text(self, max_chars)
    def _phase_text(self) -> str: return _S.phase_text(self)
    def _slot_state_text(self, max_chars: int) -> str: return _S.slot_state_text(self, max_chars)
    def _obligation_text(self) -> str: return _S.obligation_text(self)
    def _last_action_text(self) -> str: return _S.last_action_text(self)

    def _focus_candidates(self) -> list[str]: return _F.focus_candidates(self)
    def _candidate_source_paths(self) -> list[str]: return _F.candidate_source_paths(self)
    def _candidate_test_paths(self) -> list[str]: return _F.candidate_test_paths(self)
    def _candidate_test_targets(self) -> list[str]: return _F.candidate_test_targets(self)
    def _focus_files_text(self) -> str: return _F.focus_files_text(self)
    def _changed_paths(self) -> list[str]: return _F.changed_paths(self)
    def _visible_paths(self) -> list[str]: return _F.visible_paths(self)
    def _recent_action_targets(self) -> list[str]: return _F.recent_action_targets(self)
    def _test_target_text(self) -> str: return _F.test_target_text(self)
    def _is_repo_file_candidate(self, path: str) -> bool: return _F.is_repo_file_candidate(self, path)
    def _needs_test_read(self) -> bool: return _F.needs_test_read(self)

    def _last_verdict_text(self) -> str: return _R.last_verdict_text(self)
    def _disallowed_repeat_text(self) -> str: return _R.disallowed_repeat_text(self)
    def _slot_next_action_text(self) -> str: return _R.slot_next_action_text(self)
    def _repeated_verify_run(self) -> tuple[TraceRecord, int, int] | None: return _R.repeated_verify_run(self)
    def _recovery_state(self) -> tuple[str, str] | None: return _R.recovery_state(self)
    def _latest_repeated_trace_run(self) -> tuple[TraceRecord, int, int] | None: return _R.latest_repeated_trace_run(self)
    def _repeated_target_text(self, action: str) -> str: return _R.repeated_target_text(self, action)

    def _format_path_list(self, paths: list[str], limit: int = 4) -> str:
        if not paths:
            return ""
        head = paths[:limit]
        suffix = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
        return ", ".join(head) + suffix

    def _trace_records(self) -> list[TraceRecord]: return _E.trace_records(self)
    def _evidence_records(self) -> list[EvidenceRecord]: return _E.evidence_records(self)
    def _latest_evidence_record(self) -> EvidenceRecord | None: return _E.latest_evidence_record(self)
    def _blocking_record(self) -> EvidenceRecord | None: return _E.blocking_record(self)
    def _gate_payload_text(self, max_chars: int) -> str: return _E.gate_payload_text(self, max_chars)
    def _summary_line(self, rec: EvidenceRecord) -> str: return _E.summary_line(rec)
    def _checks_text(self, max_chars: int) -> str: return _E.checks_text(self, max_chars)
    def _evidence_text(self, max_chars: int) -> str: return _E.evidence_text(self, max_chars)
    def _trace_text(self, max_chars: int) -> str: return _E.trace_text(self, max_chars)

    def _trace_block_generic(self, rec: TraceRecord, *, run_len: int, last_turn: int) -> str:
        return _E.trace_block_generic(self, rec, run_len=run_len, last_turn=last_turn)

    def _trace_block_yuj(self, rec: TraceRecord, *, prev_reasoning: str | None, run_len: int, last_turn: int) -> str:
        return _E.trace_block_yuj(self, rec, prev_reasoning=prev_reasoning, run_len=run_len, last_turn=last_turn)

    def _files_text(self, max_chars: int) -> str:
        if self._contract == "slot":
            selected = self._candidate_source_paths()[: self._slot_inline_files]
            if not selected:
                selected = self._changed_paths()[: self._slot_inline_files]
            rendered, elided = self._ws.project_selected_files(selected, max_chars)
            if not rendered and not elided:
                return ""
            lines = [rendered] if rendered else []
            if elided:
                lines.append("Files elided for budget: " + ", ".join(elided))
            return "\n".join(lines)
        rendered, elided = self._ws.project_files(max_chars)
        if not rendered and not elided:
            return ""
        lines = [rendered] if rendered else []
        if elided:
            lines.append("Files elided for budget: " + ", ".join(elided))
        return "\n".join(lines)

    def _artifacts_text(self, max_chars: int) -> str:
        return self._ws.project_artifacts(max_chars)

    def _generic_sections(self) -> list[SectionSpec]:
        if self._contract == "slot":
            return [
                SectionSpec("State:", 40, self._state_text),
                SectionSpec("Candidate file:", 38, self._files_text),
                SectionSpec("Blocking output:", 22, self._gate_payload_text),
            ]
        return [
            SectionSpec("State:", 18, self._state_text),
            SectionSpec("Blocking output:", 22, self._gate_payload_text),
            SectionSpec("Files (current content):", 30, self._files_text),
            SectionSpec("Checks:", 15, self._checks_text),
            SectionSpec("Progress:", 10, self._trace_text),
            SectionSpec("Recent outputs:", 5, self._artifacts_text),
        ]

    def _yuj_sections(self) -> list[SectionSpec]:
        if self._contract == "slot":
            return [
                SectionSpec("=== State ===", 40, self._state_text),
                SectionSpec("=== Candidate File ===", 38, self._files_text),
                SectionSpec("=== Gate (blocking) ===", 22, self._gate_payload_text),
            ]
        return [
            SectionSpec("=== State ===", 18, self._state_text),
            SectionSpec("=== Gate (blocking) ===", 22, self._gate_payload_text),
            SectionSpec("=== Evidence ===", 15, self._evidence_text),
            SectionSpec("=== Files ===", 30, self._files_text),
            SectionSpec("=== Trace ===", 15, self._trace_text),
            SectionSpec("=== Recent outputs ===", 5, self._artifacts_text),
        ]

    def _build(self) -> list[dict]:
        parts = [f"Task: {self._original_prompt}"]
        sections = self._yuj_sections() if self._style == "yuj" else self._generic_sections()
        remaining = self._char_budget
        remaining_weight = sum(section.weight for section in sections)

        for idx, section in enumerate(sections):
            if remaining <= 0:
                break
            if idx == len(sections) - 1 or remaining_weight <= 0:
                allocated = remaining
            else:
                allocated = max(256, int(remaining * section.weight / remaining_weight))
                allocated = min(remaining, allocated)
            text = section.renderer(allocated)
            remaining_weight -= section.weight
            if not text:
                continue
            parts.append(f"{section.title}\n{text}")
            remaining -= min(len(text), allocated)

        if self._suffix:
            parts.append(self._suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
