"""YslotContext — with_yuj slot-contract experiment.

Slot-based alternative to yconcise: yuj labels, `.solver/state.json`
continuity, and one inline candidate file.
"""
from __future__ import annotations

from collections.abc import Callable

from ..context import chars_div_4
from ._working_set_baseline import WorkingSetBaselineContext
from ._metadata import (
    STATEFUL_BUDGET_CONFIG_ATTRS,
    YSLOT_CONSTRUCTOR_CONFIG_ATTRS,
    ContextModeMetadata,
)


class YslotContext(WorkingSetBaselineContext):
    """Yuj-label slot-contract mode for the with_yuj arm."""

    def __init__(
        self,
        cwd: str,
        original_prompt: str,
        *,
        trace_lines: int,
        evidence_lines: int,
        recent_tool_results_chars: int,
        trace_stub_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        suffix: str,
        todos_char_budget: int = 2000,
        inspect_repeat_threshold: int = 0,
        recovery_same_target_threshold: int = 0,
        recovery_verify_repeat_threshold: int = 0,
        slot_max_candidates: int = 1,
        slot_inline_files: int = 1,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        del trace_stub_chars
        super().__init__(
            cwd=cwd,
            original_prompt=original_prompt,
            recent_results_chars=recent_tool_results_chars,
            trace_reasoning_chars=trace_reasoning_chars,
            min_turns=min_turns,
            args_summary_chars=args_summary_chars,
            trace_lines=trace_lines,
            evidence_lines=evidence_lines,
            suffix=suffix,
            todos_char_budget=todos_char_budget,
            use_solver_state=True,
            style="yuj",
            contract="slot",
            inspect_repeat_threshold=inspect_repeat_threshold,
            recovery_same_target_threshold=recovery_same_target_threshold,
            recovery_verify_repeat_threshold=recovery_verify_repeat_threshold,
            slot_max_candidates=slot_max_candidates,
            slot_inline_files=slot_inline_files,
            savings_mechanism="yslot_context",
            token_estimator=token_estimator,
        )


CONTEXT_MODE = "yslot"
CONTEXT_CLASS = YslotContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=6,
    state_source="append_only_messages+.solver/state.json",
    source_type="hybrid",
    normal_prompt_sources=(
        "in_memory_append_log",
        "in_memory_working_set",
        ".solver/state.json_when_present",
        "live_workspace_files_for_touched_paths",
    ),
    file_freshness="live",
    injection_support="buried_in_projection",
    budget_config_attrs=STATEFUL_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=YSLOT_CONSTRUCTOR_CONFIG_ATTRS,
)
