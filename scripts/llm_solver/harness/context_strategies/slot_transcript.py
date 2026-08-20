"""SlotTranscript — wo_yuj slot-contract experiment.

Renderer-only alternative to concise: small slot-based state, one inline
candidate file, and no rolling prose trace.
"""
from __future__ import annotations

from collections.abc import Callable

from ..context import chars_div_4
from ._working_set_baseline import WorkingSetBaselineContext
from ._metadata import SLOT_CONSTRUCTOR_CONFIG_ATTRS, ContextModeMetadata


class SlotTranscript(WorkingSetBaselineContext):
    """Generic-label slot-contract mode for the wo_yuj arm."""

    def __init__(
        self,
        original_prompt: str,
        *,
        cwd: str,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        inspect_repeat_threshold: int = 0,
        recovery_same_target_threshold: int = 0,
        recovery_verify_repeat_threshold: int = 0,
        slot_max_candidates: int = 1,
        slot_inline_files: int = 1,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(
            cwd=cwd,
            original_prompt=original_prompt,
            recent_results_chars=recent_results_chars,
            trace_reasoning_chars=trace_reasoning_chars,
            min_turns=min_turns,
            args_summary_chars=args_summary_chars,
            style="generic",
            contract="slot",
            inspect_repeat_threshold=inspect_repeat_threshold,
            recovery_same_target_threshold=recovery_same_target_threshold,
            recovery_verify_repeat_threshold=recovery_verify_repeat_threshold,
            slot_max_candidates=slot_max_candidates,
            slot_inline_files=slot_inline_files,
            savings_mechanism="slot_transcript",
            token_estimator=token_estimator,
        )


CONTEXT_MODE = "slot"
CONTEXT_CLASS = SlotTranscript
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=3,
    state_source="append_only_messages",
    source_type="append_log",
    normal_prompt_sources=(
        "in_memory_append_log",
        "in_memory_working_set",
        "live_workspace_files_for_touched_paths",
    ),
    file_freshness="live",
    injection_support="buried_in_projection",
    constructor_config_attrs=SLOT_CONSTRUCTOR_CONFIG_ATTRS,
)
