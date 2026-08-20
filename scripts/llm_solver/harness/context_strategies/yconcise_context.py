"""YconciseContext — with_yuj concise baseline.

Action-oriented replacement for the original yconcise experiment. Uses the
shared WorkingSet baseline, yuj protocol labels, and reads `.solver/state.json`
when present for cross-session state continuity.
"""
from __future__ import annotations

from collections.abc import Callable

from ..context import chars_div_4
from ._working_set_baseline import WorkingSetBaselineContext
from ._metadata import YWORKING_SET_CONSTRUCTOR_CONFIG_ATTRS, ContextModeMetadata


class YconciseContext(WorkingSetBaselineContext):
    """Yuj-label concise mode for the with_yuj arm."""

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
        inspect_repeat_threshold: int = 0,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        # `trace_stub_chars` is intentionally unused in this baseline.
        # We surface one blocking payload and bound the whole prompt
        # globally instead of duplicating stub/full result pairs.
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
            use_solver_state=True,
            style="yuj",
            inspect_repeat_threshold=inspect_repeat_threshold,
            savings_mechanism="yconcise_context",
            token_estimator=token_estimator,
        )


CONTEXT_MODE = "yconcise"
CONTEXT_CLASS = YconciseContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=5,
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
    constructor_config_attrs=YWORKING_SET_CONSTRUCTOR_CONFIG_ATTRS,
)
