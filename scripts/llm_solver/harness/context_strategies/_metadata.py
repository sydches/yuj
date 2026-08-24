"""First-class context mode metadata.

The context manager class owns rendering behavior. The context mode owns the
registry-facing identity and artifact contract fields for that behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


BASE_BUDGET_CONFIG_ATTRS = (
    "min_turns_before_context",
    "solver_trace_lines",
    "solver_evidence_lines",
    "solver_inference_lines",
    "recent_tool_results_chars",
    "trace_stub_chars",
)

STATEFUL_BUDGET_CONFIG_ATTRS = (
    *BASE_BUDGET_CONFIG_ATTRS,
    "state_todos_char_budget",
)

FOCUSED_COMPOUND_BUDGET_CONFIG_ATTRS = (
    *STATEFUL_BUDGET_CONFIG_ATTRS,
    "focused_compound_trace_lines",
    "focused_compound_evidence_lines",
    "focused_compound_recent_tool_results_chars",
    "focused_compound_include_resolved_evidence",
)

COMPOUND_SELECTIVE_BUDGET_CONFIG_ATTRS = (
    *STATEFUL_BUDGET_CONFIG_ATTRS,
    "compound_selective_trace_lines",
    "compound_selective_unresolved_evidence_lines",
    "compound_selective_resolved_evidence_lines",
    "compound_selective_resolved_evidence_stub_chars",
    "compound_selective_recent_tool_results_chars",
    "compound_selective_trace_action_repeat_cap",
    "compound_selective_resolved_action_repeat_cap",
    "compound_selective_trace_anchor_lines",
    "compound_selective_resolved_anchor_lines",
    "compound_selective_trace_source_anchor_lines",
    "compound_selective_trace_test_anchor_lines",
    "compound_selective_resolved_source_anchor_lines",
    "compound_selective_resolved_test_anchor_lines",
)

HALFLIFE_BUDGET_CONFIG_ATTRS = (
    *BASE_BUDGET_CONFIG_ATTRS,
    "context_size",
    "halflife_context_limit_tokens",
    "halflife_no_decay_ratio",
    "halflife_verbatim_tool_results",
    "halflife_cap_7_chars",
    "halflife_cap_15_chars",
    "halflife_cap_31_chars",
    "halflife_cap_63_chars",
    "halflife_cap_older_chars",
)


@dataclass(frozen=True)
class ContextModeMetadata:
    """Registry and artifact metadata for one context mode."""

    cli_order: int
    message_shape: str = "strategy-defined"
    state_source: str = "strategy-defined"
    source_type: str = "strategy-defined"
    normal_prompt_sources: tuple[str, ...] = ("strategy-defined",)
    section_order: tuple[str, ...] = ("strategy_defined",)
    section_labels: Mapping[str, str] = field(default_factory=dict)
    file_freshness: str = "strategy-defined"
    injection_support: str = "strategy-defined"
    state_ignored_when_context_ignore_state: bool = False
    budget_config_attrs: tuple[str, ...] = BASE_BUDGET_CONFIG_ATTRS
    constructor_config_attrs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextMode:
    """Discovered context mode record."""

    name: str
    cls: type
    metadata: ContextModeMetadata


STATEFUL_SECTION_ORDER = (
    "task",
    "current_state",
    "progress_trace",
    "evidence",
    "tool_results",
    "todos",
    "continuation_suffix",
)

STATEFUL_SECTION_LABELS = {
    "task": "Task: <prompt>",
    "current_state": "=== Current state ===",
    "progress_trace": "=== Progress trace (recent) ===",
    "evidence": "=== Evidence ===",
    "tool_results": "=== Tool result from your last action ===",
    "todos": "=== Todos ===",
    "continuation_suffix": "<state_context_suffix>",
}

COMPOUND_SECTION_ORDER = (
    "task",
    "state",
    "gate_blocking",
    "trace",
    "evidence",
    "tool_results",
    "todos",
    "continuation_suffix",
)

COMPOUND_SECTION_LABELS = {
    "task": "Task: <prompt>",
    "state": "=== State ===",
    "gate_blocking": "=== Gate (blocking) ===",
    "trace": "=== Trace ===",
    "evidence": "=== Evidence ===",
    "tool_results": "=== Tool result(s) ===",
    "todos": "=== Todos ===",
    "continuation_suffix": "<state_context_suffix>",
}

SALIENCE_SECTION_ORDER = (
    "task",
    "next_action_contract",
    "state",
    "gate_blocking",
    "salience_pressure",
    "trace",
    "evidence",
    "tool_results",
    "todos",
    "continuation_suffix",
)

SALIENCE_SECTION_LABELS = {
    **COMPOUND_SECTION_LABELS,
    "next_action_contract": "=== Next Action Contract ===",
    "salience_pressure": "=== Salience Pressure ===",
}


COMPACT_CONSTRUCTOR_CONFIG_ATTRS = {
    "recent_results_chars": "recent_tool_results_chars",
    "trace_reasoning_chars": "trace_reasoning_chars",
    "min_turns": "min_turns_before_context",
    "args_summary_chars": "args_summary_chars",
}

WORKING_SET_CONSTRUCTOR_CONFIG_ATTRS = {
    "cwd": "cwd",
    **COMPACT_CONSTRUCTOR_CONFIG_ATTRS,
    "inspect_repeat_threshold": "context_inspect_repeat_threshold",
}

SLOT_CONSTRUCTOR_CONFIG_ATTRS = {
    **WORKING_SET_CONSTRUCTOR_CONFIG_ATTRS,
    "recovery_same_target_threshold": "contract_recovery_same_target_threshold",
    "recovery_verify_repeat_threshold": "contract_recovery_verify_repeat_threshold",
    "slot_max_candidates": "context_slot_max_candidates",
    "slot_inline_files": "context_slot_inline_files",
}

STATEFUL_CONSTRUCTOR_CONFIG_ATTRS = {
    "cwd": "cwd",
    "trace_lines": "solver_trace_lines",
    "evidence_lines": "solver_evidence_lines",
    "inference_lines": "solver_inference_lines",
    "recent_tool_results_chars": "recent_tool_results_chars",
    "trace_stub_chars": "trace_stub_chars",
    "min_turns": "min_turns_before_context",
    "suffix": "state_context_suffix",
    "todos_char_budget": "state_todos_char_budget",
    "ignore_state": "context_ignore_state",
}

YWORKING_SET_CONSTRUCTOR_CONFIG_ATTRS = {
    **STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
    "trace_reasoning_chars": "trace_reasoning_chars",
    "args_summary_chars": "args_summary_chars",
    "inspect_repeat_threshold": "context_inspect_repeat_threshold",
}

YSLOT_CONSTRUCTOR_CONFIG_ATTRS = {
    **YWORKING_SET_CONSTRUCTOR_CONFIG_ATTRS,
    "recovery_same_target_threshold": "contract_recovery_same_target_threshold",
    "recovery_verify_repeat_threshold": "contract_recovery_verify_repeat_threshold",
    "slot_max_candidates": "context_slot_max_candidates",
    "slot_inline_files": "context_slot_inline_files",
}

FOCUSED_COMPOUND_CONSTRUCTOR_CONFIG_ATTRS = {
    **STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
    "focused_trace_lines": "focused_compound_trace_lines",
    "focused_evidence_lines": "focused_compound_evidence_lines",
    "focused_recent_tool_results_chars": "focused_compound_recent_tool_results_chars",
    "focused_include_resolved_evidence": (
        "focused_compound_include_resolved_evidence"
    ),
}

COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS = {
    **STATEFUL_CONSTRUCTOR_CONFIG_ATTRS,
    "selective_trace_lines": "compound_selective_trace_lines",
    "selective_unresolved_evidence_lines": (
        "compound_selective_unresolved_evidence_lines"
    ),
    "selective_resolved_evidence_lines": (
        "compound_selective_resolved_evidence_lines"
    ),
    "selective_resolved_evidence_stub_chars": (
        "compound_selective_resolved_evidence_stub_chars"
    ),
    "selective_recent_tool_results_chars": (
        "compound_selective_recent_tool_results_chars"
    ),
    "selective_trace_action_repeat_cap": (
        "compound_selective_trace_action_repeat_cap"
    ),
    "selective_resolved_action_repeat_cap": (
        "compound_selective_resolved_action_repeat_cap"
    ),
    "selective_trace_anchor_lines": "compound_selective_trace_anchor_lines",
    "selective_resolved_anchor_lines": "compound_selective_resolved_anchor_lines",
    "selective_trace_source_anchor_lines": (
        "compound_selective_trace_source_anchor_lines"
    ),
    "selective_trace_test_anchor_lines": (
        "compound_selective_trace_test_anchor_lines"
    ),
    "selective_resolved_source_anchor_lines": (
        "compound_selective_resolved_source_anchor_lines"
    ),
    "selective_resolved_test_anchor_lines": (
        "compound_selective_resolved_test_anchor_lines"
    ),
}

HALFLIFE_CONSTRUCTOR_CONFIG_ATTRS = {
    "context_size": "context_size",
    "context_limit_tokens": "halflife_context_limit_tokens",
    "activation_ratio": "halflife_no_decay_ratio",
    "verbatim_tool_results": "halflife_verbatim_tool_results",
    "cap_7_chars": "halflife_cap_7_chars",
    "cap_15_chars": "halflife_cap_15_chars",
    "cap_31_chars": "halflife_cap_31_chars",
    "cap_63_chars": "halflife_cap_63_chars",
    "cap_older_chars": "halflife_cap_older_chars",
}

LEGACY_CONTEXT_CONSTRUCTOR_CONFIG_ATTRS = {
    **COMPACT_CONSTRUCTOR_CONFIG_ATTRS,
    **SLOT_CONSTRUCTOR_CONFIG_ATTRS,
    **YWORKING_SET_CONSTRUCTOR_CONFIG_ATTRS,
    **YSLOT_CONSTRUCTOR_CONFIG_ATTRS,
    **FOCUSED_COMPOUND_CONSTRUCTOR_CONFIG_ATTRS,
    **COMPOUND_SELECTIVE_CONSTRUCTOR_CONFIG_ATTRS,
    **HALFLIFE_CONSTRUCTOR_CONFIG_ATTRS,
}
