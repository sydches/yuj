"""Config field extraction and validation — extracted from config.py.

Internal to scripts.llm_solver; load_config() in config.py is the public entry.
"""
import copy
import logging
import math
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from ._shared.toml_compat import tomllib
from .config import Config, _REQUIRED_SECTIONS


def _matches_annotation(value: object, annotation: object) -> bool:
    """Return whether a TOML-derived value matches one Config annotation."""
    if annotation in {Any, object}:
        return True
    origin = get_origin(annotation)
    if origin in {types.UnionType, Union}:
        return any(
            _matches_annotation(value, item) for item in get_args(annotation)
        )
    if origin is list:
        if not isinstance(value, list):
            return False
        arguments = get_args(annotation)
        return not arguments or all(
            _matches_annotation(item, arguments[0]) for item in value
        )
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        arguments = get_args(annotation)
        if not arguments:
            return True
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches_annotation(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches_annotation(item, expected)
            for item, expected in zip(value, arguments)
        )
    if origin is dict:
        if not isinstance(value, dict):
            return False
        arguments = get_args(annotation)
        if not arguments:
            return True
        key_type, value_type = arguments
        return all(
            _matches_annotation(key, key_type)
            and _matches_annotation(child, value_type)
            for key, child in value.items()
        )
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation in {str, list, tuple, dict}:
        return isinstance(value, annotation)
    return True


def _validate_config_field_types(cfg: Config) -> None:
    """Reject TOML values that dataclasses otherwise accept without checks."""
    for config_field in fields(cfg):
        value = getattr(cfg, config_field.name)
        if _matches_annotation(value, config_field.type):
            continue
        expected = (
            str(config_field.type).replace("<class '", "").replace("'>", "")
        )
        raise ValueError(
            f"config error: resolved field {config_field.name} must be "
            f"{expected}, got {type(value).__name__}."
        )


def _string_tuple(value: object, *, path: str) -> tuple[str, ...]:
    """Validate a TOML string array without coercing scalars to characters."""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"config error: {path} must be an array of strings.")
    return tuple(value)


def _integer_tuple(value: object, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"config error: {path} must be an array of integers.")
    return tuple(value)


def _mapping_copy(value: object, *, path: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"config error: {path} must be a table/mapping.")
    return copy.deepcopy(dict(value))


def _list_copy(value: object, *, path: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"config error: {path} must be an array.")
    return copy.deepcopy(value)


def _require(data: dict, section: str, key: str) -> object:
    if section not in data:
        raise KeyError(f"config.toml missing section [{section}]")
    if key not in data[section]:
        raise KeyError(f"config.toml missing key '{key}' in [{section}]")
    return data[section][key]


def _extract_config_fields(d: dict) -> dict:
    """Project the nested TOML dict onto Config field names.

    Required keys raise KeyError on absence (no silent fallback). Experiment
    fields default to empty since they are meant to be per-run overrides.
    """
    missing = [s for s in _REQUIRED_SECTIONS if s not in d]
    if missing:
        raise KeyError(f"config.toml missing section(s): {missing}")

    experiment = d.get("experiment", {})
    analysis = d.get("analysis", {})
    from .harness.sandbox.env_policy import (
        DEFAULT_FIXED_ENVIRONMENT,
        EnvironmentPolicy,
    )
    sandbox_section = d.get("sandbox", {})
    sandbox_env_raw = sandbox_section.get("env")
    if sandbox_env_raw is None:
        # Preserve the pre-policy deterministic command baseline for older
        # complete configs that do not yet carry [sandbox.env].
        sandbox_env_raw = {"set": dict(DEFAULT_FIXED_ENVIRONMENT)}
    sandbox_env = EnvironmentPolicy.from_mapping(sandbox_env_raw)
    hooks_section = _mapping_copy(d.get("hooks", {}), path="hooks")
    return {
        "base_url": _require(d, "server", "base_url"),
        "api_key": _require(d, "server", "api_key"),
        "provider": d.get("server", {}).get("provider", "openai-compatible"),
        "timeout_connect": _require(d, "server", "timeout_connect"),
        "timeout_read": _require(d, "server", "timeout_read"),
        "health_poll_interval": _require(d, "server", "health_poll_interval"),
        "health_timeout": _require(d, "server", "health_timeout"),
        "launch_timeout": _require(d, "server", "launch_timeout"),
        "stop_settle": _require(d, "server", "stop_settle"),
        "server_request_extra": _mapping_copy(
            d.get("server", {}).get("request_extra", {}),
            path="server.request_extra",
        ),
        "cache_affinity": d.get("server", {}).get("cache_affinity", False),
        "cache_retention": d.get("server", {}).get("cache_retention", "off"),
        "cache_miss_warn_ratio": d.get("server", {}).get("cache_miss_warn_ratio", 0.0),
        "model": _require(d, "model", "name"),
        "profile_name": d.get("model", {}).get("profile_name", ""),
        "context_size": _require(d, "model", "context_size"),
        "context_fill_ratio": _require(d, "model", "context_fill_ratio"),
        "max_tokens_fraction": _require(d, "model", "max_tokens_fraction"),
        # max_tokens is derived from context_size * max_tokens_fraction at
        # solver startup (see __main__.py). The 0 here is a placeholder so
        # the dataclass is fully populated; __main__ replaces it.
        "max_tokens": 0,
        # HF tokenizer id (e.g. "Qwen/Qwen3-30B-A3B-Instruct-2507") or a
        # local directory path. When set, Session loads the tokenizer
        # once and uses it for exact token counts in
        # _maybe_compact_messages. Empty string falls back to chars_div_4.
        "tokenizer_id": d.get("model", {}).get("tokenizer_id", ""),
        "thinking_level": d.get("model", {}).get("thinking_level", "off"),
        "model_roles": _mapping_copy(
            d.get("models", {}).get("roles", {}),
            path="models.roles",
        ),
        "model_fallback_chain": d.get("models", {}).get("fallback_chain", {}),
        "model_fallback_revert": d.get("models", {}).get(
            "fallback_revert", "never"
        ),
        "advisor_enabled": d.get("advisor", {}).get("enabled", False),
        "advisor_model": d.get("advisor", {}).get("model", ""),
        "advisor_endpoint": d.get("advisor", {}).get("endpoint", ""),
        "advisor_every_n_turns": d.get("advisor", {}).get(
            "every_n_turns", 5
        ),
        "advisor_immune_turns": d.get("advisor", {}).get("immune_turns", 3),
        "advisor_max_note_chars": d.get("advisor", {}).get(
            "max_note_chars", 1200
        ),
        "max_turns": _require(d, "loop", "max_turns"),
        "max_sessions": _require(d, "loop", "max_sessions"),
        "rewind_enabled": d.get("loop", {}).get("rewind_enabled", False),
        "rewind_max_per_session": d.get("loop", {}).get(
            "rewind_max_per_session", 1
        ),
        "duplicate_abort": _require(d, "loop", "duplicate_abort"),
        "error_nudge_threshold": _require(d, "loop", "error_nudge_threshold"),
        "rumination_nudge_threshold": _require(d, "loop", "rumination_nudge_threshold"),
        "think_streak_nudge_after": d.get("loop", {}).get(
            "think_streak_nudge_after", 3
        ),
        "rumination_gate_max_blocks": d.get("loop", {}).get("rumination_gate_max_blocks", 0),
        "rumination_gate_arm_threshold": d.get("loop", {}).get("rumination_gate_arm_threshold", 0),
        "rumination_gate_arm_threshold_abs": d.get("loop", {}).get("rumination_gate_arm_threshold_abs", 0),
        "rumination_nudge_threshold_abs": d.get("loop", {}).get("rumination_nudge_threshold_abs", 0),
        "rumination_nudge_threshold_abs_post_mutation": d.get("loop", {}).get("rumination_nudge_threshold_abs_post_mutation", 0),
        "rumination_nudge_only_pre_mutation": d.get("loop", {}).get("rumination_nudge_only_pre_mutation", False),
        "rumination_same_target_warn_count": d.get("loop", {}).get("rumination_same_target_warn_count", 0),
        "rumination_same_target_arm_count": d.get("loop", {}).get("rumination_same_target_arm_count", 0),
        "test_read_warn_after": d.get("loop", {}).get("test_read_warn_after", 0),
        "context_inspect_repeat_threshold": d.get("loop", {}).get("context_inspect_repeat_threshold", 0),
        "tools_output_dedup_enabled": d.get("loop", {}).get("tools_output_dedup_enabled", True),
        "contract_commit_warn_after": d.get("loop", {}).get("contract_commit_warn_after", 0),
        "contract_commit_block_after": d.get("loop", {}).get("contract_commit_block_after", 0),
        "contract_recovery_same_target_threshold": d.get("loop", {}).get("contract_recovery_same_target_threshold", 0),
        "contract_recovery_verify_repeat_threshold": d.get("loop", {}).get("contract_recovery_verify_repeat_threshold", 0),
        "contract_invalid_repeat_abort_after": d.get("loop", {}).get("contract_invalid_repeat_abort_after", 0),
        "contract_abort_min_turns_since_commit_arm": d.get("loop", {}).get("contract_abort_min_turns_since_commit_arm", 0),
        "contract_abort_min_turns_since_recovery_arm": d.get("loop", {}).get("contract_abort_min_turns_since_recovery_arm", 0),
        "contract_abort_requires_zero_mutation": d.get("loop", {}).get("contract_abort_requires_zero_mutation", False),
        "contract_equivalent_action_classes_enabled": d.get("loop", {}).get("contract_equivalent_action_classes_enabled", False),
        "mutation_repeat_warn_after": d.get("loop", {}).get("mutation_repeat_warn_after", 0),
        "mutation_repeat_block_after": d.get("loop", {}).get("mutation_repeat_block_after", 0),
        "mutation_repeat_abort_after": d.get("loop", {}).get("mutation_repeat_abort_after", 0),
        "duplicate_warn_count": d.get("loop", {}).get("duplicate_warn_count", 0),
        "error_abort_threshold": d.get("loop", {}).get("error_abort_threshold", 0),
        "error_same_class_threshold": d.get("loop", {}).get("error_same_class_threshold", 0),
        "intent_abort_threshold": d.get("loop", {}).get("intent_abort_threshold", 0),
        "duplicate_guard_enabled": d.get("loop", {}).get("duplicate_guard_enabled", True),
        "post_edit_check_enabled": d.get("post_edit_check", {}).get("enabled", False),
        "post_edit_check_timeout": d.get("post_edit_check", {}).get("timeout", 10),
        "post_edit_checks": _list_copy(
            d.get("post_edit_check", {}).get("checks", []),
            path="post_edit_check.checks",
        ),
        "tools_lazy_loading_enabled": d.get("tools", {}).get(
            "lazy_loading_enabled", False
        ),
        "tools_active_default": _string_tuple(
            d.get("tools", {}).get(
                "active_default",
                ["bash", "read", "edit", "glob", "grep", "done"],
            ),
            path="tools.active_default",
        ),
        "tools_run_tests_enabled": d.get("tools", {}).get("run_tests", {}).get("enabled", False),
        "tools_run_tests_timeout": d.get("tools", {}).get("run_tests", {}).get("timeout", 240),
        "tools_run_tests_structured_output": d.get("tools", {}).get("run_tests", {}).get("structured_output", True),
        "tools_run_tests_assertion_context_lines": d.get("tools", {}).get("run_tests", {}).get("assertion_context_lines", 5),
        "tools_run_tests_assertion_context_max": d.get("tools", {}).get("run_tests", {}).get("assertion_context_max", 3),
        "tools_think_enabled": d.get("tools", {}).get(
            "think_enabled", False
        ),
        "tools_think_keep_turns": d.get("tools", {}).get(
            "think_keep_turns", 4
        ),
        "tools_list_definitions_enabled": d.get("tools", {}).get("list_definitions", {}).get("enabled", False),
        "tools_ast_search_enabled": d.get("tools", {}).get(
            "ast_search_enabled", False
        ),
        "tools_ast_search_max_rows": d.get("tools", {}).get(
            "ast_search_max_rows", 1000
        ),
        "tools_checkpoint_enabled": d.get("tools", {}).get(
            "checkpoint_enabled", False
        ),
        "tools_file_checkpoints_enabled": d.get("tools", {}).get(
            "file_checkpoints_enabled", False
        ),
        "tools_file_checkpoints_exclude": _string_tuple(
            d.get("tools", {}).get(
                "file_checkpoints_exclude",
                [
                    ".solver/**",
                    ".tool_output/**",
                    "prompt.txt",
                    "checkpoint.json",
                    "metrics.json",
                ],
            ),
            path="tools.file_checkpoints_exclude",
        ),
        "tools_stale_guard_mode": d.get("tools", {}).get(
            "stale_guard_mode", "warn"
        ),
        "tools_bash_redirect_read_side": d.get("tools", {}).get(
            "bash_redirect_read_side", False
        ),
        "tools_schema_validation": d.get("tools", {}).get(
            "schema_validation", "off"
        ),
        "tools_constrained_decoding": d.get("tools", {}).get(
            "constrained_decoding", "off"
        ),
        "tools_todos_enabled": d.get("tools", {}).get(
            "todos_enabled", False
        ),
        "tools_todos_max_items": d.get("tools", {}).get(
            "todos_max_items", 20
        ),
        "tools_background_enabled": d.get("tools", {}).get(
            "background_enabled", False
        ),
        "tools_background_max_procs": d.get("tools", {}).get(
            "background_max_procs", 4
        ),
        "tools_background_poll_timeout": d.get("tools", {}).get(
            "background_poll_timeout", 300.0
        ),
        "tools_task_enabled": d.get("tools", {}).get(
            "task_enabled", False
        ),
        "tools_subagent_depth": d.get("tools", {}).get(
            "subagent_depth", 1
        ),
        "tools_subagent_max_turns": d.get("tools", {}).get(
            "subagent_max_turns", 20
        ),
        "tools_exec_cell_enabled": d.get("tools", {}).get(
            "exec_cell_enabled", False
        ),
        "tools_exec_cell_timeout": d.get("tools", {}).get(
            "exec_cell_timeout", 30
        ),
        "lsp_enabled": d.get("lsp", {}).get("enabled", False),
        "lsp_servers": _mapping_copy(
            d.get("lsp", {}).get("servers", {}),
            path="lsp.servers",
        ),
        "lsp_diagnostics_timeout_s": d.get("lsp", {}).get(
            "diagnostics_timeout_s", 2.0
        ),
        "lsp_min_severity": d.get("lsp", {}).get("min_severity", "error"),
        "lsp_tool_enabled": d.get("lsp", {}).get("tool_enabled", False),
        "tools_apply_patch_enabled": d.get("tools", {}).get("apply_patch", {}).get("enabled", False),
        "tools_edit_format": d.get("tools", {}).get("edit_format", ""),
        "effective_edit_format": "",
        "tools_unified_envelope_enabled": d.get("tools", {}).get("unified_envelope", {}).get("enabled", False),
        "state_writer_enabled": d.get("state", {}).get("writer_enabled", True),
        "context_ignore_state": d.get("state", {}).get("context_ignore", False),
        "state_imperative_projection_enabled": d.get("state", {}).get("imperative_projection_enabled", False),
        "state_todos_char_budget": d.get("state", {}).get(
            "todos_char_budget", 2000
        ),
        "state_ignore_file_enabled": d.get("state", {}).get(
            "ignore_file_enabled", True
        ),
        "state_ignore_file_names": _string_tuple(
            d.get("state", {}).get("ignore_file_names", [".yujignore"]),
            path="state.ignore_file_names",
        ),
        "parallel_readonly_enabled": d.get("loop", {}).get("parallel_readonly_enabled", False),
        "parallel_max_workers": d.get("loop", {}).get("parallel_max_workers", 4),
        "injections_enabled": d.get("injections", {}).get("enabled", False),
        "injections_dir": d.get("injections", {}).get("dir", ".harness/injections"),
        "stream_rules_enabled": d.get("loop", {}).get(
            "stream_rules_enabled", False
        ),
        "stream_rules_dir": d.get("loop", {}).get(
            "stream_rules_dir", ".harness/stream_rules"
        ),
        "stream_rules_context_mode": d.get("loop", {}).get(
            "stream_rules_context_mode", "discard"
        ),
        "stream_rules_repeat_gap": d.get("loop", {}).get(
            "stream_rules_repeat_gap", 10
        ),
        "injections_path_rules_enabled": d.get("injections", {}).get(
            "path_rules_enabled", False
        ),
        "injections_path_rule_repeat": d.get("injections", {}).get(
            "path_rule_repeat", False
        ),
        "loop_detect_enabled": d.get("loop", {}).get("loop_detect_enabled", False),
        "turn_snapshots_enabled": d.get("loop", {}).get("turn_snapshots_enabled", False),
        "loop_detect_threshold": d.get("loop", {}).get("loop_detect_threshold", 5),
        "loop_detect_recovery": d.get("prompts", {}).get(
            "loop_detect_recovery",
            "<system-reminder>Loop detected: the last {streak} tool calls "
            "all have identical name and arguments. Stop repeating. Re-read "
            "the task, read a file you have not inspected yet, or change "
            "approach. One more repeat ends the session.</system-reminder>",
        ),
        "done_guard_enabled": d.get("loop", {}).get("done_guard_enabled", True),
        "rumination_enabled": d.get("loop", {}).get("rumination_enabled", True),
        "error_ladder_enabled": d.get("loop", {}).get("error_ladder_enabled", True),
        "preflight_reclip_enabled": d.get("loop", {}).get("preflight_reclip_enabled", True),
        "bash_transforms_universal_enabled": d.get("loop", {}).get("bash_transforms_universal_enabled", True),
        "bash_transforms_task_format_enabled": d.get("loop", {}).get("bash_transforms_task_format_enabled", True),
        "bash_transforms_structured_output_enabled": d.get("loop", {}).get("bash_transforms_structured_output_enabled", False),
        "bash_transforms_sink_threshold_chars": d.get("loop", {}).get("bash_transforms_sink_threshold_chars", 0),
        "rumination_gate_grace_calls": d.get("loop", {}).get("rumination_gate_grace_calls", 1),
        "rumination_min_threshold": d.get("loop", {}).get("rumination_min_threshold", 6),
        "done_require_mutation": d.get("loop", {}).get("done_require_mutation", True),
        "done_require_verify": d.get("loop", {}).get("done_require_verify", True),
        "done_verified_bash_min_chars": d.get("loop", {}).get("done_verified_bash_min_chars", 200),
        "allow_implicit_done": d.get("loop", {}).get("allow_implicit_done", True),
        "done_require_pretest_parity": d.get("loop", {}).get("done_require_pretest_parity", False),
        "done_parity_runs_required": d.get("loop", {}).get("done_parity_runs_required", 1),
        "adaptive_policy_enabled": d.get("loop", {}).get("adaptive_policy_enabled", False),
        "adaptive_switch_min_turn": d.get("loop", {}).get("adaptive_switch_min_turn", 0),
        "adaptive_requires_mutation": d.get("loop", {}).get("adaptive_requires_mutation", True),
        "adaptive_requires_test_signal": d.get("loop", {}).get("adaptive_requires_test_signal", True),
        "adaptive_low_pressure_window": d.get("loop", {}).get("adaptive_low_pressure_window", 0),
        "adaptive_low_pressure_max_events": d.get("loop", {}).get("adaptive_low_pressure_max_events", 0),
        "adaptive_phase2_done_guard_enabled": d.get("loop", {}).get("adaptive_phase2_done_guard_enabled", True),
        "adaptive_phase2_bash_task_format_enabled": d.get("loop", {}).get("adaptive_phase2_bash_task_format_enabled", True),
        "adaptive_phase2_bash_structured_output_enabled": d.get("loop", {}).get("adaptive_phase2_bash_structured_output_enabled", True),
        "adaptive_phase2_bash_sink_threshold_chars": d.get("loop", {}).get("adaptive_phase2_bash_sink_threshold_chars", 0),
        "adaptive_control_enabled": d.get("adaptive_control", {}).get("enabled", False),
        "adaptive_control_delivery": d.get("adaptive_control", {}).get("delivery", "in_place"),
        "adaptive_control_ledger_path": d.get("adaptive_control", {}).get("ledger_path", ""),
        "adaptive_control_evidence_regime": d.get("adaptive_control", {}).get("evidence_regime", "causal_live"),
        "adaptive_control_model": d.get("adaptive_control", {}).get("model", "in_process_pause"),
        "adaptive_control_target_hurdle_mode": d.get("adaptive_control", {}).get("target_hurdle_mode", ""),
        "adaptive_control_source_hindsight_hurdle_mode": d.get("adaptive_control", {}).get("source_hindsight_hurdle_mode", ""),
        "adaptive_control_online_signal_id": d.get("adaptive_control", {}).get("online_signal_id", ""),
        "adaptive_control_online_signal_ids": _string_tuple(
            d.get("adaptive_control", {}).get("online_signal_ids", []) or (),
            path="adaptive_control.online_signal_ids",
        ),
        "adaptive_control_intervention_target": d.get("adaptive_control", {}).get("intervention_target", ""),
        "adaptive_control_candidate_medicine_knob": d.get("adaptive_control", {}).get("candidate_medicine_knob", ""),
        "adaptive_control_candidate_config_path": d.get("adaptive_control", {}).get("candidate_config_path", ""),
        "adaptive_control_baseline_config_paths": _string_tuple(
            d.get("adaptive_control", {}).get("baseline_config_paths", []) or (),
            path="adaptive_control.baseline_config_paths",
        ),
        "adaptive_control_source_static_cell_id": d.get("adaptive_control", {}).get("source_static_cell_id", ""),
        "adaptive_control_source_instance_id": d.get("adaptive_control", {}).get("source_instance_id", ""),
        "adaptive_control_source_wave_id": d.get("adaptive_control", {}).get("source_wave_id", ""),
        "adaptive_control_source_cell_id": d.get("adaptive_control", {}).get("source_cell_id", ""),
        "adaptive_control_source_run_dir": d.get("adaptive_control", {}).get("source_run_dir", ""),
        "adaptive_control_debug": d.get("adaptive_control", {}).get("debug", "none"),
        "adaptive_control_debug_ledger_path": d.get("adaptive_control", {}).get("debug_ledger_path", ""),
        "adaptive_control_debug_include_prefix": d.get("adaptive_control", {}).get("debug_include_prefix", False),
        "adaptive_control_lookup_table_path": d.get("adaptive_control", {}).get("lookup_table_path", ""),
        "adaptive_control_policy_version": d.get("adaptive_control", {}).get("policy_version", "adaptive_policy_v0_replay"),
        "adaptive_control_detector_mode": d.get("adaptive_control", {}).get("detector_mode", "manual"),
        "adaptive_control_detector_version": d.get("adaptive_control", {}).get("detector_version", "zero_detector_v0"),
        "adaptive_control_detector_input_contract_path": d.get("adaptive_control", {}).get("detector_input_contract_path", ""),
        "adaptive_control_detector_rule_catalog_path": d.get("adaptive_control", {}).get("detector_rule_catalog_path", ""),
        "adaptive_control_medicine_table_version": d.get("adaptive_control", {}).get("medicine_table_version", "medicine_runtime_classes_v0"),
        "adaptive_control_intervention_space_version": d.get("adaptive_control", {}).get("intervention_space_version", "toml_overlay_control_v1"),
        "adaptive_control_runtime_executor_id": d.get("adaptive_control", {}).get("runtime_executor_id", ""),
        "adaptive_control_executor_status": d.get("adaptive_control", {}).get("executor_status", ""),
        "adaptive_control_max_interventions": d.get("adaptive_control", {}).get("max_interventions", 1),
        "adaptive_control_max_same_signal_interventions": d.get("adaptive_control", {}).get("max_same_signal_interventions", 1),
        "adaptive_control_disallow_repeat_intervention": d.get("adaptive_control", {}).get("disallow_repeat_intervention", True),
        "adaptive_control_watch_window_turns": d.get("adaptive_control", {}).get("watch_window_turns", 5),
        "adaptive_control_multi_intervention_enabled": d.get("adaptive_control", {}).get("multi_intervention_enabled", False),
        "adaptive_control_max_interventions_per_attempt": d.get("adaptive_control", {}).get("max_interventions_per_attempt", 1),
        "adaptive_control_max_interventions_per_hurdle_episode": d.get("adaptive_control", {}).get("max_interventions_per_hurdle_episode", 1),
        "adaptive_control_max_distinct_hurdle_episodes_per_attempt": d.get("adaptive_control", {}).get("max_distinct_hurdle_episodes_per_attempt", 1),
        "adaptive_control_cooldown_after_apply_slots": d.get("adaptive_control", {}).get(
            "cooldown_after_apply_slots",
            d.get("adaptive_control", {}).get("watch_window_turns", 5),
        ),
        "adaptive_control_branch_bundle_enabled": d.get("adaptive_control", {}).get("branch_bundle_enabled", False),
        "adaptive_control_branch_bundle_root": d.get("adaptive_control", {}).get("branch_bundle_root", ""),
        "adaptive_control_branch_bundle_source_run_id": d.get("adaptive_control", {}).get("branch_bundle_source_run_id", ""),
        "adaptive_control_branch_bundle_max_per_attempt": d.get("adaptive_control", {}).get("branch_bundle_max_per_attempt", 1),
        "adaptive_control_branch_watch_policy_id": d.get("adaptive_control", {}).get("branch_watch_policy_id", "prefix_rewind_watch_v1"),
        "harness_observation_enabled": d.get("harness_observation", {}).get("enabled", False),
        "harness_observation_grace_activity_turns": d.get("harness_observation", {}).get("grace_activity_turns", 2),
        "harness_observation_cadence_turns": d.get("harness_observation", {}).get("cadence_turns", 10),
        "harness_observation_packet_char_budget": d.get("harness_observation", {}).get("packet_char_budget", 1200),
        "harness_observation_evidence_lines": d.get("harness_observation", {}).get("evidence_lines", 3),
        "llm_hurdle_detector_enabled": d.get("llm_hurdle_detector", {}).get("enabled", False),
        "llm_hurdle_detector_cadence_turns": d.get("llm_hurdle_detector", {}).get("cadence_turns", 1),
        "llm_hurdle_detector_atlas_dictionary_path": d.get("llm_hurdle_detector", {}).get("atlas_dictionary_path", ""),
        "llm_hurdle_detector_input_contract_path": d.get("llm_hurdle_detector", {}).get("input_contract_path", ""),
        "llm_hurdle_detector_backend": d.get("adaptive_control", {}).get("detector_backend", "llm"),
        "llm_hurdle_detector_log_path": d.get("llm_hurdle_detector", {}).get("log_path", ""),
        "llm_hurdle_detector_max_trace_events": d.get("llm_hurdle_detector", {}).get("max_trace_events", 80),
        "llm_hurdle_detector_max_field_chars": d.get("llm_hurdle_detector", {}).get("max_field_chars", 800),
        "llm_hurdle_detector_max_state_snapshots": d.get("llm_hurdle_detector", {}).get("max_state_snapshots", 24),
        "llm_hurdle_detector_prompt_version": d.get(
            "llm_hurdle_detector", {}).get("prompt_version", "llm_hurdle_detector_prompt_v4"),
        # trace_nets thresholds live under [llm_hurdle_detector.trace_nets].
        # An omitted key uses the dataclass default.
        **{
            f"trace_nets_{k}": d.get("llm_hurdle_detector", {}).get(
                "trace_nets", {}
            ).get(k, _dflt)
            for k, _dflt in (
                ("fail_min_streak", 4), ("pass_lookback", 20),
                ("pass_min_prior", 2), ("pass_min_gap", 2),
                ("reread_min_args_len", 20), ("reread_min_gap", 3),
                ("reread_max_gap", 30), ("window", 30),
            )
        },
        "require_intent": d.get("loop", {}).get("require_intent", False),
        "intent_grace_turns": _require(d, "loop", "intent_grace_turns"),
        "min_turns_before_context": _require(d, "loop", "min_turns_before_context"),
        "guardrails_arm_after_turn": d.get("loop", {}).get("guardrails_arm_after_turn", 0),
        "max_output_chars": _require(d, "output", "max_output_chars"),
        "truncate_head_ratio": _require(d, "output", "truncate_head_ratio"),
        "truncate_head_lines": _require(d, "output", "truncate_head_lines"),
        "truncate_tail_lines": _require(d, "output", "truncate_tail_lines"),
        "args_summary_chars": _require(d, "output", "args_summary_chars"),
        "trace_args_summary_chars": _require(d, "output", "trace_args_summary_chars"),
        "trace_reasoning_store_chars": _require(d, "output", "trace_reasoning_store_chars"),
        "trace_result_summary_chars": _require(d, "output", "trace_result_summary_chars"),
        "solver_trace_lines": _require(d, "output", "solver_trace_lines"),
        "solver_evidence_lines": _require(d, "output", "solver_evidence_lines"),
        "solver_inference_lines": _require(d, "output", "solver_inference_lines"),
        "recent_tool_results_chars": _require(d, "output", "recent_tool_results_chars"),
        "trace_stub_chars": _require(d, "output", "trace_stub_chars"),
        "trace_reasoning_chars": _require(d, "output", "trace_reasoning_chars"),
        "context_slot_max_candidates": d.get("output", {}).get("context_slot_max_candidates", 1),
        "context_slot_inline_files": d.get("output", {}).get("context_slot_inline_files", 1),
        "focused_compound_trace_lines": d.get("output", {}).get("focused_compound_trace_lines", 0),
        "focused_compound_evidence_lines": d.get("output", {}).get("focused_compound_evidence_lines", 0),
        "focused_compound_recent_tool_results_chars": d.get("output", {}).get("focused_compound_recent_tool_results_chars", 0),
        "focused_compound_include_resolved_evidence": d.get("output", {}).get("focused_compound_include_resolved_evidence", False),
        "compound_selective_trace_lines": d.get("output", {}).get("compound_selective_trace_lines", 0),
        "compound_selective_unresolved_evidence_lines": d.get("output", {}).get("compound_selective_unresolved_evidence_lines", 0),
        "compound_selective_resolved_evidence_lines": d.get("output", {}).get("compound_selective_resolved_evidence_lines", 0),
        "compound_selective_resolved_evidence_stub_chars": d.get("output", {}).get("compound_selective_resolved_evidence_stub_chars", 0),
        "compound_selective_recent_tool_results_chars": d.get("output", {}).get("compound_selective_recent_tool_results_chars", 0),
        "compound_selective_trace_action_repeat_cap": d.get("output", {}).get("compound_selective_trace_action_repeat_cap", 0),
        "compound_selective_resolved_action_repeat_cap": d.get("output", {}).get("compound_selective_resolved_action_repeat_cap", 0),
        "compound_selective_trace_anchor_lines": d.get("output", {}).get("compound_selective_trace_anchor_lines", 0),
        "compound_selective_resolved_anchor_lines": d.get("output", {}).get("compound_selective_resolved_anchor_lines", 0),
        "compound_selective_trace_source_anchor_lines": d.get("output", {}).get("compound_selective_trace_source_anchor_lines", 0),
        "compound_selective_trace_test_anchor_lines": d.get("output", {}).get("compound_selective_trace_test_anchor_lines", 0),
        "compound_selective_resolved_source_anchor_lines": d.get("output", {}).get("compound_selective_resolved_source_anchor_lines", 0),
        "compound_selective_resolved_test_anchor_lines": d.get("output", {}).get("compound_selective_resolved_test_anchor_lines", 0),
        "pretest_head_chars": _require(d, "output", "pretest_head_chars"),
        "pretest_tail_chars": _require(d, "output", "pretest_tail_chars"),
        "bash_timeout": _require(d, "tools", "bash_timeout"),
        "grep_timeout": _require(d, "tools", "grep_timeout"),
        "search_pagination_enabled": d.get("tools", {}).get("search_pagination_enabled", True),
        "grep_max_matches_per_page": d.get("tools", {}).get("grep_max_matches_per_page", 25),
        "glob_max_matches_per_page": d.get("tools", {}).get("glob_max_matches_per_page", 25),
        "tools_glob_max_listed_paths": d.get("tools", {}).get("glob_max_listed_paths", 50),
        "tools_glob_refuse_unscoped_recursive": d.get("tools", {}).get("glob_refuse_unscoped_recursive", True),
        "bash_quirks_forbidden_enabled": d.get("loop", {}).get("bash_quirks_forbidden_enabled", True),
        "pre_mutation_turn_cap": d.get("loop", {}).get("pre_mutation_turn_cap", 0),
        "plan_mode": d.get("loop", {}).get("plan_mode", "off"),
        "plan_mode_max_turns": d.get("loop", {}).get("plan_mode_max_turns", 15),
        "digest_compaction_safety_margin": d.get("context", {}).get("digest_compaction_safety_margin", 0.05),
        "digest_keep_recent_turns": d.get("context", {}).get("digest_keep_recent_turns", 8),
        "digest_compaction_gate_min_mutations": d.get("context", {}).get("digest_compaction_gate_min_mutations", 0),
        "repo_map_tokens": d.get("context", {}).get("repo_map_tokens", 0),
        "repo_map_refresh": d.get("context", {}).get("repo_map_refresh", "auto"),
        "compaction_method": d.get("context", {}).get("compaction_method", "digest"),
        "compaction_hook": d.get("context", {}).get("compaction_hook", ""),
        "checkpoint_keep_recent_tokens": d.get("context", {}).get("checkpoint_keep_recent_tokens", 0),
        "checkpoint_max_summary_tokens": d.get("context", {}).get("checkpoint_max_summary_tokens", 4000),
        "handoff_summary_enabled": d.get("loop", {}).get("handoff_summary_enabled", False),
        "handoff_max_tokens": d.get("prompts", {}).get("handoff_max_tokens", 2000),
        "edit_strict_match": d.get("tools", {}).get("edit_strict_match", True),
        "edit_fuzzy_cascade_enabled": d.get("tools", {}).get("edit_fuzzy_cascade_enabled", False),
        "edit_candidate_count": d.get("tools", {}).get("edit_candidate_count", 3),
        "pretest_timeout": _require(d, "tools", "pretest_timeout"),
        "llama_server_bin": _require(d, "tools", "llama_server_bin"),
        "sandbox_bash": _require(d, "tools", "sandbox_bash"),
        "strip_ansi": _require(d, "tools", "strip_ansi"),
        "collapse_blank_lines": _require(d, "tools", "collapse_blank_lines"),
        "collapse_duplicate_lines": _require(d, "tools", "collapse_duplicate_lines"),
        "collapse_similar_lines": _require(d, "tools", "collapse_similar_lines"),
        "bwrap_bin": _require(d, "tools", "bwrap_bin"),
        "sandbox_required": d.get("tools", {}).get("sandbox_required", False),
        "unreadable_paths": _string_tuple(
            d.get("sandbox", {}).get("unreadable_paths", []) or (),
            path="sandbox.unreadable_paths",
        ),
        "sandbox_backend": d.get("sandbox", {}).get("backend", "bwrap"),
        "sandbox_container_runtime": d.get("sandbox", {}).get(
            "container_runtime", "docker"
        ),
        "sandbox_container_image": d.get("sandbox", {}).get(
            "container_image", ""
        ),
        "sandbox_container_flags": _string_tuple(
            d.get("sandbox", {}).get("container_flags", []),
            path="sandbox.container_flags",
        ),
        "sandbox_env_inherit": sandbox_env.inherit,
        "sandbox_env_set": dict(sandbox_env.set),
        "sandbox_env_filters": dict(sandbox_env.filters),
        "sandbox_env_ignore_default_excludes": (
            sandbox_env.ignore_default_excludes
        ),
        "sandbox_env_allow_login_shell": sandbox_env.allow_login_shell,
        "runtime_worktree": d.get("runtime", {}).get("worktree", "off"),
        "hooks_enabled": hooks_section.get("enabled", False),
        "hooks": {
            name: copy.deepcopy(value)
            for name, value in hooks_section.items()
            if name != "enabled"
        },
        "max_transient_retries": _require(d, "loop", "max_transient_retries"),
        "retry_backoff": _integer_tuple(
            _require(d, "loop", "retry_backoff"),
            path="loop.retry_backoff",
        ),
        "interrupted_turn_mode": d.get("loop", {}).get(
            "interrupted_turn_mode", "mechanical"
        ),
        "length_continue_max": d.get("loop", {}).get(
            "length_continue_max", 0
        ),
        "project_docs_enabled": d.get("prompts", {}).get(
            "project_docs_enabled", False
        ),
        "project_doc_names": _string_tuple(
            d.get("prompts", {}).get(
                "project_doc_names", ["AGENTS.md", "CLAUDE.md"]
            ),
            path="prompts.project_doc_names",
        ),
        "project_doc_max_bytes": d.get("prompts", {}).get(
            "project_doc_max_bytes", 32768
        ),
        "project_root_markers": _string_tuple(
            d.get("prompts", {}).get(
                "project_root_markers", [".git", ".hg", ".sl"]
            ),
            path="prompts.project_root_markers",
        ),
        "project_doc_global_dir": d.get("prompts", {}).get(
            "project_doc_global_dir", "~/.config/yuj"
        ),
        "imports_enabled": d.get("prompts", {}).get(
            "imports_enabled", True
        ),
        "imports_max_depth": d.get("prompts", {}).get(
            "imports_max_depth", 5
        ),
        "skills_enabled": d.get("prompts", {}).get(
            "skills_enabled", False
        ),
        "skills_dirs": _string_tuple(
            d.get("prompts", {}).get(
                "skills_dirs",
                [
                    "~/.pi/agent/skills",
                    "~/.agents/skills",
                    ".pi/skills",
                    ".agents/skills",
                ],
            ),
            path="prompts.skills_dirs",
        ),
        "skill_paths": _string_tuple(
            d.get("prompts", {}).get("skill_paths", []),
            path="prompts.skill_paths",
        ),
        "system_header": _require(d, "prompts", "system_header"),
        "state_context_suffix": _require(d, "prompts", "state_context_suffix"),
        "intent_gate_first": _require(d, "prompts", "intent_gate_first"),
        "intent_gate_repeat": _require(d, "prompts", "intent_gate_repeat"),
        "resume_base": _require(d, "prompts", "resume_base"),
        "error_nudge": _require(d, "prompts", "error_nudge"),
        "rumination_nudge": _require(d, "prompts", "rumination_nudge"),
        "rumination_gate": _require(d, "prompts", "rumination_gate"),
        "rumination_gate_grace_prefix": d.get(
            "prompts", {}
        ).get(
            "rumination_gate_grace_prefix",
            "[HARNESS: Gate armed. Next call must mutate a file — all else blocked.]",
        ),
        "pre_mutation_gate": d.get(
            "prompts", {}
        ).get(
            "pre_mutation_gate",
            "[HARNESS: {turn_number} read-only turns elapsed without a file mutation; the next tool call must use the selected file-edit tool, run a bash command that mutates a source file, or call done(). This call was not executed.]",
        ),
        "rumination_same_target_nudge": d.get(
            "prompts", {}
        ).get(
            "rumination_same_target_nudge",
            "[HARNESS: same target hit {count} times without a file mutation ({target}). Stop rereading it; mutate, verify, or move to a different target.]",
        ),
        "rumination_outside_cwd_nudge": d.get(
            "prompts", {}
        ).get(
            "rumination_outside_cwd_nudge",
            "[HARNESS: repeated inspection is anchored outside the repo root ({target}). The working directory is already correct; search and read relative to it.]",
        ),
        "test_read_nudge": d.get(
            "prompts", {}
        ).get(
            "test_read_nudge",
            "[HARNESS: ran verification {count} time(s) without reading the target test file ({target}). Read the test before more checks.]",
        ),
        "contract_commit_warn": d.get(
            "prompts", {}
        ).get(
            "contract_commit_warn",
            "[HARNESS: source file {source} is already in view. Choose a concrete next move: mutate a file, read a test file, or run verification. Do not continue broad inspection.]",
        ),
        "contract_commit_block": d.get(
            "prompts", {}
        ).get(
            "contract_commit_block",
            "[HARNESS: commit contract active from {source}. This tool call was not executed. Allowed next moves: mutate a file, read a test file, or run verification.]",
        ),
        "contract_recovery_block": d.get(
            "prompts", {}
        ).get(
            "contract_recovery_block",
            "[HARNESS: recovery mode for {reason} ({target}). This tool call was not executed. Allowed next moves: read a concrete file, mutate a file, or run verification.]",
        ),
        "mutation_repeat_warn": d.get(
            "prompts", {}
        ).get(
            "mutation_repeat_warn",
            "[HARNESS: the same mutation was already applied to {target}. Do not repeat it unchanged; read new evidence, run verification, or change the mutation.]",
        ),
        "mutation_repeat_block": d.get(
            "prompts", {}
        ).get(
            "mutation_repeat_block",
            "[HARNESS: repeated identical mutation on {target}. This tool call was not executed. Read new evidence, run verification, or change the mutation.]",
        ),
        "read_truncated_reminder": d.get("prompts", {}).get(
            "read_truncated_reminder",
            "<system-reminder>Read returned the first {returned_lines} lines of {path}. The file is longer — re-read with a higher limit or a specific offset to see more.</system-reminder>",
        ),
        "read_empty_reminder": d.get("prompts", {}).get(
            "read_empty_reminder",
            "<system-reminder>File {path} exists but is empty (0 bytes).</system-reminder>",
        ),
        "sink_pointer": d.get("prompts", {}).get(
            "sink_pointer",
            '<tool_result_meta truncated="true" original_bytes="{chars}" original_lines="{lines}" full_path="{path}"/>',
        ),
        "sink_body_marker": d.get("prompts", {}).get(
            "sink_body_marker",
            "... [body truncated — full output available via full_path attribute] ...",
        ),
        "sink_head_bytes": d.get("tools", {}).get("sink_head_bytes", 1000),
        "sink_tail_bytes": d.get("tools", {}).get("sink_tail_bytes", 1000),
        "resume_duplicate_abort": _require(d, "prompts", "resume_duplicate_abort"),
        "resume_context_full": _require(d, "prompts", "resume_context_full"),
        "resume_max_turns": _require(d, "prompts", "resume_max_turns"),
        "resume_length": _require(d, "prompts", "resume_length"),
        "resume_gate_escalation": d.get("prompts", {}).get("resume_gate_escalation", "Session ended: rumination gate blocked {n} consecutive calls. Your current code has been preserved."),
        "resume_last_n_actions": _require(d, "prompts", "resume_last_n_actions"),
        "tool_desc": experiment.get("tool_desc", "minimal"),
        "prompt_addendum": experiment.get("prompt_addendum", ""),
        "variant_name": experiment.get("variant_name", ""),
        "runtime_mode": d.get("runtime", {}).get("mode", "measurement"),
        "security_scan_mode": _require(d, "security", "scan_mode"),
        "security_patterns_file": _require(d, "security", "patterns_file"),
        "security_block_classes": _string_tuple(
            _require(d, "security", "block_classes"),
            path="security.block_classes",
        ),
        "permissions_rules": copy.deepcopy(
            d.get("permissions", {}).get("rules", {})
        ),
        "permissions_ask_fallback": d.get("permissions", {}).get(
            "ask_fallback", "deny"
        ),
        "analysis_task_format": analysis.get("task_format", "auto"),
    }


def _validate_coupling(cfg: Config, strict_dial_gates: bool = False,
                       user_set_keys: frozenset[str] = frozenset()) -> None:
    """Reject config combinations that produce silent fallthrough.

    Bucket B toggles have coupling constraints (one feature's effective
    behaviour depends on another being enabled). Without validation the
    model sees an unhelpful fallback — e.g. structured output enabled
    but task-format control disabled produces no parser, no digest, and
    no error. The run can then use the wrong behavior without warning.

    Rules:
      - bash_transforms_structured_output_enabled requires
        bash_transforms_task_format_enabled — the parser is loaded
        through the task-format path.
      - done_require_pretest_parity only meaningfully activates when
        bash_transforms_structured_output_enabled is also on. Parity
        falls back to the heuristic otherwise; a warning makes the
        silent downgrade visible.
    """
    if not isinstance(cfg.provider, str) or cfg.provider not in {
        "openai-compatible",
        "anthropic",
    }:
        raise ValueError(
            "config error: server.provider must be 'openai-compatible' or "
            "'anthropic'."
        )
    for field_name, value in (
        ("loop.max_turns", cfg.max_turns),
        ("loop.max_sessions", cfg.max_sessions),
        ("model.context_size", cfg.context_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"config error: {field_name} must be an integer >= 1."
            )
    for field_name, value in (
        ("model.context_fill_ratio", cfg.context_fill_ratio),
        ("model.max_tokens_fraction", cfg.max_tokens_fraction),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < float(value) <= 1
        ):
            raise ValueError(
                f"config error: {field_name} must be a finite number in (0, 1]."
            )
    if (
        isinstance(cfg.max_transient_retries, bool)
        or not isinstance(cfg.max_transient_retries, int)
        or cfg.max_transient_retries < 0
    ):
        raise ValueError(
            "config error: loop.max_transient_retries must be a non-negative "
            "integer."
        )
    if any(
        isinstance(delay, bool) or not isinstance(delay, int) or delay < 0
        for delay in cfg.retry_backoff
    ):
        raise ValueError(
            "config error: loop.retry_backoff must contain only non-negative "
            "integers."
        )

    from .server.request_controls import (
        normalize_cache_affinity,
        normalize_cache_retention,
        normalize_thinking_level,
        validate_cache_miss_warn_ratio,
        validate_request_extra,
    )

    validate_request_extra(
        cfg.server_request_extra, path="server.request_extra"
    )
    normalize_cache_affinity(cfg.cache_affinity)
    normalize_cache_retention(cfg.cache_retention)
    validate_cache_miss_warn_ratio(cfg.cache_miss_warn_ratio)
    normalize_thinking_level(cfg.thinking_level)
    if cfg.plan_mode not in {"off", "required"}:
        raise ValueError(
            "config error: loop.plan_mode must be 'off' or 'required', "
            f"got {cfg.plan_mode!r}."
        )
    if (
        isinstance(cfg.plan_mode_max_turns, bool)
        or not isinstance(cfg.plan_mode_max_turns, int)
        or cfg.plan_mode_max_turns < 1
    ):
        raise ValueError(
            "config error: loop.plan_mode_max_turns must be an integer >= 1."
        )
    from ._shared.edit_formats import validate_edit_format
    validate_edit_format(
        cfg.tools_edit_format,
        field="config error: tools.edit_format",
        allow_inherit=True,
    )
    if cfg.effective_edit_format:
        validate_edit_format(
            cfg.effective_edit_format,
            field="config error: effective_edit_format",
        )
    if cfg.tools_stale_guard_mode not in {"off", "warn", "block"}:
        raise ValueError(
            "config error: tools.stale_guard_mode must be 'off', 'warn', "
            f"or 'block', got {cfg.tools_stale_guard_mode!r}."
        )
    if not isinstance(cfg.tools_bash_redirect_read_side, bool):
        raise ValueError(
            "config error: tools.bash_redirect_read_side must be a boolean."
        )
    from .harness.tool_validation import (
        normalize_constrained_decoding_mode,
        normalize_schema_validation_mode,
    )
    normalize_schema_validation_mode(cfg.tools_schema_validation)
    normalize_constrained_decoding_mode(cfg.tools_constrained_decoding)
    from .harness.security_scan import validate_security_settings
    validate_security_settings(
        cfg.security_scan_mode,
        cfg.security_patterns_file,
        cfg.security_block_classes,
    )
    if not isinstance(cfg.tools_think_enabled, bool):
        raise ValueError(
            "config error: tools.think_enabled must be a boolean."
        )
    if (
        isinstance(cfg.tools_think_keep_turns, bool)
        or not isinstance(cfg.tools_think_keep_turns, int)
        or cfg.tools_think_keep_turns < 0
    ):
        raise ValueError(
            "config error: tools.think_keep_turns must be a non-negative "
            "integer."
        )
    if (
        isinstance(cfg.think_streak_nudge_after, bool)
        or not isinstance(cfg.think_streak_nudge_after, int)
        or cfg.think_streak_nudge_after < 0
    ):
        raise ValueError(
            "config error: loop.think_streak_nudge_after must be a "
            "non-negative integer."
        )
    if not isinstance(cfg.tools_todos_enabled, bool):
        raise ValueError(
            "config error: tools.todos_enabled must be a boolean."
        )
    if (
        isinstance(cfg.tools_todos_max_items, bool)
        or not isinstance(cfg.tools_todos_max_items, int)
        or cfg.tools_todos_max_items < 1
    ):
        raise ValueError(
            "config error: tools.todos_max_items must be an integer >= 1."
        )
    if (
        isinstance(cfg.state_todos_char_budget, bool)
        or not isinstance(cfg.state_todos_char_budget, int)
        or cfg.state_todos_char_budget < 1
    ):
        raise ValueError(
            "config error: state.todos_char_budget must be an integer >= 1."
        )
    if not isinstance(cfg.tools_lazy_loading_enabled, bool):
        raise ValueError(
            "config error: tools.lazy_loading_enabled must be a boolean."
        )
    if any(not name.strip() for name in cfg.tools_active_default):
        raise ValueError(
            "config error: tools.active_default entries must be non-empty strings."
        )
    if len(cfg.tools_active_default) != len(set(cfg.tools_active_default)):
        raise ValueError(
            "config error: tools.active_default must not contain duplicates."
        )
    from .harness.tool_specs import ACTIVE_TOOL_NAMES
    unknown_active_tools = sorted(
        set(cfg.tools_active_default) - set(ACTIVE_TOOL_NAMES)
    )
    if unknown_active_tools:
        raise ValueError(
            "config error: tools.active_default contains unknown tool names: "
            + ", ".join(unknown_active_tools)
        )
    if not isinstance(cfg.tools_checkpoint_enabled, bool):
        raise ValueError(
            "config error: tools.checkpoint_enabled must be a boolean."
        )
    from .harness.tool_policy import (
        PermissionPolicy,
        normalize_ask_fallback,
    )
    PermissionPolicy.from_rule_tables(cfg.permissions_rules)
    normalize_ask_fallback(cfg.permissions_ask_fallback)
    from .harness.hooks import validate_hook_settings

    validate_hook_settings(cfg.hooks_enabled, cfg.hooks)
    if not isinstance(cfg.tools_background_enabled, bool):
        raise ValueError(
            "config error: tools.background_enabled must be a boolean."
        )
    if (
        isinstance(cfg.tools_background_max_procs, bool)
        or not isinstance(cfg.tools_background_max_procs, int)
        or cfg.tools_background_max_procs < 1
    ):
        raise ValueError(
            "config error: tools.background_max_procs must be an integer >= 1."
        )
    if (
        isinstance(cfg.tools_background_poll_timeout, bool)
        or not isinstance(cfg.tools_background_poll_timeout, (int, float))
        or not math.isfinite(float(cfg.tools_background_poll_timeout))
        or float(cfg.tools_background_poll_timeout) < 0
    ):
        raise ValueError(
            "config error: tools.background_poll_timeout must be a finite "
            "number >= 0."
        )
    if not isinstance(cfg.tools_task_enabled, bool):
        raise ValueError(
            "config error: tools.task_enabled must be a boolean."
        )
    if (
        isinstance(cfg.tools_subagent_depth, bool)
        or not isinstance(cfg.tools_subagent_depth, int)
        or cfg.tools_subagent_depth < 0
    ):
        raise ValueError(
            "config error: tools.subagent_depth must be a non-negative "
            "integer."
        )
    if (
        isinstance(cfg.tools_subagent_max_turns, bool)
        or not isinstance(cfg.tools_subagent_max_turns, int)
        or cfg.tools_subagent_max_turns < 1
    ):
        raise ValueError(
            "config error: tools.subagent_max_turns must be an integer >= 1."
        )
    if not isinstance(cfg.tools_exec_cell_enabled, bool):
        raise ValueError(
            "config error: tools.exec_cell_enabled must be a boolean."
        )
    if cfg.tools_lazy_loading_enabled and cfg.tools_exec_cell_enabled:
        raise ValueError(
            "config error: tools.lazy_loading_enabled and "
            "tools.exec_cell_enabled cannot be enabled together."
        )
    if (
        isinstance(cfg.tools_exec_cell_timeout, bool)
        or not isinstance(cfg.tools_exec_cell_timeout, int)
        or cfg.tools_exec_cell_timeout < 1
    ):
        raise ValueError(
            "config error: tools.exec_cell_timeout must be an integer >= 1."
        )
    if cfg.sandbox_backend not in {"bwrap", "container"}:
        raise ValueError(
            "config error: sandbox.backend must be 'bwrap' or 'container', "
            f"got {cfg.sandbox_backend!r}."
        )
    if not isinstance(cfg.runtime_worktree, str) or not cfg.runtime_worktree.strip():
        raise ValueError(
            "config error: runtime.worktree must be 'off', 'auto', or a "
            "non-empty Git branch name."
        )
    if cfg.interrupted_turn_mode not in {"off", "mechanical"}:
        raise ValueError(
            "config error: loop.interrupted_turn_mode must be 'off' or "
            f"'mechanical', got {cfg.interrupted_turn_mode!r}."
        )
    if (
        isinstance(cfg.length_continue_max, bool)
        or not isinstance(cfg.length_continue_max, int)
        or cfg.length_continue_max < 0
    ):
        raise ValueError(
            "config error: loop.length_continue_max must be a non-negative "
            "integer."
        )
    if not isinstance(cfg.stream_rules_enabled, bool):
        raise ValueError(
            "config error: loop.stream_rules_enabled must be a boolean."
        )
    if not isinstance(cfg.stream_rules_dir, str) or not cfg.stream_rules_dir.strip():
        raise ValueError(
            "config error: loop.stream_rules_dir must be a non-empty relative path."
        )
    stream_rules_path = Path(cfg.stream_rules_dir)
    if stream_rules_path.is_absolute() or ".." in stream_rules_path.parts:
        raise ValueError(
            "config error: loop.stream_rules_dir must stay inside the task "
            "repository (absolute paths and '..' are not allowed)."
        )
    if cfg.stream_rules_context_mode not in {"discard", "keep"}:
        raise ValueError(
            "config error: loop.stream_rules_context_mode must be 'discard' "
            f"or 'keep', got {cfg.stream_rules_context_mode!r}."
        )
    if (
        isinstance(cfg.stream_rules_repeat_gap, bool)
        or not isinstance(cfg.stream_rules_repeat_gap, int)
        or cfg.stream_rules_repeat_gap < 1
    ):
        raise ValueError(
            "config error: loop.stream_rules_repeat_gap must be an integer >= 1."
        )
    if not isinstance(cfg.project_docs_enabled, bool):
        raise ValueError(
            "config error: prompts.project_docs_enabled must be a boolean."
        )
    if not isinstance(cfg.state_ignore_file_enabled, bool):
        raise ValueError(
            "config error: state.ignore_file_enabled must be a boolean."
        )
    from .harness.sandbox.ignore_policy import validate_ignore_file_names
    validate_ignore_file_names(cfg.state_ignore_file_names)
    if not isinstance(cfg.project_doc_global_dir, str):
        raise ValueError(
            "config error: prompts.project_doc_global_dir must be a string."
        )
    if not isinstance(cfg.imports_enabled, bool):
        raise ValueError(
            "config error: prompts.imports_enabled must be a boolean."
        )
    if (
        isinstance(cfg.imports_max_depth, bool)
        or not isinstance(cfg.imports_max_depth, int)
        or cfg.imports_max_depth < 0
    ):
        raise ValueError(
            "config error: prompts.imports_max_depth must be a non-negative "
            "integer."
        )
    from .harness.skills import validate_skill_settings
    try:
        validate_skill_settings(
            cfg.skills_enabled,
            cfg.skills_dirs,
            cfg.skill_paths,
        )
    except ValueError as exc:
        raise ValueError(f"config error: prompts.{exc}") from exc
    if not isinstance(cfg.injections_enabled, bool):
        raise ValueError(
            "config error: injections.enabled must be a boolean."
        )
    if not isinstance(cfg.injections_dir, str) or not cfg.injections_dir.strip():
        raise ValueError(
            "config error: injections.dir must be a non-empty string."
        )
    if not isinstance(cfg.injections_path_rules_enabled, bool):
        raise ValueError(
            "config error: injections.path_rules_enabled must be a boolean."
        )
    if not isinstance(cfg.injections_path_rule_repeat, bool):
        raise ValueError(
            "config error: injections.path_rule_repeat must be a boolean."
        )
    if cfg.injections_path_rules_enabled and not cfg.injections_enabled:
        raise ValueError(
            "config error: injections.path_rules_enabled requires "
            "injections.enabled=true."
        )
    from .harness.project_instructions import (
        validate_project_instruction_settings,
    )
    try:
        validate_project_instruction_settings(
            cfg.project_doc_names,
            cfg.project_doc_max_bytes,
            cfg.project_root_markers,
        )
    except ValueError as exc:
        raise ValueError(f"config error: prompts.{exc}") from exc
    from .harness.sandbox.container_backend import (
        CONTAINER_RUNTIMES,
        ContainerBackend,
        normalize_container_flags,
    )
    if cfg.sandbox_container_runtime not in CONTAINER_RUNTIMES:
        raise ValueError(
            "config error: sandbox.container_runtime must be 'docker' or "
            f"'podman', got {cfg.sandbox_container_runtime!r}."
        )
    normalize_container_flags(cfg.sandbox_container_flags)
    from .harness.sandbox.env_policy import (
        EnvironmentPolicy,
        EnvironmentPolicyError,
    )
    try:
        EnvironmentPolicy(
            inherit=cfg.sandbox_env_inherit,
            set=cfg.sandbox_env_set,
            filters=cfg.sandbox_env_filters,
            ignore_default_excludes=cfg.sandbox_env_ignore_default_excludes,
            allow_login_shell=cfg.sandbox_env_allow_login_shell,
        )
    except EnvironmentPolicyError as exc:
        raise ValueError(f"config error: {exc}") from exc
    if cfg.sandbox_backend == "container":
        ContainerBackend(
            runtime=cfg.sandbox_container_runtime,
            image=cfg.sandbox_container_image,
            flags=cfg.sandbox_container_flags,
        )
    if (
        isinstance(cfg.tools_ast_search_max_rows, bool)
        or not isinstance(cfg.tools_ast_search_max_rows, int)
        or cfg.tools_ast_search_max_rows < 1
    ):
        raise ValueError(
            "config error: tools.ast_search_max_rows must be an integer >= 1."
        )
    if isinstance(cfg.lsp_diagnostics_timeout_s, bool) or not isinstance(
        cfg.lsp_diagnostics_timeout_s, (int, float)
    ) or not math.isfinite(float(cfg.lsp_diagnostics_timeout_s)) or float(
        cfg.lsp_diagnostics_timeout_s
    ) < 0:
        raise ValueError(
            "config error: lsp.diagnostics_timeout_s must be a finite "
            "non-negative number."
        )
    from .harness.lsp_support import LspManager, parse_server_specs
    parse_server_specs(cfg.lsp_servers)
    # Constructing a disabled manager validates the public severity value
    # without starting a process or touching the task filesystem.
    LspManager(
        cwd=Path.cwd(), servers=(), argv_builder=lambda _spec, _root: (),
        diagnostics_timeout_s=float(cfg.lsp_diagnostics_timeout_s),
        min_severity=cfg.lsp_min_severity, enabled=False,
    )
    from .harness._loop.model_roles import (
        normalize_fallback_revert,
        validate_fallback_chains,
        validate_role_specs,
    )
    validate_role_specs(cfg.model_roles)
    validate_fallback_chains(cfg.model_fallback_chain)
    normalize_fallback_revert(cfg.model_fallback_revert)
    if not isinstance(cfg.advisor_enabled, bool):
        raise ValueError("config error: advisor.enabled must be a boolean.")
    if not isinstance(cfg.advisor_model, str):
        raise ValueError("config error: advisor.model must be a string.")
    if not isinstance(cfg.advisor_endpoint, str):
        raise ValueError("config error: advisor.endpoint must be a string.")
    for field_name, value, minimum in (
        ("every_n_turns", cfg.advisor_every_n_turns, 1),
        ("immune_turns", cfg.advisor_immune_turns, 0),
        ("max_note_chars", cfg.advisor_max_note_chars, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(
                f"config error: advisor.{field_name} must be an integer >= {minimum}."
            )
    if cfg.advisor_enabled or cfg.advisor_model or cfg.advisor_endpoint:
        from .harness._loop.model_roles import parse_model_target

        target: dict[str, object] = {
            "profile": cfg.profile_name or cfg.model,
        }
        if cfg.advisor_endpoint:
            target["endpoint"] = cfg.advisor_endpoint
        if cfg.advisor_model:
            target["model"] = cfg.advisor_model
        parse_model_target(target, field="advisor")
    if (
        isinstance(cfg.rewind_max_per_session, bool)
        or not isinstance(cfg.rewind_max_per_session, int)
        or cfg.rewind_max_per_session < 1
    ):
        raise ValueError(
            "config error: loop.rewind_max_per_session must be an integer >= 1."
        )
    if cfg.rewind_enabled and not cfg.tools_file_checkpoints_enabled:
        raise ValueError(
            "config error: loop.rewind_enabled requires "
            "tools.file_checkpoints_enabled = true."
        )
    if (
        isinstance(cfg.repo_map_tokens, bool)
        or not isinstance(cfg.repo_map_tokens, int)
        or cfg.repo_map_tokens < 0
    ):
        raise ValueError(
            "config error: context.repo_map_tokens must be a non-negative "
            "integer."
        )
    from .harness.repo_map import normalize_repo_map_refresh
    normalize_repo_map_refresh(cfg.repo_map_refresh)
    from .harness.compaction_hooks import resolve_compaction_hook
    resolve_compaction_hook(cfg.compaction_hook)
    if cfg.compaction_method not in {"digest", "checkpoint"}:
        raise ValueError(
            "config error: context.compaction_method must be 'digest' or "
            f"'checkpoint', got {cfg.compaction_method!r}."
        )
    if cfg.checkpoint_keep_recent_tokens < 0:
        raise ValueError(
            "config error: context.checkpoint_keep_recent_tokens must be "
            "zero (auto) or a positive integer."
        )
    if cfg.checkpoint_max_summary_tokens <= 0:
        raise ValueError(
            "config error: context.checkpoint_max_summary_tokens must be positive."
        )
    if cfg.handoff_summary_enabled and cfg.handoff_max_tokens <= 0:
        raise ValueError(
            "config error: prompts.handoff_max_tokens must be positive when "
            "loop.handoff_summary_enabled is true."
        )
    if (cfg.bash_transforms_structured_output_enabled
            and not cfg.bash_transforms_task_format_enabled):
        raise ValueError(
            "config error: bash_transforms_structured_output_enabled = true "
            "requires bash_transforms_task_format_enabled = true — the "
            "structured parser is loaded via the task-format path."
        )
    # Dial-requires-gate rules: setting a family's dial while its enable
    # gate is off produces a silently inert config.
    _DIAL_GATES = (
        ("duplicate_warn_count", "duplicate_guard_enabled"),
        ("error_nudge_threshold", "error_ladder_enabled"),
        ("error_abort_threshold", "error_ladder_enabled"),
        ("error_same_class_threshold", "error_ladder_enabled"),
        ("intent_grace_turns", "require_intent"),
        ("rumination_same_target_warn_after", "rumination_enabled"),
    )
    for dial, gate in _DIAL_GATES:
        # only when a USER layer explicitly set the dial: repo defaults keep
        # dormant dial values (e.g. intent_grace_turns = 3) by convention.
        if dial not in user_set_keys:
            continue
        if getattr(cfg, dial, 0) and not getattr(cfg, gate, False):
            msg = (f"{dial} is set but {gate} = false — the dial is inert "
                   f"without its gate; set {gate} = true or unset {dial}.")
            if strict_dial_gates:
                raise ValueError(f"config error: {msg}")
            logging.getLogger(__name__).warning("coupling: %s", msg)
    if (cfg.done_require_pretest_parity
            and not cfg.bash_transforms_structured_output_enabled):
        log = logging.getLogger(__name__)
        log.warning(
            "done_require_pretest_parity is on but structured output is "
            "disabled; parity will fall back to the heuristic preconditions "
            "(done_require_mutation / done_require_verify). Enable "
            "bash_transforms_structured_output_enabled for ground-truth parity."
        )
    if (cfg.adaptive_policy_enabled
            and cfg.adaptive_phase2_bash_structured_output_enabled
            and not cfg.adaptive_phase2_bash_task_format_enabled):
        raise ValueError(
            "config error: adaptive phase2 structured output requires "
            "adaptive_phase2_bash_task_format_enabled = true."
        )
    # Keep this complete type backstop after path-specific validators so their
    # established, actionable TOML-path diagnostics remain the public errors.
    _validate_config_field_types(cfg)
