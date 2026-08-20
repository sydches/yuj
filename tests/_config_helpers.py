"""Shared test config factory — single source of truth for Config defaults.

Each test file used to inline ~50 Config field defaults in its own
``make_config``. When Config grew a new required field, every test
file had to be updated — drift-prone. This module holds one canonical
default dict; callers get a Config instance via ``make_config(**overrides)``.

Usage from any test module (after the sys.path shim is in place):

    from _config_helpers import make_config
    cfg = make_config(max_turns=10, duplicate_abort=3)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from llm_solver.config import Config  # noqa: E402


def _defaults() -> dict:
    """Canonical test Config defaults. All required Config fields filled."""
    return dict(
        base_url="http://localhost:8080/v1", api_key="local",
        timeout_connect=10, timeout_read=120,
        health_poll_interval=2, health_timeout=2, launch_timeout=120, stop_settle=2,
        model="test-model", profile_name="test-model", context_size=8192, context_fill_ratio=0.85,
        max_tokens_fraction=0.5, max_tokens=4096, tokenizer_id="",
        max_turns=5, max_sessions=2, duplicate_abort=3,
        error_nudge_threshold=3, rumination_nudge_threshold=200, require_intent=False,
        intent_grace_turns=3,
        min_turns_before_context=2,
        max_output_chars=20000, truncate_head_ratio=0.6,
        truncate_head_lines=100, truncate_tail_lines=50,
        args_summary_chars=80, trace_args_summary_chars=200,
        trace_reasoning_store_chars=800,
        trace_result_summary_chars=1200,
        solver_trace_lines=50, solver_evidence_lines=30,
        solver_inference_lines=20,
        recent_tool_results_chars=30000, trace_stub_chars=200,
        trace_reasoning_chars=150,
        pretest_head_chars=2000, pretest_tail_chars=1500,
        bash_timeout=60, grep_timeout=30, pretest_timeout=240,
        llama_server_bin="~/.local/bin/llama-server",
        sandbox_bash=False,
        strip_ansi=True, collapse_blank_lines=True, collapse_duplicate_lines=True,
        collapse_similar_lines=True,
        bwrap_bin="/usr/bin/bwrap",
        # Keep sandbox settings inactive in this fixture.
        sandbox_required=False,
        unreadable_paths=(),
        max_transient_retries=3, retry_backoff=(1, 4, 16),
        system_header=(
            "You are a software engineering solver. Work in the current directory.\n"
            "Use tools to read, write, edit code, search files, and run commands."
        ),
        state_context_suffix=(
            "Continue working. Your progress is tracked in .solver/state.json — "
            "read it to see what you've already done."
        ),
        intent_gate_first=(
            "[harness] Silent tool call rejected. Before each tool call, state in "
            "one sentence what you are doing and why. Re-issue this tool call with "
            "a reasoning prefix."
        ),
        intent_gate_repeat="[intent gate: state your reasoning — {count} silent calls since turn {first_turn}]",
        resume_base="Continue working on the task. Review your previous actions and do the next unit of work.",
        error_nudge="[harness: {count} consecutive errors, consider re-reading the file]",
        rumination_nudge=(
            "[harness: {count} non-write tool calls since your last write/edit. "
            "Your next tool call must be write or edit. Further non-write tool "
            "calls will be rejected by the harness until you make a code change.]"
        ),
        rumination_gate=(
            "[harness gate] This tool call was NOT executed. The harness is "
            "blocking non-write tool calls until you make a write or edit call. "
            "You have enough information to produce code — work from what you "
            "already know. Your next tool call must be write or edit; any other "
            "tool will be rejected with this same message until you make a code "
            "change."
        ),
        rumination_same_target_nudge=(
            "[HARNESS: same target hit {count} times without a write/edit "
            "({target}). Stop rereading it; either edit, verify, or move to a "
            "different target.]"
        ),
        rumination_outside_cwd_nudge=(
            "[HARNESS: repeated inspection is anchored outside the repo root "
            "({target}). The working directory is already correct; search and "
            "read relative to it.]"
        ),
        test_read_nudge=(
            "[HARNESS: ran verification {count} time(s) without reading the "
            "target test file ({target}). Read the test before more checks.]"
        ),
        contract_commit_warn=(
            "[HARNESS: source file {source} is already in view. Choose a "
            "concrete next move: edit/write, read a test file, or run "
            "verification. Do not continue broad inspection.]"
        ),
        contract_commit_block=(
            "[HARNESS: commit contract active from {source}. This tool call "
            "was not executed. Allowed next moves: edit/write, read a test "
            "file, or run verification.]"
        ),
        contract_recovery_block=(
            "[HARNESS: recovery mode for {reason} ({target}). This tool call "
            "was not executed. Allowed next moves: read a concrete file, "
            "edit/write, or run verification.]"
        ),
        mutation_repeat_warn=(
            "[HARNESS: the same mutation was already applied to {target}. Do "
            "not repeat it unchanged; read new evidence, run verification, or "
            "change the mutation.]"
        ),
        mutation_repeat_block=(
            "[HARNESS: repeated identical mutation on {target}. This tool "
            "call was not executed. Read new evidence, run verification, or "
            "change the mutation.]"
        ),
        resume_duplicate_abort=(
            "Last {n} tool calls were identical: {call}. The approach is not "
            "working — try something different."
        ),
        resume_context_full="Context was {pct}% full. This session starts fresh.",
        resume_max_turns=(
            "Last {n} actions: {actions}. Budget exhausted — prioritize "
            "completing the most critical remaining work."
        ),
        resume_length=(
            "Response was truncated by max_tokens. Consider shorter responses or "
            "breaking work into smaller steps."
        ),
        resume_last_n_actions=3,
        tool_desc="minimal",
        prompt_addendum="", variant_name="",
    )


def make_config(**overrides) -> Config:
    """Return a test Config with defaults + optional overrides.

    Optional Config fields (those with dataclass defaults) are NOT
    listed here — the Config dataclass supplies them automatically.
    That means adding a new optional Config field doesn't require
    touching this helper, only a new required field does.
    """
    kwargs = _defaults()
    kwargs.update(overrides)
    return Config(**kwargs)
