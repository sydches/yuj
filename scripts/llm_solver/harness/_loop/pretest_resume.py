"""Pretest + resume helpers — extracted from loop.py."""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, TYPE_CHECKING

log = logging.getLogger(__name__)

from .session_io import _sanitize_runner_timing  # noqa: E402

if TYPE_CHECKING:
    from ...config import Config
    from ..loop import Session, SessionResult


def _truncate_pretest_output(output: str, head_chars: int, tail_chars: int) -> str:
    """Head + tail slice with a middle-drop marker. No-op if below limit.

    Reject head_chars + tail_chars <= 0 — Python's `output[-0:]` returns
    the entire string, so a (0, 0) config silently produced a no-op
    truncation instead of an empty string. To disable truncation, set
    both knobs to a value
    larger than any expected output.
    """
    if head_chars < 0 or tail_chars < 0:
        raise ValueError(
            f"_truncate_pretest_output: head_chars and tail_chars must be "
            f"non-negative (got head={head_chars}, tail={tail_chars})"
        )
    if head_chars == 0 and tail_chars == 0:
        raise ValueError(
            "_truncate_pretest_output: head_chars + tail_chars must be > 0; "
            "set a large value to disable truncation"
        )
    limit = head_chars + tail_chars
    if len(output) <= limit:
        return output
    dropped = len(output) - limit
    head = output[:head_chars]
    tail = output[-tail_chars:]
    return f"{head}\n\n... [truncated {dropped} chars] ...\n\n{tail}"


def run_pretest(repo_dir: Path, *, pretest_script: Path | None = None, pretest_timeout: int,
                pretest_head_chars: int, pretest_tail_chars: int, report_cfg=None) -> str:
    """Run an explicitly supplied script through the legacy pretest adapter.

    An absent selection returns no block; neighboring filenames are not input.
    An unavailable explicit selection reports that execution did not start.
    Existing scripts retain the subprocess and output-formatting behavior.
    Explicit selection alone does not establish task permission, test coverage
    or revision-bound result authority; this adapter does not bind those facts.
    """
    if pretest_script is None:
        return ""
    script = Path(pretest_script)
    if not script.is_file():
        return "## Pretest execution\n(pretest not started: declared script unavailable)\n"

    native_report = None
    try:
        from ..test_report import capture_test_report
        with capture_test_report(repo_dir, report_cfg) as capture:
            result = subprocess.run(
                ["bash", str(script.resolve())],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=pretest_timeout,
                **({"env": capture.environment} if capture is not None else {}),
            )
            if capture is not None:
                capture.finish(result.returncode)
                native_report = capture.record
    except subprocess.TimeoutExpired:
        return (
            "## Current test state\n"
            f"(pretest timed out after {pretest_timeout}s — exceeded budget)\n"
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
        return f"## Current test state\n(pretest crashed: {e})\n"

    merged = (result.stdout or "") + (result.stderr or "")
    merged = _sanitize_runner_timing(merged)
    truncated = _truncate_pretest_output(merged, pretest_head_chars, pretest_tail_chars)
    from .._tools._common import ToolExecutionText
    block = ToolExecutionText(
        "## Current test state\n"
        "```\n"
        f"{truncated}\n"
        "```\n\n"
        f"exit code: {result.returncode}\n", exit_status=result.returncode,
    )
    block.native_test_report = native_report
    return block


def _pretest_is_green(block: str) -> bool:
    """Use the current pretest's native status, never a printed exit marker."""
    return (getattr(block, "exit_status", None) == 0
            and not getattr(block, "timed_out", False))


def build_resume_prompt(
    prev_result: SessionResult,
    prev_session: Session,
    cfg: Config,
    task_description: str = "",
) -> str:
    """Build a context-rich resume prompt from the previous session's outcome."""
    parts = []

    if task_description:
        parts.append(f"Task:\n{task_description}")

    # Session summary
    parts.append(
        f"Previous session ended after {prev_result.turns} turns: "
        f"{prev_result.finish_reason}. "
        f"Consumed {prev_result.total_prompt_tokens} prompt tokens."
    )

    reason = prev_result.finish_reason

    if reason == "duplicate_abort":
        calls = prev_session.last_tool_calls
        if calls:
            name, args = calls[-1]
            parts.append(cfg.resume_duplicate_abort.format(
                n=len(calls), call=f"{name}({args})"))

    elif reason == "context_full":
        pct = int(prev_session.context_fill_ratio * 100)
        parts.append(cfg.resume_context_full.format(pct=pct))

    elif reason == "max_turns":
        calls = prev_session.last_tool_calls[-cfg.resume_last_n_actions:]
        if calls:
            summaries = "; ".join(f"{n}({a})" for n, a in calls)
            parts.append(cfg.resume_max_turns.format(
                n=len(calls), actions=summaries))

    elif reason == "gate_escalation":
        parts.append(cfg.resume_gate_escalation.format(n=5))

    elif reason == "length":
        parts.append(cfg.resume_length)

    # Cover finish reasons that need tailored recovery advice.
    elif reason == "done_loop":
        parts.append(cfg.resume_done_loop)
    elif reason == "mutation_repeat_abort":
        parts.append(cfg.resume_mutation_repeat_abort)
    elif reason == "contract_recovery_abort":
        parts.append(cfg.resume_contract_recovery_abort)
    elif reason == "contract_commit_abort":
        parts.append(cfg.resume_contract_commit_abort)
    elif reason == "intent_abort":
        # Use the intent-rejection counter at session end if available.
        n = getattr(prev_session, "_guards", None)
        n = getattr(n, "consecutive_intent_rejections", 1) if n is not None else 1
        parts.append(cfg.resume_intent_abort.format(n=n))
    elif reason == "loop_detected":
        streak = getattr(prev_session, "_guards", None)
        streak = getattr(streak, "loop_detect_streak", 5) if streak is not None else 5
        parts.append(cfg.resume_loop_detect.format(streak=streak))
    elif reason == "no_tool_call":
        parts.append(cfg.resume_no_tool_call)
    elif reason == "error":
        parts.append(cfg.resume_error)
    elif reason == "stop":
        parts.append(cfg.resume_stop)

    parts.append(cfg.resume_base)
    return "\n\n".join(parts)
