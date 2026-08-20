"""Bash output projection (digest + sink-to-disk) and pretest-parity tracking."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..state_writer import write_state_from_events

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)

_NEWLINE = "\n"


def refresh_state(session: "Session") -> None:
    """Rebuild .solver/state.json from the in-memory trace event list.

    No-op if state_path was not provided (wo_yuj arm). The events list
    is kept in sync with the on-disk trace by Session._write_trace, so
    the projection is equivalent to re-reading the file — without the
    O(T) file read + JSON parse per call.
    """
    if session._state_path is None:
        return
    write_state_from_events(
        session._trace_events, session._state_path,
        max_result_chars=session.cfg.max_output_chars,
        imperative_projection=session.cfg.state_imperative_projection_enabled,
    )


def sink_to_disk(session: "Session", raw: str, turn: int) -> str:
    """Write raw bash output to .tool_output/<session>_<counter>.log.

    Returns a one-line pointer to append to the model-visible result,
    or empty string on failure (sink is best-effort; never blocks
    the loop).
    """
    session._sink_counter += 1
    try:
        sink_dir = Path(session.cwd) / ".tool_output"
        sink_dir.mkdir(parents=True, exist_ok=True)
        sink_name = f"{session._session_number}_{session._sink_counter:04d}_t{turn}.log"
        sink_path = sink_dir / sink_name
        # write_bytes skips the codec-negotiation layer that write_text
        # runs on every call; raw is already-decoded subprocess output.
        sink_path.write_bytes(raw.encode("utf-8", errors="replace"))
    except OSError as e:
        log.debug("Sink write failed: %s", e)
        return ""
    rel = sink_path.relative_to(session.cwd)
    return session.cfg.sink_pointer.format(
        path=rel,
        chars=len(raw),
        lines=raw.count(_NEWLINE) + 1,
    )


def project_and_sink(session: "Session", tc_name: str, cmd: str, result: str, turn: int) -> str:
    """Apply structured output projection + sink-and-surface.

    Order:
      1. Structured output (when enabled + parser present + test cmd):
         Parse raw output, render digest, write raw to
         .tool_output/<session>_<sink_counter>.log in cwd, replace
         result with "digest\\n[raw output: <path>, <chars>, <lines>]".
         Preserves the full raw output on disk so the model can read
         it via the existing `read` tool; projects a compact digest
         into the context window.

      2. Sink-only (when result > sink_threshold_chars and step 1
         did not fire): write raw to same location, replace body
         with head/tail head/tail + pointer.

    Applies only to bash results; all other tools return unchanged.
    """
    if tc_name != "bash" or not result:
        return result

    cfg = session.cfg
    projected = False
    pointer_line = ""
    raw_input_chars = len(result)
    if (cfg.bash_transforms_structured_output_enabled
            and session.output_parser is not None
            and session.output_control is not None):
        # Only project for test commands — other bash invocations
        # (build, config, etc.) don't have meaningful structured
        # output and raw is better.
        from ...bash_quirks.transforms import _is_test_command
        if _is_test_command(cmd, session.output_control):
            from ...bash_quirks import parse_structured, render_digest
            parsed = parse_structured(result, session.output_parser)
            digest = render_digest(parsed)
            # Even when the digest is empty (unparseable), the parsed
            # dict still carries whatever per-test records matched.
            # Update the parity-streak state from the parsed record.
            update_parity_from_parsed(session, parsed)
            if digest:
                pointer_line = sink_to_disk(session, result, turn)
                result = digest + ("\n" + pointer_line if pointer_line else "")
                projected = True
                # Token accounting: exact raw-vs-digest delta.
                from ..savings import get_ledger
                get_ledger().record(
                    bucket="structured_projection",
                    layer="L2_bash_quirks",
                    mechanism=f"{cfg.analysis_task_format}_digest",
                    input_chars=raw_input_chars,
                    output_chars=len(result),
                    measure_type="exact",
                    ctx={"cmd": cmd[:120],
                         "n_tests_parsed": len(parsed.get("tests") or {}),
                         "summary": parsed.get("summary")},
                )

    if (not projected
            and cfg.bash_transforms_sink_threshold_chars > 0
            and len(result) > cfg.bash_transforms_sink_threshold_chars):
        pointer_line = sink_to_disk(session, result, turn)
        if pointer_line:
            # Keep a short head+tail preview so the model still sees
            # SOMETHING without needing to open the file. Slice sizes
            # and the body-truncated marker text live in cfg (no
            # prompt literal in harness code).
            head = result[:cfg.sink_head_bytes]
            tail = result[-cfg.sink_tail_bytes:]
            result = (
                f"{head}\n{cfg.sink_body_marker}\n{tail}\n"
                f"{pointer_line}"
            )
            # Token accounting: exact raw-vs-preview delta.
            from ..savings import get_ledger
            get_ledger().record(
                bucket="sink_surface",
                layer="L2_bash_quirks",
                mechanism="head_tail_with_pointer",
                input_chars=raw_input_chars,
                output_chars=len(result),
                measure_type="exact",
                ctx={"cmd": cmd[:120],
                     "threshold": cfg.bash_transforms_sink_threshold_chars},
            )

    return result


def update_parity_from_parsed(session: "Session", parsed: dict) -> None:
    """Process a parsed test run: regression detection + parity update.

    Regression detection runs unconditionally (observability only —
    no gate, no termination). For every test in the prior run's
    verdicts that was PASSED and is now FAILED/ERROR, with at
    least one intervening mutation, a trace event of type
    ``regression`` is written for post-hoc analysis.

    Parity update runs only when ``cfg.done_require_pretest_parity``
    is set AND the pretest baseline was captured. Checks whether
    every pretest-failing test is now PASSED and no pretest-passing
    test has regressed; increments green_parity_streak on match.
    """
    tests = parsed.get("tests") or {}
    if not tests:
        return

    # ── Regression observability (always on when we have a parse) ──
    prev = session._guards.prev_test_parsed
    mutations_between = (
        session._guards.mutation_count - session._guards.mutation_count_at_prev_test
    )
    if prev and mutations_between > 0:
        regressed = [
            tid for tid, prev_v in prev.items()
            if prev_v == "PASSED"
            and tests.get(tid) in ("FAILED", "ERROR")
        ]
        if regressed:
            log.info("Regression detected: %d tests (mutations_between=%d)",
                     len(regressed), mutations_between)
            session._emit(
                "regression",
                session_number=session._session_number,
                tests_regressed=sorted(regressed)[:20],
                n_regressed=len(regressed),
                mutations_between=mutations_between,
            )
    # Update prior-state trackers for the next call.
    session._guards.prev_test_parsed = dict(tests)
    session._guards.mutation_count_at_prev_test = session._guards.mutation_count

    # ── Pretest-parity update (opt-in) ─────────────────────────────
    if not getattr(session.cfg, "done_require_pretest_parity", False):
        return
    if not session._guards.pretest_failing_tests:
        return
    session._guards.latest_test_parsed = dict(tests)
    passed_now = {t for t, v in tests.items() if v in ("PASSED", "PASS")}
    targets_hit = session._guards.pretest_failing_tests.issubset(passed_now)
    regressed_parity = any(
        tests.get(t) not in (None, "PASSED", "PASS")
        for t in session._guards.pretest_passing_tests
    )
    if targets_hit and not regressed_parity:
        session._guards.green_parity_streak += 1
    else:
        session._guards.green_parity_streak = 0
