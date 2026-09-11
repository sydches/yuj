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


def validate_state_destination(state_path: Path, trace_path: Path, cfg) -> None:
    """Do not adopt an inferred output path based on its filename or schema."""
    import json
    from ..state_writer import project

    if not state_path.parent.is_symlink() and not state_path.is_symlink():
        if not state_path.exists():
            return
        try:
            if not state_path.is_file():
                raise ValueError("state output is not a regular file")
            prior = json.loads(state_path.read_text())
            count = prior.get("meta", {}).get("event_count")
            events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            if type(count) is int and 0 <= count <= len(events):
                for prefix in (events[:count], events):
                    expected = project(
                        prefix, max_result_chars=cfg.max_output_chars,
                        imperative_projection="process" in prior,
                        think_keep_turns=cfg.tools_think_keep_turns,
                    )
                    # Live state can omit trace-only bookkeeping events.
                    # Compare every content field, not that mirror's row count.
                    expected["meta"]["event_count"] = count
                    if prior == expected:
                        return
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    raise ValueError(
        f"Refusing to replace existing state output {state_path}: it is not a "
        "regular projection of the selected trace. Preserve or move that file, "
        "or choose a separate artifact directory."
    )


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
        think_keep_turns=session.cfg.tools_think_keep_turns,
    )


def sink_to_disk(session: "Session", raw: str, turn: int) -> str:
    """Write raw bash output to .tool_output/<session>_<counter>.log.

    Returns a one-line pointer to append to the model-visible result,
    or empty string on failure (sink is best-effort; never blocks
    the loop).
    """
    from ..retained_output import save_output
    rel = save_output(session, raw, turn)
    if not rel:
        return ""
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
         into the context window. If saving fails or yields no pointer,
         preserve the supplied result without a second sink attempt.

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
            # Display parsing remains diagnostic. Only private runner reports
            # may update completion parity.
            update_parity_from_parsed(session, parsed)
            if digest:
                before_projection = result
                pointer_line = sink_to_disk(session, result, turn)
                if not pointer_line:
                    # A short digest cannot replace diagnostics whose saved
                    # copy is unavailable. Preserve this stage's input.
                    return result
                result = digest + "\n" + pointer_line
                projected = True
                # Exact raw-vs-digest transformation record.
                from ..savings import get_ledger
                get_ledger().record_transform(
                    bucket="structured_projection",
                    layer="L2_bash_quirks",
                    mechanism=f"{cfg.analysis_task_format}_digest",
                    before=before_projection,
                    after=result,
                    surface="tool_output",
                    ctx={
                        "n_tests_parsed": len(parsed.get("tests") or {}),
                    },
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
            before_sink = result
            head = before_sink[:cfg.sink_head_bytes]
            tail = before_sink[-cfg.sink_tail_bytes:] if cfg.sink_tail_bytes > 0 else ""
            result = (
                f"{head}\n{cfg.sink_body_marker}\n{tail}\n"
                f"{pointer_line}"
            )
            # Exact raw-vs-preview transformation record.
            from ..savings import get_ledger
            get_ledger().record_transform(
                bucket="sink_surface",
                layer="L2_bash_quirks",
                mechanism="head_tail_with_pointer",
                before=before_sink,
                after=result,
                surface="tool_output",
                ctx={
                    "threshold": cfg.bash_transforms_sink_threshold_chars,
                },
            )

    return result


def update_parity_from_report(session: "Session", report: dict) -> None:
    """Consume a private completed report once, never a model-visible marker."""
    invocation = report.get("invocation")
    if not invocation or invocation == session._guards.last_test_report_invocation:
        return
    session._guards.last_test_report_invocation = invocation
    parsed = report if report.get("status") == "available" else {"tests": {}}
    update_parity_from_parsed(session, parsed, native=True)


def update_parity_from_parsed(session: "Session", parsed: dict, *, native=False) -> None:
    """Process a parsed test run: regression detection + parity update.

    Regression detection runs unconditionally (observability only —
    no gate, no termination). For every test in the prior run's
    verdicts that was PASSED and is now FAILED/ERROR, with at
    least one intervening mutation, a trace event of type
    ``regression`` is written for post-hoc analysis.

    Parity update runs only when ``cfg.done_require_pretest_parity``
    is set, a native report is supplied and the baseline was captured. Checks whether
    every test in both baseline sets is now observed passing; increments
    green_parity_streak on match. Missing results do not establish parity.
    """
    tests = parsed.get("tests") or {}
    parity_enabled = getattr(session.cfg, "done_require_pretest_parity", False)
    baseline = (
        session._guards.pretest_failing_tests | session._guards.pretest_passing_tests
    ) if parity_enabled else set()
    if baseline and native:
        session._guards.latest_test_parsed = dict(tests)
        passed_now = {t for t, v in tests.items() if v in ("PASSED", "PASS")}
        if baseline.issubset(passed_now):
            session._guards.green_parity_streak += 1
        else:
            session._guards.green_parity_streak = 0
    # Keep display-based regression observations separate from report identities.
    if native or not tests:
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
                evidence_source="diagnostic_text",
            )
    # Update prior-state trackers for the next call.
    session._guards.prev_test_parsed = dict(tests)
    session._guards.mutation_count_at_prev_test = session._guards.mutation_count
