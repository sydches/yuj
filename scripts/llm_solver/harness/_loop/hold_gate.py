"""Wait for a caller's signal within the invocation's remaining allowance."""
import math
import os
import time

from ..time_budget import command_time_budget, remaining_run_seconds
from .trace_schema import emit_trace_event

POLL_SECONDS = 0.05


def wait_for_hold(trace_path, log):
    signal = os.environ.get("YUJ_HOLD_UNTIL", "")
    if not signal:
        return
    raw_timeout = os.environ.get("YUJ_HOLD_TIMEOUT_S")
    try:
        timeout = None if raw_timeout is None else float(raw_timeout)
    except ValueError:
        raise ValueError("YUJ_HOLD_TIMEOUT_S must be finite and nonnegative") from None
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("YUJ_HOLD_TIMEOUT_S must be finite and nonnegative")

    started = time.monotonic()
    with command_time_budget(timeout or 0) as shared:
        deadline = shared.deadline
        record = {**shared.record, "hold_limit_seconds": timeout}
        # Include allocation work in an explicit hold ceiling. Zero permits
        # only an immediate signal check, unlike a command's zero cap.
        if timeout is not None:
            hold_deadline = started + timeout
            if deadline is None or hold_deadline <= deadline:
                deadline = hold_deadline
                record["limiting_source"] = "hold_limit"
            record["effective_seconds"] = max(0.0, deadline - started)
            if timeout == 0 and record["status"] != "exhausted":
                record["status"] = "no_wait"
        if record["limiting_source"] == "per_call_limit":
            record["limiting_source"] = "hold_limit"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a") as trace:
            def emit(status):
                emit_trace_event(
                    trace, "hold_gate", status=status,
                    elapsed_seconds=max(0.0, time.monotonic() - started),
                    allocation=record, run_remaining_seconds=remaining_run_seconds(),
                )

            emit("waiting")
            status = "interrupted"
            log.info("hold_until: waiting for signal %s; allocation=%s", signal, record)
            try:
                while True:
                    remaining = remaining_run_seconds()
                    if remaining is not None and remaining <= 0:
                        status = "run_exhausted"
                        raise RuntimeError("YUJ_HOLD_UNTIL: run time budget exhausted")
                    present = os.path.exists(signal)
                    # File probing also consumes the shared run allowance.
                    remaining = remaining_run_seconds()
                    if remaining is not None and remaining <= 0:
                        status = "run_exhausted"
                        raise RuntimeError("YUJ_HOLD_UNTIL: run time budget exhausted")
                    now = time.monotonic()
                    if (present and (shared.deadline is None or now < shared.deadline)
                            and (timeout == 0 or deadline is None or now < deadline)):
                        status = "released"
                        log.info("hold_until: released")
                        return
                    left = None if deadline is None else deadline - now
                    if left is not None and left <= 0:
                        status = "timed_out"
                        raise RuntimeError(f"YUJ_HOLD_UNTIL signal never arrived: {signal}")
                    time.sleep(POLL_SECONDS if left is None else min(POLL_SECONDS, left))
            finally:
                emit(status)
