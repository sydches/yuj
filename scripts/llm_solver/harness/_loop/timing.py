"""Monotonic dispatch and model-boundary timing, without summing parallel work."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import time


@dataclass
class DispatchTiming:
    name: str
    started: float
    finished: float
    ended_at: str
    completed: bool
    emitted: bool = False

    @property
    def milliseconds(self):
        return (self.finished - self.started) * 1000


def timed_dispatch(state, tc, dispatch, *args, emit_end=True, **kwargs):
    """Measure dispatch itself; workers leave event emission to their caller."""
    started = time.perf_counter()
    completed = False
    try:
        from ..read_reuse import reuse_scope
        with reuse_scope(state.session, tc, state.turn,
                         allow_reuse=len(getattr(state, 'schema_validations', {})) <= 1) as reuse:
            result = dispatch(*args, **kwargs)
            if reuse is not None:
                if reuse.reused and reuse.name != 'read' and reuse.after != reuse.before:
                    # A concurrent change during validation needs a fresh
                    # execution, not a reference to an older answer.
                    reuse.previous = None
                    reuse.reused = False
                    (kwargs.get('execution_metadata') or {}).pop('observation_reuse', None)
                    result = dispatch(*args, **kwargs)
                reuse.finish(result, kwargs.get('execution_metadata') or {})
        completed = True
        return result
    finally:
        finished = time.perf_counter()
        record = DispatchTiming(tc.name, started, finished,
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"), completed)
        records = getattr(state.session, "_tool_dispatch_timings", None)
        if records is None:
            records = state.session._tool_dispatch_timings = {}
        records[tc.id] = record
        state.preexecuted_dispatch_ms[tc.id] = record.milliseconds
        if emit_end:
            emit_tool_end(state.session, state.turn, tc.id, record)


def emit_tool_end(session, turn, call_id, record):
    if record.emitted:
        return
    session._emit("tool_end", session_number=session._session_number,
                  turn_number=turn, tool_call_id=call_id, tool_name=record.name,
                  ended_at=record.ended_at, duration_ms=round(record.milliseconds, 2),
                  scope="dispatch", includes_queue_wait=False,
                  dispatch_executed=True, completed=record.completed)
    record.emitted = True


def dispatch_trace_fields(session, fields):
    record = getattr(session, "_tool_dispatch_timings", {}).get(fields.get("tool_call_id"))
    return {"duration_ms": round(record.milliseconds, 2) if record else 0.0,
            "dispatch_executed": record is not None}


def dispatch_union_ms(records):
    end = float("-inf")
    total = 0.0
    for record in sorted(records, key=lambda row: row.started):
        total += max(0.0, record.finished - max(end, record.started))
        end = max(end, record.finished)
    return total * 1000


@dataclass
class TurnTiming:
    """Boundary is entry to _chat_with_retry, before its client preparation.

    The timing row's own emission is outside the measured interval. Terminal
    rows stop at run_session_loop exit, before Session cleanup and writer join.
    post_ms is last dispatch return to that boundary; harness_ms additionally
    includes work before dispatch and between non-overlapping dispatches.
    """
    turn: int
    turn_started: float
    model_started: float
    model_finished: float | None = None
    tools_started: float | None = None
    tools_finished: float | None = None
    post_finished: float | None = None
    dispatches: dict = field(default_factory=dict)

    def finish(self, session, *, next_turn=None):
        ready = time.perf_counter()
        for call_id, record in self.dispatches.items():
            emit_tool_end(session, self.turn, call_id, record)
        tools_started = self.tools_started if self.tools_started is not None else ready
        tools_finished = self.tools_finished if self.tools_finished is not None else ready
        post_finished = self.post_finished if self.post_finished is not None else ready
        chat_ms = ((self.model_finished if self.model_finished is not None else ready)
                   - self.model_started) * 1000
        tool_ms = dispatch_union_ms(self.dispatches.values())
        boundary_ms = (ready - self.model_started) * 1000
        last_dispatch = max((row.finished for row in self.dispatches.values()), default=None)
        session._emit(
            "turn_timing", session_number=session._session_number, turn_number=self.turn,
            next_turn_number=next_turn,
            scope=("tool_preparation_through_next_model_call_entry" if next_turn is not None
                   else "tool_preparation_through_session_end"),
            boundary="next_model_call_entry" if next_turn is not None else "session_end",
            duration_ms=round((ready - tools_started) * 1000, 2),
            tool_phase_ms=round((tools_finished - tools_started) * 1000, 2),
            post_turn_ms=round((post_finished - tools_finished) * 1000, 2),
            next_model_preparation_ms=(round((ready - post_finished) * 1000, 2)
                                       if next_turn is not None else None),
            turn_total_ms=round((ready - self.turn_started) * 1000, 2),
            chat_call_ms=round(chat_ms, 2), tool_ms=round(tool_ms, 2),
            post_ms=round((ready - last_dispatch) * 1000, 2) if last_dispatch is not None else None,
            harness_ms=round(max(0.0, boundary_ms - chat_ms - tool_ms), 2),
            model_to_boundary_ms=round(boundary_ms, 2),
            dispatch_executed=bool(self.dispatches),
        )


def record_turn_timings(function):
    """Finish the last observed model turn on every return or exception."""
    @wraps(function)
    def wrapped(session):
        session._pending_turn_timing = None
        session._tool_dispatch_timings = {}
        try:
            return function(session)
        finally:
            timing = session._pending_turn_timing
            if timing is not None:
                timing.finish(session)
                session._pending_turn_timing = None
            session._tool_dispatch_timings = {}
    return wrapped
