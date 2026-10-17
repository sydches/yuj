"""Aggregate named operation costs without recording arguments or file contents.

One collector belongs to a session loop. Copied worker contexts share its
totals, but each thread keeps its own nesting. Rows are drained at each model
boundary, so repeated file operations consume one row per operation path/call.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock, get_ident
import time


_collector = ContextVar("operation_timing_collector", default=None)
_frame = ContextVar("operation_timing_frame", default=None)
_call_id = ContextVar("operation_timing_call_id", default=None)


class OperationTimings:
    def __init__(self):
        self.rows = {}
        self.lock = Lock()

    def add(self, path, call_id, elapsed, children, raised):
        with self.lock:
            row = self.rows.setdefault((path, call_id), {
                "calls": 0, "elapsed_ms": 0.0, "self_ms": 0.0,
                "max_ms": 0.0, "raised": 0,
            })
            row["calls"] += 1
            row["elapsed_ms"] += elapsed
            row["self_ms"] += max(0.0, elapsed - children)
            row["max_ms"] = max(row["max_ms"], elapsed)
            row["raised"] += int(raised)

    def take(self):
        with self.lock:
            rows, self.rows = self.rows, {}
        return [
            {"operation": list(path), "tool_call_id": call_id,
             **{key: round(value, 3) if isinstance(value, float) else value
                for key, value in row.items()}}
            for (path, call_id), row in sorted(
                rows.items(), key=lambda item: (item[0][1] or "", item[0][0]))
        ]


@dataclass
class _Frame:
    collector: OperationTimings
    thread: int
    path: tuple
    children_ms: float = 0.0


@contextmanager
def operation_profile():
    collector = OperationTimings()
    token = _collector.set(collector)
    frame_token = _frame.set(None)
    call_token = _call_id.set(None)
    try:
        yield collector
    finally:
        _call_id.reset(call_token)
        _frame.reset(frame_token)
        _collector.reset(token)


@contextmanager
def operation_call(call_id):
    token = _call_id.set(call_id)
    try:
        yield
    finally:
        _call_id.reset(token)


@contextmanager
def pause_operation_timing():
    token = _collector.set(None)
    try:
        yield
    finally:
        _collector.reset(token)


def take_operations():
    collector = _collector.get()
    return collector.take() if collector is not None else []


@contextmanager
def operation(name):
    collector = _collector.get()
    if collector is None:
        yield
        return
    parent = _frame.get()
    thread = get_ident()
    if parent is not None and (parent.collector is not collector or parent.thread != thread):
        parent = None
    frame = _Frame(collector, thread, (*parent.path, name) if parent else (name,))
    token = _frame.set(frame)
    call_id = _call_id.get()
    started = time.perf_counter()
    raised = True
    try:
        yield
        raised = False
    finally:
        elapsed = (time.perf_counter() - started) * 1000
        _frame.reset(token)
        if parent is not None:
            parent.children_ms += elapsed
        collector.add(frame.path, call_id, elapsed, frame.children_ms, raised)


def measured(name):
    """Name must be a code-owned label, never command text or a task path."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _collector.get() is None:
                return function(*args, **kwargs)
            label = name(*args, **kwargs) if callable(name) else name
            with operation(label):
                return function(*args, **kwargs)
        return wrapped
    return decorate
