"""Declared monotonic run deadlines and derived command allowances."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math
import time


class BudgetExhausted(RuntimeError):
    """No execution time remains; the command must not be launched."""


@dataclass(frozen=True)
class RunBudget:
    started: float
    deadline: float | None
    declared_seconds: float | None
    source: str


_ACTIVE: ContextVar[RunBudget | None] = ContextVar("run_time_budget", default=None)
_COMMAND: ContextVar["CommandAllowance | None"] = ContextVar("command_time_budget", default=None)


def _limit(value):
    if isinstance(value, bool):
        raise ValueError("time allowance must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("time allowance must be a finite nonnegative number")
    return value or None


@contextmanager
def run_time_budget(seconds=0, *, source="caller"):
    """Scope one invocation; a nested declaration cannot extend its parent."""
    duration = _limit(seconds)
    started = time.monotonic()
    deadline = started + duration if duration is not None else None
    parent = _ACTIVE.get()
    if parent is not None and parent.deadline is not None:
        if deadline is None or parent.deadline <= deadline:
            deadline = parent.deadline
            source = parent.source
            duration = parent.declared_seconds
    token = _ACTIVE.set(RunBudget(started, deadline, duration, source))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def solve_time_budget(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        cfg = kwargs.get("cfg") if "cfg" in kwargs else args[1]
        with run_time_budget(getattr(cfg, "task_wall_clock_limit_s", 0),
                             source="loop.task_wall_clock_limit_s"):
            return function(*args, **kwargs)
    return scoped


def remaining_run_seconds():
    budget = _ACTIVE.get()
    if budget is None or budget.deadline is None:
        return None
    return max(0.0, budget.deadline - time.monotonic())


def execution_deadline():
    command = _COMMAND.get()
    run = _ACTIVE.get()
    deadlines = [b.deadline for b in (command, run) if b is not None and b.deadline is not None]
    return min(deadlines) if deadlines else None


def command_timeout(ceiling=None):
    """Recalculate an active command's remainder at the backend launch point."""
    command = _COMMAND.get()
    return remaining_before(command.deadline, ceiling) if command is not None else ceiling


def command_is_scoped():
    return _COMMAND.get() is not None


@dataclass(frozen=True)
class CommandAllowance:
    deadline: float | None
    record: dict

    def remaining(self):
        return remaining_before(self.deadline)


def remaining_before(deadline, ceiling=None):
    if deadline is None:
        return ceiling
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExhausted("execution time budget is exhausted")
    return min(remaining, ceiling) if ceiling is not None else remaining


def command_allowance(per_call_seconds=0):
    cap = _limit(per_call_seconds)
    now = time.monotonic()
    run = _ACTIVE.get()
    run_deadline = run.deadline if run else None
    deadlines = [d for d in (run_deadline, now + cap if cap is not None else None) if d is not None]
    deadline = min(deadlines) if deadlines else None
    remaining = max(0.0, run_deadline - now) if run_deadline is not None else None
    effective = max(0.0, deadline - now) if deadline is not None else None
    source = ("no_declared_limit" if deadline is None else
              "run_remaining" if deadline == run_deadline else "per_call_limit")
    return CommandAllowance(deadline, {
        "policy": "remaining_run_capped_by_declared_call_limit_v1",
        "run_budget_source": run.source if run else "not_supplied",
        "declared_run_seconds": run.declared_seconds if run else None,
        "per_call_limit_seconds": cap, "run_remaining_seconds": remaining,
        "effective_seconds": effective, "limiting_source": source,
        "status": "exhausted" if effective == 0 else "unbounded" if effective is None else "allocated",
    })


@contextmanager
def command_time_budget(per_call_seconds=0):
    """Share the smallest run, parent-command and explicit call allowance."""
    allocation = command_allowance(per_call_seconds)
    parent = _COMMAND.get()
    if parent is not None and parent.deadline is not None:
        if allocation.deadline is None or parent.deadline < allocation.deadline:
            allocation = CommandAllowance(parent.deadline, {**allocation.record,
                "limiting_source": "parent_command", "effective_seconds": max(0.0, parent.deadline - time.monotonic())})
            allocation.record["status"] = "allocated" if allocation.record["effective_seconds"] > 0 else "exhausted"
    token = _COMMAND.set(allocation)
    try:
        yield allocation
    finally:
        _COMMAND.reset(token)


def budgeted_bash_execution(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        from ._tools._common import ToolExecutionText
        ceiling = kwargs.get("timeout", 0)
        with command_time_budget(0 if ceiling is None else ceiling) as allocation:
            try:
                allocation.remaining()
                result = function(*args, **{**kwargs,
                    "timeout": allocation.record["effective_seconds"]})
                result.execution_budget = allocation.record
                return result
            except BudgetExhausted:
                return ToolExecutionText(
                    "ERROR: bash did not start: execution time budget is exhausted.",
                    exit_status=None, executed=False, verification_status="budget_exhausted",
                    execution_budget={**allocation.record, "status": "exhausted"})
    return scoped


def budgeted_test_execution(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        from ._tools._common import ToolExecutionText
        cfg = kwargs["cfg"]
        if not getattr(cfg, "tools_run_tests_enabled", False):
            return function(*args, **kwargs)
        with command_time_budget(getattr(cfg, "tools_run_tests_timeout", 0)) as allocation:
            try:
                allocation.remaining()
                result = function(*args, **kwargs)
                if hasattr(result, "execution_budget"):
                    result.execution_budget = allocation.record
                return result
            except BudgetExhausted:
                message = "ERROR: run_tests did not start: execution time budget is exhausted."
                if getattr(cfg, "tools_run_tests_structured_output", True):
                    message = '<test_results status="budget_exhausted">\n' + message + '\n</test_results>'
                return ToolExecutionText(message, exit_status=None, executed=False,
                                         verification_status="budget_exhausted",
                                         execution_budget={**allocation.record, "status": "exhausted"})
    return scoped


def budgeted_test_dispatch(function):
    """Keep test preparation and postprocessing within the same call scope."""
    @wraps(function)
    def scoped(name, arguments, **kwargs):
        cfg = kwargs['cfg']
        if name != 'run_tests' or not getattr(cfg, 'tools_run_tests_enabled', False):
            return function(name, arguments, **kwargs)
        with command_time_budget(getattr(cfg, 'tools_run_tests_timeout', 0)) as allocation:
            try:
                allocation.remaining()
                return function(name, arguments, **kwargs)
            except BudgetExhausted:
                metadata = kwargs.get('execution_metadata')
                if metadata is not None:
                    metadata.update(executed=False, exit_status=None, exit_status_known=False,
                        timed_out=False, verification_status='budget_exhausted',
                        execution_budget={**allocation.record, 'status': 'exhausted',
                                          'effective_seconds': 0.0})
                message = 'ERROR: run_tests did not start: execution time budget is exhausted.'
                if getattr(cfg, 'tools_run_tests_structured_output', True):
                    message = '<test_results status="budget_exhausted">\n' + message + '\n</test_results>'
                return message
    return scoped
