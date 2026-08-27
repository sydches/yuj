"""Append-only accounting for costs and exact text transformations.

The ledger keeps the older ``savings`` records for configured costs and
counterfactual estimates. A ``transformation`` record is different: the
harness has the exact input and output text at the point of change, so it
stores UTF-8 byte counts, character counts, hashes, location, and order.
Debug mode also stores changed regions and complete before/after files.

Transformation size accounting is always on. Content retention is controlled
by ``loop.transform_log_mode``.

The module-level singleton is justified by the harness's single-
process-per-task execution model: ``solve_task`` opens a ledger at
the start of a task and closes it at the end. ``Session`` sets the
current ``(session, turn)`` on the ledger at the top of each turn,
so downstream transforms call ``get_ledger().record_transform(...)`` without
threading a reference through dispatch / tool / context call stacks.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import difflib
import functools
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any
import uuid

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TRANSFORM_SCHEMA_VERSION = 1
TRANSFORM_LOG_MODES = ("counts", "debug")
_DEBUG_CONTEXT_CHARS = 48
_DEBUG_SNIPPET_MAX_CHARS = 800
_DEBUG_MAX_CHANGES = 50
_DEBUG_DIFF_MAX_CHARS = 200_000


@dataclass
class _TransformScope:
    """Per-tool-call ordering state for nested transformation hooks."""

    tool_call_id: str = ""
    chains: dict[str, str] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)


_transform_scope: ContextVar[_TransformScope | None] = ContextVar(
    "yuj_transform_scope", default=None,
)


@contextmanager
def transform_scope(tool_call_id: str = ""):
    """Group nested transformation records under one tool call.

    Re-entering with the same tool-call ID keeps the current chains. This lets
    the loop own the outer scope while direct and parallel dispatch calls can
    still establish one when no outer scope exists.
    """
    current = _transform_scope.get()
    normalized = str(tool_call_id or "")
    if current is not None and (
        not normalized or current.tool_call_id == normalized
    ):
        yield current
        return
    token = _transform_scope.set(_TransformScope(tool_call_id=normalized))
    try:
        yield _transform_scope.get()
    finally:
        _transform_scope.reset(token)


def transformation_scoped(function):
    """Run a function inside the ``tool_call_id`` keyword's scope."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with transform_scope(str(kwargs.get("tool_call_id", "") or "")):
            return function(*args, **kwargs)
    return wrapped


class _NullLedger:
    """No-op ledger returned by get_ledger() when no ledger is open.

    Every hook site calls ``get_ledger().record(...)``; the null
    variant silently drops the record so hook sites do not need an
    is-open check.
    """
    def set_turn(self, session: int, turn: int) -> None:
        pass

    def record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_transform(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def close(self) -> None:
        pass


class SavingsLedger:
    """Append-only JSONL ledger of transform savings / cost events.

    One file per task at ``<cwd>/.savings.jsonl``. The file is opened
    in append mode so resumed runs do not clobber prior records.
    Schema fields are documented in the module docstring.
    """

    def __init__(
        self,
        path: Path,
        *,
        transform_log_mode: str = "counts",
        task: str = "",
        run: str = "",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = str(transform_log_mode or "counts").strip().lower()
        if mode not in TRANSFORM_LOG_MODES:
            raise ValueError(
                "transform_log_mode must be one of "
                f"{', '.join(TRANSFORM_LOG_MODES)}; got {transform_log_mode!r}"
            )
        self._transform_log_mode = mode
        self._task = str(task or self._path.stem)
        default_run = (
            self._path.parent.parent.name
            if self._path.parent.name == "savings" else self._path.parent.name
        )
        self._run = str(run or default_run)
        self._run_id = uuid.uuid4().hex[:10]
        self._transform_event_count = 0
        self._transform_chain_count = 0
        self._tool_chains: dict[tuple[int, int, str, str], str] = {}
        self._tool_chain_steps: dict[tuple[int, int, str, str], int] = {}
        self._write_lock = threading.Lock()
        # Open for append. ``open_ledger`` handles an unavailable path by
        # installing the null ledger, so accounting never blocks a run.
        self._file = open(self._path, "a", encoding="utf-8")
        self._session = 0
        self._turn = 0

    def set_turn(self, session: int, turn: int) -> None:
        """Update the (session, turn) context stamped on subsequent records."""
        self._session = int(session)
        self._turn = int(turn)

    def record(self, bucket: str, layer: str, mechanism: str,
               *, input_chars: int, output_chars: int,
               measure_type: str = "exact",
               ctx: dict | None = None) -> None:
        """Append one savings/cost event to the ledger."""
        delta = int(output_chars) - int(input_chars)
        record = {
            "event": "savings",
            "schema_version": SCHEMA_VERSION,
            "session": self._session,
            "turn": self._turn,
            "bucket": bucket,
            "layer": layer,
            "mechanism": mechanism,
            "measure_type": measure_type,
            "input_chars": int(input_chars),
            "output_chars": int(output_chars),
            "delta_chars": delta,
            "delta_tokens_est": delta // 4,
            "ctx": ctx or {},
        }
        self._write_record(record)

    def record_transform(
        self,
        bucket: str,
        layer: str,
        mechanism: str,
        *,
        before: str,
        after: str,
        surface: str,
        change_count: int = 1,
        ctx: dict | None = None,
        tool_call_id: str = "",
        chain_id: str = "",
        chain_step: int | None = None,
    ) -> bool:
        """Record one exact text change and return whether text changed."""
        before = str(before)
        after = str(after)
        if before == after:
            return False
        try:
            before_bytes = before.encode("utf-8")
            after_bytes = after.encode("utf-8")
        except UnicodeError as exc:
            log.warning(
                "Transformation accounting skipped invalid UTF-8 text: %s", exc
            )
            return False

        event_id, resolved_chain, resolved_step, resolved_call = (
            self._next_transform_position(
                surface=str(surface),
                tool_call_id=str(tool_call_id or ""),
                chain_id=str(chain_id or ""),
                chain_step=chain_step,
            )
        )
        record = {
            "event": "transformation",
            "schema_version": TRANSFORM_SCHEMA_VERSION,
            "event_id": event_id,
            "log_mode": self._transform_log_mode,
            "run": self._run,
            "task": self._task,
            "session": self._session,
            "turn": self._turn,
            "tool_call_id": resolved_call,
            "surface": str(surface),
            "chain_id": resolved_chain,
            "chain_step": resolved_step,
            "bucket": str(bucket),
            "layer": str(layer),
            "mechanism": str(mechanism),
            "transform": str(mechanism),
            "measure_type": "exact",
            "input_bytes": len(before_bytes),
            "output_bytes": len(after_bytes),
            "delta_bytes": len(after_bytes) - len(before_bytes),
            "input_chars": len(before),
            "output_chars": len(after),
            "delta_chars": len(after) - len(before),
            "change_count": max(1, int(change_count)),
            "input_sha256": hashlib.sha256(before_bytes).hexdigest(),
            "output_sha256": hashlib.sha256(after_bytes).hexdigest(),
            "ctx": ctx or {},
        }
        if self._transform_log_mode == "debug":
            record.update(self._write_debug_values(event_id, before, after))
        self._write_record(record)
        return True

    def _next_transform_position(
        self,
        *,
        surface: str,
        tool_call_id: str,
        chain_id: str,
        chain_step: int | None,
    ) -> tuple[str, str, int, str]:
        scope = _transform_scope.get()
        resolved_call = tool_call_id or (
            scope.tool_call_id if scope is not None else ""
        )
        with self._write_lock:
            self._transform_event_count += 1
            event_id = (
                f"tx-{self._run_id}-{self._transform_event_count:06d}"
            )
            if chain_id:
                resolved_chain = chain_id
                resolved_step = int(chain_step or 1)
            elif resolved_call:
                key = (
                    self._session,
                    self._turn,
                    resolved_call,
                    surface,
                )
                resolved_chain = self._tool_chains.get(key, "")
                if not resolved_chain:
                    self._transform_chain_count += 1
                    resolved_chain = (
                        f"chain-{self._run_id}-"
                        f"{self._transform_chain_count:06d}"
                    )
                    self._tool_chains[key] = resolved_chain
                    self._tool_chain_steps[key] = 0
                resolved_step = self._tool_chain_steps[key] + 1
                self._tool_chain_steps[key] = resolved_step
            elif scope is not None:
                resolved_chain = scope.chains.get(surface, "")
                if not resolved_chain:
                    self._transform_chain_count += 1
                    resolved_chain = (
                        f"chain-{self._run_id}-"
                        f"{self._transform_chain_count:06d}"
                    )
                    scope.chains[surface] = resolved_chain
                    scope.steps[surface] = 0
                resolved_step = scope.steps[surface] + 1
                scope.steps[surface] = resolved_step
            else:
                self._transform_chain_count += 1
                resolved_chain = (
                    f"chain-{self._run_id}-"
                    f"{self._transform_chain_count:06d}"
                )
                resolved_step = int(chain_step or 1)
        return event_id, resolved_chain, resolved_step, resolved_call

    def _write_debug_values(
        self, event_id: str, before: str, after: str,
    ) -> dict[str, Any]:
        debug_dir = self._path.parent / f"{self._path.stem}.transform_debug"
        before_path = debug_dir / f"{event_id}.before.txt"
        after_path = debug_dir / f"{event_id}.after.txt"
        debug_fields: dict[str, Any] = {
            "changes": _changed_snippets(before, after),
        }
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            before_path.write_text(before, encoding="utf-8")
            after_path.write_text(after, encoding="utf-8")
            debug_fields["input_full_path"] = str(
                before_path.relative_to(self._path.parent)
            )
            debug_fields["output_full_path"] = str(
                after_path.relative_to(self._path.parent)
            )
        except (OSError, UnicodeError) as exc:
            log.warning("Transformation debug write failed: %s", exc)
            debug_fields["debug_write_error"] = str(exc)
        return debug_fields

    def _write_record(self, record: dict[str, Any]) -> None:
        try:
            with self._write_lock:
                if self._file is None:
                    return
                self._file.write(json.dumps(record, default=str) + "\n")
                self._file.flush()
        except Exception as exc:
            log.warning("Savings ledger write failed; accounting disabled: %s", exc)
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def close(self) -> None:
        """Release the file handle. Idempotent."""
        if self._file is not None:
            with self._write_lock:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


def _changed_snippets(before: str, after: str) -> list[dict[str, Any]]:
    """Return readable, bounded before/after regions for a debug record."""
    if len(before) + len(after) > _DEBUG_DIFF_MAX_CHARS:
        return [_one_changed_region(before, after)]
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changes: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append(_change_region(before, after, i1, i2, j1, j2))
        if len(changes) >= _DEBUG_MAX_CHANGES:
            break
    return changes


def _one_changed_region(before: str, after: str) -> dict[str, Any]:
    prefix = 0
    prefix_limit = min(len(before), len(after))
    while prefix < prefix_limit and before[prefix] == after[prefix]:
        prefix += 1
    before_tail = len(before)
    after_tail = len(after)
    while (
        before_tail > prefix
        and after_tail > prefix
        and before[before_tail - 1] == after[after_tail - 1]
    ):
        before_tail -= 1
        after_tail -= 1
    return _change_region(
        before, after, prefix, before_tail, prefix, after_tail,
    )


def _change_region(
    before: str,
    after: str,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> dict[str, Any]:
    before_start = max(0, i1 - _DEBUG_CONTEXT_CHARS)
    before_end = min(len(before), i2 + _DEBUG_CONTEXT_CHARS)
    after_start = max(0, j1 - _DEBUG_CONTEXT_CHARS)
    after_end = min(len(after), j2 + _DEBUG_CONTEXT_CHARS)
    return {
        "input_byte_range": [
            len(before[:i1].encode("utf-8")),
            len(before[:i2].encode("utf-8")),
        ],
        "output_byte_range": [
            len(after[:j1].encode("utf-8")),
            len(after[:j2].encode("utf-8")),
        ],
        "before": _bounded_snippet(before[before_start:before_end]),
        "after": _bounded_snippet(after[after_start:after_end]),
    }


def _bounded_snippet(value: str) -> str:
    """Keep a debug JSON region readable; full values live in sidecars."""
    if len(value) <= _DEBUG_SNIPPET_MAX_CHARS:
        return value
    head = _DEBUG_SNIPPET_MAX_CHARS // 2
    tail = _DEBUG_SNIPPET_MAX_CHARS - head
    omitted = len(value) - head - tail
    return (
        value[:head]
        + f"\n[... {omitted} chars omitted from debug snippet ...]\n"
        + value[-tail:]
    )


_ledger: SavingsLedger | _NullLedger = _NullLedger()


def open_ledger(
    path: Path, *, transform_log_mode: str = "counts", task: str = "", run: str = "",
) -> SavingsLedger | _NullLedger:
    """Open a ledger at path and register as the process-level singleton."""
    global _ledger
    close_ledger()
    try:
        _ledger = SavingsLedger(
            path,
            transform_log_mode=transform_log_mode,
            task=task,
            run=run,
        )
    except OSError as exc:
        log.warning("Savings ledger unavailable; accounting disabled: %s", exc)
        _ledger = _NullLedger()
    return _ledger


def close_ledger() -> None:
    """Close the singleton and reset to a no-op."""
    global _ledger
    if isinstance(_ledger, SavingsLedger):
        _ledger.close()
    _ledger = _NullLedger()


def get_ledger() -> SavingsLedger | _NullLedger:
    """Return the current ledger (or the no-op variant if none is open)."""
    return _ledger


def record_text_transform(
    before: str,
    after: str,
    *,
    bucket: str,
    mechanism: str,
    layer: str = "harness",
    surface: str = "tool_output",
    change_count: int = 1,
    ctx: dict | None = None,
    tool_call_id: str = "",
) -> str:
    """Record one exact text change and return the replacement text."""
    get_ledger().record_transform(
        bucket=bucket, layer=layer, mechanism=mechanism,
        before=before, after=after, surface=surface,
        change_count=change_count,
        ctx=ctx,
        tool_call_id=tool_call_id,
    )
    return after


def serialize_messages(messages: list[dict]) -> str:
    """Return the stable UTF-8 JSON representation used for context deltas."""
    return json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
