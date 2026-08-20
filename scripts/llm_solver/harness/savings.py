"""Savings ledger — token-accounting for harness transforms.

Writes ``<task_cwd>/.savings.jsonl`` as one JSON record per transform
event. Records input/output char counts, computes the delta, and tags
each event as ``exact`` or ``estimate``.

Always on. The bucket taxonomy records exact and estimated savings
separately.

The module-level singleton is justified by the harness's single-
process-per-task execution model: ``solve_task`` opens a ledger at
the start of a task and closes it at the end. ``Session`` sets the
current ``(session, turn)`` on the ledger at the top of each turn,
so downstream transforms call ``get_ledger().record(...)`` without
threading a reference through dispatch / tool / context call stacks.

Sign convention for ``delta_chars``:
  negative  → tokens saved (output smaller than input)
  positive  → tokens paid (one-time costs like system prompt)
  zero      → no-op (recorded for completeness)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


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

    def close(self) -> None:
        pass


class SavingsLedger:
    """Append-only JSONL ledger of transform savings / cost events.

    One file per task at ``<cwd>/.savings.jsonl``. The file is opened
    in append mode so resumed runs do not clobber prior records.
    Schema fields are documented in the module docstring.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Best-effort: open for append. A write failure later flips to
        # debug log — a broken ledger must not block the run.
        self._file = open(self._path, "a")
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
        try:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except OSError as e:
            log.debug("Savings ledger write failed: %s", e)

    def close(self) -> None:
        """Release the file handle. Idempotent."""
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


_ledger: SavingsLedger | _NullLedger = _NullLedger()


def open_ledger(path: Path) -> SavingsLedger:
    """Open a ledger at path and register as the process-level singleton."""
    global _ledger
    close_ledger()
    _ledger = SavingsLedger(path)
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
