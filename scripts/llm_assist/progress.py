"""Incremental progress renderer for assistant runs.

Tails ``.trace.jsonl`` while a session is running and prints newly
observed events in order, so ``run`` and ``resume`` stop feeling like
they block silently until exit. The follower owns only artifacts the
shell already owns — it never talks to the harness loop directly and
never touches measurement-mode runs.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .runner import _format_trace_event


def format_trace_event(event: dict) -> str:
    """Public wrapper around the runner's trace event formatter."""
    return _format_trace_event(event)


class TraceFollower:
    """Poll ``.trace.jsonl`` and print each new event as it lands.

    The follower starts at the current end of the trace file so events
    already on disk (from prior sessions or the same session before the
    follower started) are not replayed. Events written after ``start``
    is called are printed exactly once, in file order.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        print_fn: Callable[[str], None] = print,
        poll_interval: float = 0.2,
    ):
        self._trace_path = Path(artifact_dir) / ".trace.jsonl"
        self._print_fn = print_fn
        self._poll_interval = poll_interval
        self._cursor = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "TraceFollower":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._cursor = self._trace_path.stat().st_size if self._trace_path.is_file() else 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.drain()

    def drain(self) -> list[str]:
        """Read and print any events beyond the current cursor.

        Returns the list of rendered lines in order. Exposed so tests
        can drive the follower deterministically without threading.
        """
        rendered: list[str] = []
        for event in self._read_new_events():
            line = format_trace_event(event)
            self._print_fn(line)
            rendered.append(line)
        return rendered

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self.drain()
            if self._stop.wait(self._poll_interval):
                break

    def _read_new_events(self) -> Iterable[dict]:
        if not self._trace_path.is_file():
            return []
        events: list[dict] = []
        with open(self._trace_path, "r") as f:
            f.seek(self._cursor)
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    # incomplete line; leave cursor before it for next poll
                    break
                self._cursor = f.tell()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    events.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        return events


__all__ = ["TraceFollower", "format_trace_event"]
