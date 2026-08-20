"""Async daemon-thread writer for the .trace.jsonl file.

Per-emit ``trace_file.write(line); trace_file.flush()`` blocks the
harness loop on disk I/O for ~0.5–2 ms each (fsync). The async writer
moves the write+flush to a background daemon thread; the hot path
just enqueues the pre-formatted JSON line and returns.

Crash safety:
  - Writer flushes after EVERY queue item (not after a batch), so
    the window of unflushed events on a hard crash is one entry max.
  - On clean teardown (Session.run finally → stop()), the queue is
    fully drained and the file is fsynced before stop() returns.
  - On Python exit without stop(), the daemon thread is killed but
    items already written are persisted (file handle is owned by
    the caller — driver.py opens with `open(trace_path, "a")`).

Lifetime: started + stopped from Session.run(). Sessions that are
constructed but never .run() (i.e. tests poking at internal state)
get no writer. The lazy lifecycle keeps thread count bounded by the
number of in-flight session runs, not by the number of test fixtures.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import IO

log = logging.getLogger(__name__)

# Sentinel used to signal the writer thread to drain and exit.
_STOP = object()


class AsyncTraceWriter:
    """Background daemon thread draining a queue of pre-formatted lines.

    Single producer (the harness loop) → single consumer (this thread).
    The queue is unbounded; if the consumer falls behind, memory grows
    rather than blocking the loop. In practice the loop produces ~1–3
    entries per turn while the writer drains in microseconds — backlog
    stays at 0.
    """

    def __init__(self, trace_file: IO) -> None:
        self._file = trace_file
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="harness-trace-writer", daemon=True,
        )
        self._thread.start()

    def submit(self, line: str) -> None:
        """Enqueue a pre-formatted JSON line (with trailing newline)."""
        self._queue.put(line)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._file.write(item)
                self._file.flush()
            except Exception as e:
                # Don't crash the writer thread on transient I/O
                # errors — log and continue. Persistent failures
                # leave entries silently undelivered, but the next
                # full-trace projection (state.json) will catch up.
                log.warning("async trace write failed: %s", e)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and wait for the queue to drain.

        Must be called once per writer (idempotent on re-entry — the
        join short-circuits when the thread is already dead).
        """
        if not self._thread.is_alive():
            return
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning(
                "async trace writer did not exit within %.1fs; "
                "queue depth=%d (entries may be lost)",
                timeout, self._queue.qsize(),
            )
