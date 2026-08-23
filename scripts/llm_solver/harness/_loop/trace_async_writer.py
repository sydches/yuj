"""Async daemon-thread writer for the .trace.jsonl file.

Per-emit ``trace_file.write(line); trace_file.flush()`` blocks the
harness loop on disk I/O for ~0.5–2 ms each (fsync). The async writer
moves the write+flush to a background daemon thread; the hot path
just enqueues the pre-formatted JSON line and returns.

Crash safety:
  - Writer flushes after EVERY queue item (not after a batch), so
    another file descriptor can observe each completed item promptly.
  - ``barrier()`` drains prior rows and fsyncs before acknowledging;
    durable diagnostic rows use this to preserve event ordering.
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
import os
import queue
import threading
from typing import IO

log = logging.getLogger(__name__)

# Sentinel used to signal the writer thread to drain and exit.
_STOP = object()


class _DurabilityBarrier:
    """Queue marker acknowledged only after earlier rows reach ``fsync``."""

    def __init__(self, *, require_fsync: bool) -> None:
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.require_fsync = require_fsync


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
        self._failure: BaseException | None = None
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
            if isinstance(item, _DurabilityBarrier):
                try:
                    if self._failure is not None:
                        raise RuntimeError(
                            "an earlier asynchronous trace write failed"
                        ) from self._failure
                    self._file.flush()
                    if item.require_fsync:
                        os.fsync(self._file.fileno())
                except BaseException as exc:
                    item.error = exc
                finally:
                    item.done.set()
                continue
            try:
                self._file.write(item)
                self._file.flush()
            except BaseException as exc:
                # Keep draining so every later barrier is acknowledged, but
                # retain the first lost-row failure and propagate it to the
                # producer. A durability barrier must never report success
                # after an earlier trace row failed.
                if self._failure is None:
                    self._failure = exc
                log.warning("async trace write failed: %s", exc)

    def barrier(
        self, timeout: float = 5.0, *, require_fsync: bool = True
    ) -> None:
        """Drain, flush, and fsync every row submitted before this call."""
        if not self._thread.is_alive():
            if self._failure is not None:
                raise RuntimeError("asynchronous trace writer failed") from self._failure
            raise RuntimeError("asynchronous trace writer is not running")
        marker = _DurabilityBarrier(require_fsync=require_fsync)
        self._queue.put(marker)
        if not marker.done.wait(timeout=timeout):
            raise TimeoutError(
                f"trace durability barrier timed out after {timeout:.1f}s"
            )
        if marker.error is not None:
            raise RuntimeError("trace durability barrier failed") from marker.error

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and wait for the queue to drain.

        Must be called once per writer (idempotent on re-entry — the
        join short-circuits when the thread is already dead).
        """
        if not self._thread.is_alive():
            return
        barrier_error: BaseException | None = None
        try:
            try:
                self._file.fileno()
            except (AttributeError, OSError, ValueError):
                # StringIO and similar test sinks have no kernel durability
                # boundary. They still need an ordered drain before stop.
                require_fsync = False
            else:
                require_fsync = True
            self.barrier(
                timeout=timeout, require_fsync=require_fsync
            )
        except BaseException as exc:
            barrier_error = exc
        finally:
            self._queue.put(_STOP)
            self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning(
                "async trace writer did not exit within %.1fs; "
                "queue depth=%d (entries may be lost)",
                timeout, self._queue.qsize(),
            )
        if barrier_error is not None:
            raise barrier_error
