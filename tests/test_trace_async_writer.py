"""AsyncTraceWriter — drain + crash-safety regression tests.

The writer moves the per-emit ``trace_file.write + flush`` off the
harness hot path. Critical contracts:
  - Every submitted line is on disk after stop() returns.
  - stop() is bounded (timeout, never hangs).
  - Multiple writers can coexist without cross-talk.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.llm_solver.harness._loop.trace_async_writer import AsyncTraceWriter


def test_drain_after_stop(tmp_path: Path):
    p = tmp_path / "trace.jsonl"
    with open(p, "a") as f:
        w = AsyncTraceWriter(f)
        for i in range(100):
            w.submit(f'{{"i":{i}}}\n')
        w.stop(timeout=5.0)
    lines = p.read_text().splitlines()
    assert len(lines) == 100
    assert lines[0] == '{"i":0}' and lines[-1] == '{"i":99}'


def test_stop_idempotent(tmp_path: Path):
    p = tmp_path / "trace.jsonl"
    with open(p, "a") as f:
        w = AsyncTraceWriter(f)
        w.submit('{"x":1}\n')
        w.stop(timeout=5.0)
        # Second stop must not hang or raise.
        w.stop(timeout=1.0)
    assert p.read_text() == '{"x":1}\n'


def test_per_item_flush_visibility(tmp_path: Path):
    """Each item is flushed; clean stop adds the durability fsync."""
    p = tmp_path / "trace.jsonl"
    with open(p, "a") as f:
        w = AsyncTraceWriter(f)
        w.submit('{"a":1}\n')
        w.submit('{"b":2}\n')
        w.stop(timeout=5.0)
    # Re-read from a fresh open to bypass the original handle's buffer.
    assert p.read_text() == '{"a":1}\n{"b":2}\n'


def test_two_writers_two_files(tmp_path: Path):
    p1 = tmp_path / "t1.jsonl"
    p2 = tmp_path / "t2.jsonl"
    with open(p1, "a") as f1, open(p2, "a") as f2:
        w1 = AsyncTraceWriter(f1)
        w2 = AsyncTraceWriter(f2)
        for i in range(50):
            w1.submit(f'{{"src":1,"i":{i}}}\n')
            w2.submit(f'{{"src":2,"i":{i}}}\n')
        w1.stop(timeout=5.0)
        w2.stop(timeout=5.0)
    assert p1.read_text().count('"src":1') == 50
    assert p2.read_text().count('"src":2') == 50
    assert '"src":1' not in p2.read_text()
    assert '"src":2' not in p1.read_text()


def test_barrier_fsyncs_prior_rows_before_ack(tmp_path: Path, monkeypatch):
    p = tmp_path / "barrier.jsonl"
    calls = []
    real_fsync = os.fsync

    def checked_fsync(fd):
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", checked_fsync)
    with open(p, "a") as trace_file:
        writer = AsyncTraceWriter(trace_file)
        writer.submit('{"before":"barrier"}\n')
        writer.barrier()
        assert p.read_text() == '{"before":"barrier"}\n'
        writer.stop()
    assert calls


def test_barrier_propagates_an_earlier_writer_failure():
    class BrokenTrace:
        def write(self, _value):
            raise OSError("disk unavailable")

        def flush(self):
            pass

    writer = AsyncTraceWriter(BrokenTrace())
    writer.submit('{"lost":true}\n')
    with pytest.raises(RuntimeError, match="barrier failed"):
        writer.barrier()
    with pytest.raises(RuntimeError, match="barrier failed"):
        writer.stop()
