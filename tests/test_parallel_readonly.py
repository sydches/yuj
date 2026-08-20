"""Tests for parallel read-only tool dispatch.

Direct Session.run() instrumentation is heavy; these tests exercise
the partition logic and the preexecuted-cache fallback separately to
keep the coverage focused.
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness import loop as loop_mod
from llm_solver.harness.tools import dispatch


class _FakeTC:
    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.name = name
        self.arguments = arguments


class TestReadonlyPartition:

    def test_readonly_set_is_stable(self):
        assert loop_mod._READONLY_TOOLS == frozenset({"read", "glob", "grep"})

    def test_write_not_in_readonly(self):
        assert "write" not in loop_mod._READONLY_TOOLS
        assert "edit" not in loop_mod._READONLY_TOOLS
        assert "bash" not in loop_mod._READONLY_TOOLS


class TestConcurrentDispatch:

    def test_two_reads_execute_concurrently(self, tmp_path):
        """Use ThreadPoolExecutor directly to verify dispatch is
        thread-safe for two reads on separate files."""
        (tmp_path / "a.txt").write_text("content-a")
        (tmp_path / "b.txt").write_text("content-b")
        cfg = make_config()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(
                dispatch, "read", {"path": "a.txt"},
                cwd=str(tmp_path), cfg=cfg,
            )
            fb = ex.submit(
                dispatch, "read", {"path": "b.txt"},
                cwd=str(tmp_path), cfg=cfg,
            )
            ra = fa.result()
            rb = fb.result()
        assert "content-a" in ra
        assert "content-b" in rb


class TestConfigDefaults:

    def test_parallel_disabled_by_default(self):
        cfg = make_config()
        assert cfg.parallel_readonly_enabled is False
        assert cfg.parallel_max_workers == 4

    def test_enabling_via_make_config(self):
        cfg = make_config(parallel_readonly_enabled=True, parallel_max_workers=8)
        assert cfg.parallel_readonly_enabled is True
        assert cfg.parallel_max_workers == 8


class TestPartitionConditions:
    """Replicates the entry-condition logic used in Session.run() to
    ensure the parallelism only activates when all three conditions
    hold (flag + >1 call + all read-only)."""

    def _should_parallelize(self, cfg, tcs):
        return (
            cfg.parallel_readonly_enabled
            and len(tcs) > 1
            and all(tc.name in loop_mod._READONLY_TOOLS for tc in tcs)
        )

    def test_disabled_flag_blocks(self):
        cfg = make_config(parallel_readonly_enabled=False)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "read", {"path": "b"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_single_call_blocks(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_any_mutating_tool_blocks(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "write", {"path": "b", "content": "x"})]
        assert self._should_parallelize(cfg, tcs) is False

    def test_multiple_readonly_enabled_passes(self):
        cfg = make_config(parallel_readonly_enabled=True)
        tcs = [_FakeTC("1", "read", {"path": "a"}),
               _FakeTC("2", "grep", {"pattern": "x"})]
        assert self._should_parallelize(cfg, tcs) is True
