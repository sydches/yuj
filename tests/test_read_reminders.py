"""Tests for read() truncation + empty-read <system-reminder> injection."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.tools import read


class TestReadReminders:

    def test_no_reminder_without_cfg(self, tmp_path):
        (tmp_path / "f.txt").write_text("")
        result = read("f.txt", cwd=str(tmp_path))
        assert "<system-reminder>" not in result

    def test_empty_file_reminder(self, tmp_path):
        (tmp_path / "f.txt").write_text("")
        cfg = make_config()
        result = read("f.txt", cwd=str(tmp_path), cfg=cfg)
        assert "<system-reminder>" in result
        assert "empty" in result.lower()
        assert "f.txt" in result

    def test_truncated_reminder(self, tmp_path):
        body = "\n".join(f"line {i}" for i in range(50))
        (tmp_path / "f.txt").write_text(body)
        cfg = make_config()
        result = read("f.txt", cwd=str(tmp_path), limit=10, cfg=cfg)
        assert "<system-reminder>" in result
        assert "10" in result
        assert "f.txt" in result

    def test_no_reminder_when_full_read_fits(self, tmp_path):
        (tmp_path / "f.txt").write_text("a\nb\nc")
        cfg = make_config()
        result = read("f.txt", cwd=str(tmp_path), limit=100, cfg=cfg)
        assert "<system-reminder>" not in result

    def test_no_reminder_when_no_limit(self, tmp_path):
        (tmp_path / "f.txt").write_text("a\nb\nc")
        cfg = make_config()
        result = read("f.txt", cwd=str(tmp_path), cfg=cfg)
        assert "<system-reminder>" not in result

    def test_empty_file_reminder_no_leading_newline(self, tmp_path):
        """Empty file has empty body; reminder should not be prefixed
        with an orphan newline."""
        (tmp_path / "f.txt").write_text("")
        cfg = make_config()
        result = read("f.txt", cwd=str(tmp_path), cfg=cfg)
        assert not result.startswith("\n")
        assert result.startswith("<system-reminder>")

    def test_dispatch_passes_cfg_to_read(self, tmp_path):
        from llm_solver.harness.tools import dispatch
        (tmp_path / "f.txt").write_text("")
        cfg = make_config()
        result = dispatch("read", {"path": "f.txt"}, cwd=str(tmp_path), cfg=cfg)
        assert "<system-reminder>" in result
