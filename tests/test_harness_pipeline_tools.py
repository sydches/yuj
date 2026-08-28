"""Tests for harness loop, tools, solver, generate pipeline, config, and end-to-end integration."""
import json
import os
import subprocess as _subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import openai
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.server.types import TurnResult, Usage, ToolCall
from llm_solver.config import Config, load_config, MODEL_MAP, _deep_merge, get_sdk_config


# ──────────────────────────────────────────────
# Helper: build a Config without loading TOML
# ──────────────────────────────────────────────

from _config_helpers import make_config  # centralized defaults — see tests/_config_helpers.py


def make_turn_result(content=None, tool_calls=None, finish_reason="stop", prompt_tokens=10):
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )

# ──────────────────────────────────────────────
# 3. Harness tools
# ──────────────────────────────────────────────

class TestHarnessTools:

    def test_truncate_output_short(self):
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(max_output_chars=1000)
        text = "short text"
        assert truncate_output(text, cfg) == text

    def test_truncate_output_long(self):
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(max_output_chars=100, truncate_head_lines=2, truncate_tail_lines=2)
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = truncate_output(text, cfg)
        assert "line 0" in result
        assert "line 1" in result
        assert "line 199" in result
        assert "omitted" in result

    def test_output_cleanup_switch_keeps_shell_text_unchanged(self):
        from llm_solver.harness.tools import _filter_bash_output, truncate_output

        cfg = make_config(
            transformations_explicit=True,
            output_cleanup_and_normalization=False,
            max_output_chars=8,
        )
        text = "\x1b[31mred\x1b[0m\n\n\nrepeat\nrepeat\n"

        assert _filter_bash_output(text, "printf", cfg) == text
        assert truncate_output(text, cfg) == text

    def test_output_cleanup_switch_keeps_read_bytes_unchanged(self, tmp_path):
        from llm_solver.harness.tools import read

        target = tmp_path / "sample.txt"
        raw = b"first\r\n\r\nthird\r\n"
        target.write_bytes(raw)
        cfg = make_config(
            transformations_explicit=True,
            output_cleanup_and_normalization=False,
        )

        assert read("sample.txt", cwd=str(tmp_path), cfg=cfg) == raw.decode()

    def test_truncate_output_huge_single_line(self):
        """Cap a very long single line by its character count.

        A command can return several megabytes on only a few lines.
        A line-only limit would keep the full result.
        """
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(
            max_output_chars=1000,
            truncate_head_lines=100,
            truncate_tail_lines=50,
        )
        # One line, 100k chars. Line count < head_lines + tail_lines, so
        # the line-based path leaves the whole thing intact. Char cap
        # must kick in.
        text = "x" * 100_000
        result = truncate_output(text, cfg)
        assert len(result) <= cfg.max_output_chars + 200  # cap + bookkeeping text
        assert "chars omitted" in result

    def test_truncate_output_few_massive_lines(self):
        """Each of a few lines is individually larger than max_output_chars."""
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(
            max_output_chars=1000,
            truncate_head_lines=100,
            truncate_tail_lines=50,
        )
        # 5 lines × 50k chars = 250k. Line count fits under head+tail,
        # so line-slice returns the whole thing. Char cap must bound it.
        text = "\n".join("y" * 50_000 for _ in range(5))
        result = truncate_output(text, cfg)
        assert len(result) <= cfg.max_output_chars + 200

    def test_collapse_duplicate_lines_compresses_runs(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "a\nb\nb\nb\nc\n"
        out = _collapse_duplicate_lines(text)
        assert out == "a\nb [×3]\nc\n"

    def test_collapse_duplicate_lines_unique_lines_pass_through(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "one\ntwo\nthree\n"
        assert _collapse_duplicate_lines(text) == text

    def test_collapse_duplicate_lines_is_content_blind(self):
        # The compressor operates on byte equality. It does not know what
        # the lines represent — retry-loop spam, progress-bar repeats,
        # identical status lines from any tool all collapse the same way.
        from llm_solver.harness.tools import _collapse_duplicate_lines
        spam = "connection refused\n" * 5 + "ok\n"
        out = _collapse_duplicate_lines(spam)
        assert "connection refused [×5]" in out
        assert "ok" in out

    def test_collapse_duplicate_lines_respects_intervening_differences(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "x\nx\ny\nx\nx\nx\n"
        out = _collapse_duplicate_lines(text)
        # Two separate runs of x — each compressed independently.
        assert out == "x [×2]\ny\nx [×3]\n"

    # ── Skeleton-based similar-line collapsing ──────────────────────

    def test_line_skeleton_same_template(self):
        from llm_solver.harness.tools import _line_skeleton
        a = _line_skeleton("tests/test_foo.py::test_bar PASSED  [ 3%]")
        b = _line_skeleton("tests/test_foo.py::test_baz PASSED  [ 4%]")
        assert a == b

    def test_line_skeleton_different_punctuation(self):
        from llm_solver.harness.tools import _line_skeleton
        assert _line_skeleton("a::b") != _line_skeleton("a.b")

    def test_collapse_similar_lines_bulk_template_collapses(self):
        """Dominant skeleton (>50% of lines) collapses; rare lines survive."""
        from llm_solver.harness.tools import _collapse_similar_lines
        # 30 PASSED lines (dominant) + 3 FAILED lines (rare) + header/footer
        passed = [f"tests/test_foo.py::test_{i:03d} PASSED  [{i}%]" for i in range(30)]
        failed = [
            "FAILED tests/test_foo.py::test_broken - AssertionError",
            "FAILED tests/test_foo.py::test_other - ValueError",
            "FAILED tests/test_foo.py::test_third - TypeError",
        ]
        header = ["===== test session starts =====", "collected 33 items", ""]
        footer = ["", "===== 3 failed, 30 passed ====="]
        text = "\n".join(header + passed + failed + footer)
        out = _collapse_similar_lines(text)
        # PASSED lines collapsed
        assert "[×30 similar lines]" in out
        # FAILED lines survive individually
        for f in failed:
            assert f in out
        # Header/footer survive
        assert "test session starts" in out
        assert "3 failed, 30 passed" in out

    def test_collapse_similar_lines_no_dominant_template(self):
        """Output with no dominant template passes through unchanged."""
        from llm_solver.harness.tools import _collapse_similar_lines
        lines = [
            "drwxr-xr-x  2 user user  4096 Apr 12 16:48 ci",
            "-rw-r--r--  1 user user   156 Apr 12 16:48 .gitignore",
            "drwxr-xr-x  3 user user  4096 Apr 12 16:48 .github",
            "-rw-r--r--  1 user user  3519 Apr 12 16:48 README.md",
            "drwxr-xr-x  8 user user  4096 Apr 12 16:48 seaborn",
            "-rw-r--r--  1 user user   584 Apr 12 16:48 setup.cfg",
            "drwxr-xr-x  6 user user  4096 Apr 12 16:48 tests",
            "-rw-r--r--  1 user user   512 Apr 12 16:48 CITATION.cff",
            "-rw-r--r--  1 user user  1491 Apr 12 16:48 LICENSE.md",
            "-rw-r--r--  1 user user   219 Apr 12 16:48 Makefile",
        ]
        text = "\n".join(lines)
        out = _collapse_similar_lines(text)
        assert out == text  # no collapse — no single skeleton > 50%

    def test_collapse_similar_lines_small_output_skips(self):
        """Fewer than 10 non-blank lines → no collapse."""
        from llm_solver.harness.tools import _collapse_similar_lines
        lines = [f"tests/test_foo.py::test_{i} PASSED" for i in range(5)]
        text = "\n".join(lines)
        assert _collapse_similar_lines(text) == text

    def test_collapse_similar_lines_preserves_blank_separators(self):
        """Blank lines break consecutive runs of bulk lines."""
        from llm_solver.harness.tools import _collapse_similar_lines
        passed_a = [f"tests/test_a.py::test_{i} PASSED  [{i}%]" for i in range(15)]
        passed_b = [f"tests/test_b.py::test_{i} PASSED  [{50+i}%]" for i in range(15)]
        text = "\n".join(passed_a) + "\n\n" + "\n".join(passed_b)
        out = _collapse_similar_lines(text)
        # Both groups collapse independently; blank line preserved
        assert out.count("[×15 similar lines]") == 2

    def test_collapse_similar_lines_content_blind(self):
        """Compiler warnings collapse the same way as test output."""
        from llm_solver.harness.tools import _collapse_similar_lines
        # 20 warnings (dominant) + 2 errors (rare)
        warnings = [f"src/{chr(97+i)}.c:{i*10}: warning: unused variable" for i in range(20)]
        errors = [
            "src/main.c:5: error: undefined reference to 'foo'",
            "src/main.c:12: error: incompatible types",
        ]
        text = "\n".join(warnings + errors)
        out = _collapse_similar_lines(text)
        assert "[×20 similar lines]" in out
        for e in errors:
            assert e in out

    def test_filter_bash_output_skeleton_collapse_integrated(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=False, collapse_similar_lines=True,
            max_output_chars=200,
        )
        # 30 lines with header — dominant template collapses
        header = ["collected 30 items", ""]
        passed = [f"tests/test_foo.py::test_{i:03d} PASSED  [{i}%]" for i in range(30)]
        out = _filter_bash_output("\n".join(header + passed), "pytest", cfg)
        assert "[×30 similar lines]" in out

    def test_filter_bash_output_skeleton_skips_small_output(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=False, collapse_similar_lines=True,
            max_output_chars=20000,
        )
        lines = [f"tests/test_foo.py::test_{i} PASSED  [{i:2d}%]" for i in range(10)]
        text = "\n".join(lines)
        out = _filter_bash_output(text, "pytest", cfg)
        assert out == text  # no collapse — output too small

    def test_pipeline_byte_identical_before_skeleton(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=True, collapse_similar_lines=True,
        )
        a = "tests/test_foo.py::test_a PASSED  [ 1%]"
        b = "tests/test_foo.py::test_b PASSED  [ 2%]"
        text = "\n".join([a] * 5 + [b] * 5)
        out = _filter_bash_output(text, "pytest", cfg)
        # Byte-identical collapser fires first on each group
        assert f"{a} [×5]" in out
        assert f"{b} [×5]" in out

    # ── Bash command normalization for duplicate detection ──────────

    def test_normalize_bash_strips_trailing_tail(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v 2>&1 | tail -60")
        b = _normalize_bash_for_dedup("pytest tests/ -v 2>&1 | tail -80")
        assert a == b

    def test_normalize_bash_strips_trailing_head(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v | head -100")
        b = _normalize_bash_for_dedup("pytest tests/ -v | head -200")
        assert a == b

    def test_normalize_bash_strips_stderr_redirect(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v 2>&1")
        b = _normalize_bash_for_dedup("pytest tests/ -v")
        assert a == b

    def test_normalize_bash_strips_chained_pipes(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("make 2>&1 | tail -50 | head -20")
        b = _normalize_bash_for_dedup("make")
        assert a == b

    def test_normalize_bash_preserves_meaningful_pipes(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        # Pipes to non-filter commands should be preserved
        a = _normalize_bash_for_dedup("echo hello | python3 -c 'import sys; print(sys.stdin.read())'")
        b = _normalize_bash_for_dedup("echo hello")
        assert a != b

    def test_normalize_bash_preserves_non_bash(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        # No pipes → passthrough
        cmd = "python3 -m pytest tests/test_foo.py -v"
        assert _normalize_bash_for_dedup(cmd) == cmd

    def test_dedup_signature_normalizes_bash(self):
        from llm_solver.harness.loop import _dedup_signature
        from llm_solver.server.types import ToolCall
        tc1 = ToolCall(id="1", name="bash", arguments={"cmd": "pytest -v | tail -60"})
        tc2 = ToolCall(id="2", name="bash", arguments={"cmd": "pytest -v | tail -80"})
        assert _dedup_signature(tc1) == _dedup_signature(tc2)

    def test_dedup_signature_non_bash_unchanged(self):
        from llm_solver.harness.loop import _dedup_signature
        from llm_solver.server.types import ToolCall
        tc1 = ToolCall(id="1", name="read", arguments={"path": "foo.py"})
        tc2 = ToolCall(id="2", name="read", arguments={"path": "foo.py"})
        assert _dedup_signature(tc1) == _dedup_signature(tc2)

    def test_focus_signature_extracts_outside_cwd_find_target(self):
        from llm_solver.harness.loop import _focus_signature

        tc = ToolCall(
            id="1",
            name="bash",
            arguments={"cmd": 'find /opt/miniconda3 -name "generic.py" -path "*/groupby/*" 2>/dev/null'},
        )
        key, display = _focus_signature(tc, "cmd='find /opt/miniconda3 ...'", "/tmp/task")
        assert key.startswith("outside:")
        assert "generic.py" in display
        assert "/opt/miniconda3" in display

    def test_dispatch_unknown_tool(self):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        result = dispatch("nonexistent_tool", {}, cwd="/tmp", cfg=cfg)
        assert "ERROR: unknown tool" in result

    def test_dispatch_bad_args(self):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        result = dispatch("bash", {}, cwd="/tmp", cfg=cfg)  # missing "cmd"
        assert "ERROR" in result

    def test_bash_tool(self, tmp_path):
        from llm_solver.harness.tools import bash
        result = bash(
            "echo hello", cwd=str(tmp_path), timeout=10, sandbox=False
        )
        assert "hello" in result

    def test_bash_timeout(self, tmp_path):
        from llm_solver.harness.tools import bash
        result = bash(
            "sleep 10", cwd=str(tmp_path), timeout=1, sandbox=False
        )
        assert "timed out" in result

    def test_read_tool(self, tmp_path):
        from llm_solver.harness.tools import read
        (tmp_path / "test.txt").write_text("line1\nline2\nline3\n")
        result = read("test.txt", cwd=str(tmp_path))
        assert "1: line1" in result
        assert "2: line2" in result

    def test_read_not_found(self, tmp_path):
        from llm_solver.harness.tools import read
        result = read("nonexistent.txt", cwd=str(tmp_path))
        assert "ERROR: file not found" in result

    def test_write_tool(self, tmp_path):
        from llm_solver.harness.tools import write
        result = write("new.txt", "hello world", cwd=str(tmp_path))
        assert "OK" in result
        assert (tmp_path / "new.txt").read_text() == "hello world"

    def test_edit_tool(self, tmp_path):
        from llm_solver.harness.tools import edit
        (tmp_path / "file.txt").write_text("old text here")
        result = edit("file.txt", "old", "new", cwd=str(tmp_path))
        assert result == "OK"
        assert (tmp_path / "file.txt").read_text() == "new text here"

    def test_edit_not_found(self, tmp_path):
        from llm_solver.harness.tools import edit
        result = edit("file.txt", "old", "new", cwd=str(tmp_path))
        assert "ERROR" in result

    def test_edit_whitespace_normalized_indentation(self, tmp_path):
        """Model rebuilds source from numbered read() output and gets
        indentation slightly wrong (4 spaces instead of the file's
        tabs). Exact match fails; whitespace-normalized fallback
        succeeds when the optional cascade is enabled."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "def foo():\n\treturn 1\n"
        (tmp_path / "f.py").write_text(src)
        # old_str uses 4 spaces instead of a tab
        result = edit(
            "f.py",
            "def foo():\n    return 1",
            "def foo():\n    return 2",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "OK" in result
        assert "whitespace-normalized" in result
        assert (tmp_path / "f.py").read_text() == "def foo():\n    return 2\n"

    def test_edit_whitespace_normalized_extra_blank_lines(self, tmp_path):
        """Cascade arm: model's old_str has an extra blank line the
        file doesn't. Normalized match collapses whitespace runs."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "class X:\n    def m(self):\n        pass\n"
        (tmp_path / "x.py").write_text(src)
        result = edit(
            "x.py",
            "def m(self):\n\n    pass",  # extra blank line
            "def m(self):\n    return 42",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "OK" in result
        assert "return 42" in (tmp_path / "x.py").read_text()

    def test_edit_whitespace_fallback_preserves_non_whitespace_exactness(self, tmp_path):
        """Cascade arm: the fuzzy fallback must NOT match across a
        typo in a real identifier — we don't want to silently rewrite
        the wrong function. Only whitespace is relaxed."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "def foo_bar():\n    return 1\n"
        (tmp_path / "f.py").write_text(src)
        # Typo: foo_baz instead of foo_bar
        result = edit(
            "f.py",
            "def foo_baz():\n    return 1",
            "def foo_bar():\n    return 2",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "ERROR" in result
        # File unchanged
        assert (tmp_path / "f.py").read_text() == "def foo_bar():\n    return 1\n"

    def test_edit_whitespace_fallback_uses_first_match(self, tmp_path):
        """Cascade arm: when normalized match has multiple candidates,
        use the first occurrence — same contract as the exact-match
        path."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "pass\n\npass\n"
        (tmp_path / "a.py").write_text(src)
        result = edit("a.py", "pass", "yield",
                      cwd=str(tmp_path), cfg=cfg)
        assert "OK" in result
        # Only the first pass is replaced.
        assert (tmp_path / "a.py").read_text() == "yield\n\npass\n"

    def test_glob_tool(self, tmp_path):
        from llm_solver.harness.tools import glob_files
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = glob_files("*.py", cwd=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_dispatch_routes_correctly(self, tmp_path):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        (tmp_path / "test.txt").write_text("hello")
        result = dispatch("read", {"path": "test.txt"}, cwd=str(tmp_path), cfg=cfg)
        assert "hello" in result


# ──────────────────────────────────────────────
# 4. Harness solver
# ──────────────────────────────────────────────
