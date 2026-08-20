"""Tests for _solver_state_helpers.py — dedup command classification and
error-snippet extraction across languages (F12: go/rust/js, not just pytest).
"""
from __future__ import annotations

from scripts.llm_solver.harness.context_strategies._solver_state_helpers import (
    _classify_cmd,
    _extract_error_snippet,
)


class TestClassifyCmdMultilingual:
    """_TEST_PREFIXES additions: dedup framing must land in 'test', not
    fall through to 'other', for go/rust/js/pnpm/yarn test commands."""

    def test_pytest_still_classified_as_test(self):
        assert _classify_cmd("pytest tests/") == "test"

    def test_go_test_classified_as_test(self):
        assert _classify_cmd("go test ./...") == "test"

    def test_cargo_test_classified_as_test(self):
        assert _classify_cmd("cargo test") == "test"

    def test_jest_classified_as_test(self):
        assert _classify_cmd("jest --ci") == "test"

    def test_npx_jest_classified_as_test(self):
        assert _classify_cmd("npx jest src/foo.test.js") == "test"

    def test_vitest_classified_as_test(self):
        assert _classify_cmd("vitest run") == "test"

    def test_ctest_classified_as_test(self):
        assert _classify_cmd("ctest --output-on-failure") == "test"

    def test_npm_test_classified_as_test(self):
        assert _classify_cmd("npm test") == "test"

    def test_pnpm_test_classified_as_test(self):
        assert _classify_cmd("pnpm test") == "test"

    def test_yarn_test_classified_as_test(self):
        assert _classify_cmd("yarn test") == "test"

    def test_unrelated_command_is_other(self):
        assert _classify_cmd("echo hi") == "other"


class TestExtractErrorSnippetMultilingual:
    """_extract_error_snippet additions: go/rust/js error-line shapes,
    additive to the existing pytest `^E ` handling."""

    def test_pytest_e_line_still_wins(self):
        content = "some output\nE   assert 1 == 2\nmore\n"
        assert _extract_error_snippet(content) == "E   assert 1 == 2"

    def test_go_test_failure_line_is_extracted(self):
        content = (
            "=== RUN   TestAdd\n"
            "--- FAIL: TestAdd (0.00s)\n"
            "    add_test.go:10: expected 3, got 4\n"
            "FAIL\n"
        )
        assert _extract_error_snippet(content) == "--- FAIL: TestAdd (0.00s)"

    def test_rust_panic_line_is_extracted(self):
        content = (
            "running 1 test\n"
            "thread 'main' panicked at 'assertion failed: left == right', src/lib.rs:10:5\n"
            "test result: FAILED\n"
        )
        snippet = _extract_error_snippet(content)
        assert snippet.startswith("thread 'main' panicked")

    def test_rust_compiler_error_is_extracted(self):
        content = (
            "Compiling foo v0.1.0\n"
            "error[E0382]: use of moved value: `x`\n"
            "error: aborting due to previous error\n"
        )
        assert _extract_error_snippet(content) == "error[E0382]: use of moved value: `x`"

    def test_js_error_line_is_extracted(self):
        content = (
            "FAIL src/foo.test.js\n"
            "Error: expect(received).toBe(expected)\n"
            "  at Object.<anonymous> (src/foo.test.js:5:1)\n"
        )
        assert _extract_error_snippet(content) == "Error: expect(received).toBe(expected)"

    def test_js_typed_error_line_is_extracted(self):
        content = "TypeError: Cannot read property 'x' of undefined\n    at foo (bar.js:1:1)\n"
        assert (
            _extract_error_snippet(content)
            == "TypeError: Cannot read property 'x' of undefined"
        )

    def test_fallback_to_last_non_empty_line_when_nothing_matches(self):
        content = "line one\nline two\n"
        assert _extract_error_snippet(content) == "line two"
