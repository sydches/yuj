"""Tests for the model-callable run_tests tool.

Covers:
  - Disabled-by-default: handler refuses with a clear ERROR.
  - Enabled path: pytest invocation with deterministic flags.
  - Schema filtering: run_tests is dropped from the tool schema list
    when tools.run_tests.enabled is false; present when true.
  - Output protocol: <test_results status="..." exit_code="N">…</test_results>
    envelope, with status discriminating pytest exit codes 1/2/5 and
    separating timed_out from generic error.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness import savings
from llm_solver.harness import tools as tools_mod
from llm_solver.harness._tool_filters import _normalize_memory_addresses
from llm_solver.harness.injections import record_user_turn_delivery
from llm_solver.harness.loop import _filter_disabled_tools
from llm_solver.harness.schemas import get_tool_schemas
from llm_solver.harness.tools import run_tests


def _patch_sandbox(captured: dict, *, exit_code: int = 0,
                   text: str = "", timed_out: bool = False):
    """Install a fake _run_in_sandbox that records the cmd and
    returns the requested triple."""

    def fake(cmd, *, cwd, timeout, sandbox, bwrap_bin, **_unused):
        # This helper records cmd, cwd, and timeout and ignores other keywords.
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return (text, None, True) if timed_out else (text, exit_code, False)

    return patch.object(tools_mod, "_run_in_sandbox", side_effect=fake)


def _ledger_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _advice_text(output: str) -> str:
    return "\n\n".join(
        injection.text
        for injection in getattr(output, "user_turn_injections", ())
    )


def test_run_tests_admission_uses_detected_runner_command(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'fixture'\n")
    result = '<test_results status="passed" runner="cargo">ok</test_results>'
    cfg = make_config()

    with patch.object(
        tools_mod, "_filter_bash_output", return_value=result,
    ) as output_filter:
        admitted = tools_mod.admit_tool_output(
            "run_tests",
            result,
            arguments={},
            cfg=cfg,
            cwd=tmp_path,
        )

    assert admitted == result
    assert output_filter.call_args.args[1] == "cargo test --no-fail-fast --quiet"


# ── Handler: gating ──────────────────────────────────────────────────────

class TestRunTestsGating:

    def test_disabled_returns_error(self, tmp_path):
        cfg = make_config(tools_run_tests_enabled=False)
        out = run_tests(path="tests/", cwd=str(tmp_path), cfg=cfg)
        assert out.startswith("ERROR")
        assert "tools.run_tests.enabled" in out

    def test_enabled_invokes_pytest_with_deterministic_flags(self, tmp_path):
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="1 passed in 0.01s"):
            out = run_tests(path="tests/foo.py", cwd=str(tmp_path), cfg=cfg)
        assert "1 passed in 0.01s" in out
        assert "pytest" in captured["cmd"]
        assert "--tb=short" in captured["cmd"]
        assert "-q" in captured["cmd"]
        assert "--no-header" in captured["cmd"]
        assert "tests/foo.py" in captured["cmd"]

    def test_internal_base_command_override_reuses_resolved_interpreter(
        self, tmp_path,
    ):
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="1 passed"):
            run_tests(
                path="tests/foo.py",
                base_cmd_override=(
                    "/opt/conda/envs/project/bin/python -m pytest "
                    "--tb=short -q --no-header"
                ),
                cwd=str(tmp_path),
                cfg=cfg,
            )
        assert captured["cmd"].startswith(
            "/opt/conda/envs/project/bin/python -m pytest "
        )

    def test_enabled_with_k_expression(self, tmp_path):
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text=""):
            run_tests(
                path="", k="test_login or test_logout",
                cwd=str(tmp_path), cfg=cfg,
            )
        assert "-k" in captured["cmd"]
        assert "test_login or test_logout" in captured["cmd"]

    def test_enabled_with_last_failed_flag(self, tmp_path):
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text=""):
            run_tests(last_failed=True, cwd=str(tmp_path), cfg=cfg)
        assert "--lf" in captured["cmd"]

    def test_timeout_propagates_from_config(self, tmp_path):
        cfg = make_config(
            tools_run_tests_enabled=True, tools_run_tests_timeout=42,
        )
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text=""):
            run_tests(cwd=str(tmp_path), cfg=cfg)
        assert captured["timeout"] == 42


# ── Output protocol: structured envelope ─────────────────────────────────

_ENV_RE = re.compile(
    r'<test_results status="(?P<status>[^"]+)"'
    r'(?: exit_code="(?P<ec>\d+)")?'
    r' runner="(?P<runner>[^"]+)">'
    r'\n(?P<body>.*)\n</test_results>',
    re.DOTALL,
)


def _parse_envelope(out: str) -> dict:
    m = _ENV_RE.fullmatch(out)
    assert m is not None, f"output is not a valid envelope: {out!r}"
    return {
        "status": m.group("status"),
        "exit_code": int(m.group("ec")) if m.group("ec") else None,
        "runner": m.group("runner"),
        "body": m.group("body"),
    }


class TestStructuredOutput:

    @pytest.fixture
    def cfg(self):
        return make_config(
            tools_run_tests_enabled=True,
            tools_run_tests_structured_output=True,
        )

    def test_passed_envelope(self, cfg, tmp_path):
        with _patch_sandbox({}, exit_code=0, text="1 passed in 0.01s"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "passed"
        assert parsed["exit_code"] == 0
        assert "1 passed" in parsed["body"]

    def test_failed_envelope(self, cfg, tmp_path):
        with _patch_sandbox({}, exit_code=1, text="1 failed, 2 passed"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "failed"
        assert parsed["exit_code"] == 1

    def test_collection_error_envelope(self, cfg, tmp_path):
        # Pytest exit 2: interrupted / collection error. Easy to confuse
        # with exit 1 once the traceback is collapsed; the envelope's
        # status field is the discriminator.
        with _patch_sandbox({}, exit_code=2, text="ImportError: …"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "collection_error"
        assert parsed["exit_code"] == 2

    def test_collection_error_attaches_failing_assertion_context(self, cfg, tmp_path):
        # collection_error has the same `--tb=short` frame shape as
        # `failed` (pytest emits
        # `tests/foo.py:N: in <module>` lines) — model needs source
        # context just as much. The helper once fired only on
        # `status == "failed"`, leaving collection-error verdicts
        # without inline context.
        (tmp_path / "tests").mkdir()
        test_file = tmp_path / "tests" / "test_imp.py"
        test_file.write_text(
            "import os\n"
            "import sys\n"
            "from foo import does_not_exist\n"
            "def test_something():\n"
            "    assert True\n"
        )
        # Mimic pytest's --tb=short collection-error frame (line 3 is
        # the offending import).
        body = (
            "ERRORS\n"
            "_ ERROR collecting tests/test_imp.py _\n"
            "tests/test_imp.py:3: in <module>\n"
            "    from foo import does_not_exist\n"
            "E   ImportError: cannot import name 'does_not_exist'\n"
        )
        with _patch_sandbox({}, exit_code=2, text=body):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "collection_error"
        assert "<failing-assertion" in parsed["body"]
        assert 'file="tests/test_imp.py"' in parsed["body"]
        assert 'line="3"' in parsed["body"]
        # The cited line itself appears with the `>` marker.
        assert "from foo import does_not_exist" in parsed["body"]

    def test_timed_out_does_not_get_failing_assertion_context(self, cfg, tmp_path):
        # F1 negative: timeout has no body to mine for frames; the
        # helper must stay quiet so the envelope only carries the
        # ERROR string.
        with _patch_sandbox({}, timed_out=True, text=""):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert "<failing-assertion" not in parsed["body"]

    def test_no_tests_collected_envelope(self, cfg, tmp_path):
        # Pytest exit 5: no tests found. Indistinguishable from
        # "everything passed silently" without the envelope.
        with _patch_sandbox({}, exit_code=5, text="no tests ran in 0.00s"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "no_tests_collected"
        assert parsed["exit_code"] == 5
        # Without --lf we must NOT inject the cache-empty hint —
        # legitimate "no tests in this dir" must not get re-routed.
        assert "lastfailed cache" not in parsed["body"]

    def test_lf_cache_empty_attaches_harness_hint(self, cfg, tmp_path):
        # pytest exit 5 + last_failed=True is the canonical
        # "lastfailed cache is empty" signal. Without
        # the hint the model sees a status-only verdict and can't tell
        # whether the suite is empty or --lf is unprimed.
        with _patch_sandbox({}, exit_code=5, text="no tests ran in 0.00s"):
            out = run_tests(last_failed=True, cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "no_tests_collected"
        assert "lastfailed cache" not in parsed["body"]
        assert "lastfailed cache" in _advice_text(out)
        assert "Run run_tests once without last_failed" in _advice_text(out)

    def test_lf_with_real_failures_does_not_attach_cache_hint(self, cfg, tmp_path):
        # --lf with an actual failure (exit 1) must not
        # surface the cache-empty hint — the cache was populated, and
        # there really is a failing test.
        with _patch_sandbox({}, exit_code=1, text="1 failed"):
            out = run_tests(last_failed=True, cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "failed"
        assert "lastfailed cache" not in parsed["body"]

    def test_envelope_carries_runner_pytest_when_no_marker(self, cfg, tmp_path):
        # Runner identity goes in the envelope so trace replay knows
        # which language_quirks
        # template fired, without re-detecting from cwd contents.
        with _patch_sandbox({}, exit_code=0, text="1 passed"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["runner"] == "pytest"

    def test_envelope_carries_runner_cargo_when_cargo_toml_present(self, cfg, tmp_path):
        # F3: a Cargo.toml in cwd flips detection to cargo. The runner
        # attr must reflect that, otherwise the trace silently shows
        # the same output shape regardless of what actually ran.
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
        with _patch_sandbox({}, exit_code=0, text="ok"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["runner"] == "cargo"

    def test_envelope_runner_attr_after_exit_code_preserves_done_guard_prefix(self, cfg, tmp_path):
        # done_guard's verify-pass detector matches
        # `<test_results status="passed"` as a prefix.
        # Adding the runner attr must not push status off the front.
        with _patch_sandbox({}, exit_code=0, text="6 passed in 0.12s"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert out.startswith('<test_results status="passed"')

    def test_timed_out_envelope(self, cfg, tmp_path):
        with _patch_sandbox({}, timed_out=True, text=""):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "timed_out"
        assert parsed["exit_code"] is None
        assert "timed out" in parsed["body"]

    def test_unknown_exit_code_falls_back_to_error_n(self, cfg, tmp_path):
        # An exit code outside the known map should still produce a
        # parseable envelope with a discriminable status.
        with _patch_sandbox({}, exit_code=99, text="weird"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "error_99"
        assert parsed["exit_code"] == 99

    def test_invokes_via_python3_dash_m_pytest(self, tmp_path):
        # `python3 -m pytest` avoids relying on the optional `python`
        # alias while still avoiding bare `pytest`, which is not stable
        # across the sandboxed eval images.
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="1 passed"):
            run_tests(path="tests/", cwd=str(tmp_path), cfg=cfg)
        assert "python3 -m pytest" in captured["cmd"]

    def test_pytest_uses_current_task_environment(self, tmp_path):
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="1 passed"):
            run_tests(path="tests/", cwd=str(tmp_path), cfg=cfg)
        assert captured["cmd"] == (
            "python3 -m pytest --tb=short -q --no-header tests"
        )

    def test_detects_cargo_repo_and_uses_cargo_test(self, tmp_path):
        # Repo carries Cargo.toml → run_tests dispatches to cargo
        # rather than pytest. Verifies the language_quirks loader picks
        # the right runner from cwd contents and that no Python-specific
        # command leaks through.
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="ok"):
            run_tests(cwd=str(tmp_path), cfg=cfg)
        assert "cargo test" in captured["cmd"]
        assert "python -m pytest" not in captured["cmd"]

    def test_detects_go_repo_and_uses_go_test(self, tmp_path):
        # Repo carries go.mod → run_tests dispatches to go test.
        (tmp_path / "go.mod").write_text("module foo\n")
        cfg = make_config(tools_run_tests_enabled=True)
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="ok"):
            run_tests(cwd=str(tmp_path), cfg=cfg)
        assert "go test" in captured["cmd"]
        assert "python -m pytest" not in captured["cmd"]

    # ── Multilingual status mapping ───────────────────────────────────────

    def test_cargo_panic_exit_101_maps_to_failed_not_error_101(self, cfg, tmp_path):
        # cargo test's libtest harness exits 101 (a Rust panic code) when
        # any test fails. This once fell through to run_tests.py's
        # pytest-only _PYTEST_STATUS map and surfaced as the misleading
        # `error_101`, even though it's a completely ordinary test
        # failure in cargo's own vocabulary.
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
        with _patch_sandbox({}, exit_code=101, text="test result: FAILED. 0 passed; 1 failed"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "failed"
        assert parsed["exit_code"] == 101
        assert parsed["runner"] == "cargo"

    def test_pytest_exit_5_still_no_tests_collected_after_multilingual_change(self, cfg, tmp_path):
        # Byte-identical-behavior guard: pytest.toml now declares its own
        # explicit [run_tests.status_map], but the resolved status for
        # exit 5 on a pyproject.toml-only cwd must be unchanged.
        with _patch_sandbox({}, exit_code=5, text="no tests ran in 0.00s"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "no_tests_collected"
        assert parsed["runner"] == "pytest"

    def test_go_nonzero_exit_maps_to_failed_not_pytest_vocabulary(self, cfg, tmp_path):
        # Go's `go test` has no analogue to pytest's exit 2/4/5; any
        # nonzero should read as plain "failed", never "collection_error"
        # or "usage_error".
        (tmp_path / "go.mod").write_text("module foo\n")
        with _patch_sandbox({}, exit_code=2, text="FAIL\tfoo\t0.010s"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "failed"
        assert parsed["runner"] == "go"

    def test_pytest_binary_missing_hint_does_not_fire_for_cargo(self, cfg, tmp_path):
        # Python recovery advice must not fire against a cargo runner's
        # output, even if the raw text resembles a missing-command trigger.
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
        with _patch_sandbox({}, exit_code=127, text="bash: cargo: command not found"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        parsed = _parse_envelope(out)
        assert parsed["status"] == "runner_unavailable"
        assert "could not start" not in parsed["body"]
        assert "existing Python environment" not in parsed["body"]

    def test_binary_missing_hint_fires_on_command_not_found(self, cfg, tmp_path):
        # Sandbox with no `python` on PATH: shell returns 127 plus
        # "command not found". Surface generic Python recovery advice.
        with _patch_sandbox({}, exit_code=127,
                            text="bash: line 1: python: command not found"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert _parse_envelope(out)["status"] == "runner_unavailable"
        advice = _advice_text(out)
        assert "could not start in the current environment" in advice
        assert "existing Python environment" in advice
        assert "Conda" in advice
        assert "uv" in advice
        assert "could not start" not in out

    def test_binary_missing_hint_fires_on_no_module_pytest(self, cfg, tmp_path):
        # python is on PATH but doesn't have pytest installed. The same
        # environment-neutral recovery advice applies.
        with _patch_sandbox({}, exit_code=1,
                            text="/usr/bin/python: No module named pytest"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert _parse_envelope(out)["status"] == "runner_unavailable"
        advice = _advice_text(out)
        assert "could not start in the current environment" in advice
        assert "existing Python environment" in advice
        assert "Conda" in advice
        assert "uv" in advice
        assert "could not start" not in out

    def test_path_missing_hint_still_fires(self, cfg, tmp_path):
        # Regression: path-missing detector must keep working alongside
        # the new binary-missing branch (binary-missing checks first; if
        # binary is fine but the test path is bad, the original hint
        # still applies).
        with _patch_sandbox({}, exit_code=4,
                            text=("ERROR: file or directory not found: "
                                  "tests/missing.py\nno tests ran in 0.00s")):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert "test path does not exist" in _advice_text(out)
        assert "test path does not exist" not in out
        assert "could not start" not in out

    def test_legacy_string_contract_when_structured_disabled(self, tmp_path):
        cfg = make_config(
            tools_run_tests_enabled=True,
            tools_run_tests_structured_output=False,
        )
        # Failed run: legacy contract appends [exit code: N].
        with _patch_sandbox({}, exit_code=1, text="1 failed"):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert "<test_results" not in out
        assert "1 failed" in out
        assert "[exit code: 1]" in out

    def test_legacy_string_contract_timeout_path(self, tmp_path):
        cfg = make_config(
            tools_run_tests_enabled=True,
            tools_run_tests_structured_output=False,
            tools_run_tests_timeout=7,
        )
        with _patch_sandbox({}, timed_out=True):
            out = run_tests(cwd=str(tmp_path), cfg=cfg)
        assert out == "ERROR: command timed out after 7s"


@pytest.mark.parametrize(
    ("cmd", "exit_code", "text", "mechanisms", "buckets"),
    [
        (
            "grep missing sample.py",
            1,
            "",
            ["semantic_exit_annotation"],
            ["exit_annotation"],
        ),
        (
            "touch marker",
            0,
            "",
            ["empty_output_substitution"],
            ["empty_output_sub"],
        ),
        (
            "python -m pytest",
            1,
            "/usr/bin/python: No module named pytest",
            ["exit_code_annotation", "pytest_binary_missing_hint"],
            ["exit_annotation", "advice_injection"],
        ),
        (
            "go test ./...",
            127,
            "bash: line 1: go: command not found",
            ["exit_code_annotation"],
            ["exit_annotation"],
        ),
        (
            "python -m pytest",
            127,
            "bash: line 1: python: command not found",
            ["exit_code_annotation", "pytest_binary_missing_hint"],
            ["exit_annotation", "advice_injection"],
        ),
        (
            "python -m pytest tests/missing.py",
            4,
            "ERROR: file or directory not found: tests/missing.py\nno tests ran",
            ["exit_code_annotation", "pytest_path_missing_hint"],
            ["exit_annotation", "advice_injection"],
        ),
        (
            "pip install numpy",
            1,
            "Could not find a version that satisfies the requirement numpy",
            ["exit_code_annotation", "python_install_failure_hint"],
            ["exit_annotation", "advice_injection"],
        ),
        (
            "python -c 'import numpy'",
            1,
            "ModuleNotFoundError: No module named 'numpy'",
            ["exit_code_annotation", "python_env_missing_hint"],
            ["exit_annotation", "advice_injection"],
        ),
    ],
)
def test_bash_output_injections_are_accounted(
    tmp_path,
    cmd,
    exit_code,
    text,
    mechanisms,
    buckets,
):
    ledger_path = tmp_path / "bash-savings.jsonl"
    savings.open_ledger(ledger_path)
    try:
        with _patch_sandbox({}, exit_code=exit_code, text=text):
            output = tools_mod.bash(
                cmd,
                cwd=str(tmp_path),
                timeout=5,
                sandbox=False,
                bwrap_bin="/nonexistent",
                transform_output=True,
            )
        for injection in output.user_turn_injections:
            record_user_turn_delivery(injection.for_tool_call("call-1"))
    finally:
        savings.close_ledger()

    rows = _ledger_rows(ledger_path)
    assert [row["mechanism"] for row in rows] == mechanisms
    assert [row["bucket"] for row in rows] == buckets
    assert rows[0]["input_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    output_rows = [row for row in rows if row["surface"] == "tool_output"]
    assert output_rows[-1]["output_sha256"] == hashlib.sha256(
        str(output).encode()
    ).hexdigest()
    for left, right in zip(output_rows, output_rows[1:]):
        assert left["output_sha256"] == right["input_sha256"]
    advice_rows = [
        row for row in rows if row["surface"] == "injected_message"
    ]
    assert len(advice_rows) == len(output.user_turn_injections)
    assert all(row["tool_call_id"] == "call-1" for row in advice_rows)


def test_bash_disabled_output_cleanup_emits_no_injection_or_event(tmp_path):
    ledger_path = tmp_path / "control-savings.jsonl"
    savings.open_ledger(ledger_path)
    try:
        with _patch_sandbox(
            {},
            exit_code=1,
            text="ModuleNotFoundError: No module named 'numpy'",
        ):
            output = tools_mod.bash(
                "python -c 'import numpy'",
                cwd=str(tmp_path),
                timeout=5,
                sandbox=False,
                bwrap_bin="/nonexistent",
                transform_output=False,
            )
    finally:
        savings.close_ledger()

    assert "[HARNESS:" not in output
    assert "[exit code:" not in output
    assert _ledger_rows(ledger_path) == []


def test_dispatch_exports_advice_without_changing_tool_result(tmp_path):
    cfg = make_config()
    metadata: dict = {}
    raw = "ModuleNotFoundError: No module named 'numpy'"
    with _patch_sandbox({}, exit_code=1, text=raw):
        output = tools_mod.dispatch(
            "bash",
            {"cmd": "python -c 'import numpy'"},
            cwd=str(tmp_path),
            cfg=cfg,
            execution_metadata=metadata,
            tool_call_id="bash-1",
        )

    assert "cannot import `numpy`" not in output
    records = metadata["user_turn_injections"]
    assert len(records) == 1
    assert "cannot import `numpy`" in records[0].text
    assert records[0].mechanism == "python_env_missing_hint"


def test_memory_addresses_use_stable_per_output_identities(tmp_path):
    raw = (
        "first=0xABCDEF12 repeat=0xabcdef12 "
        "second=0x765c77f78160 literal=0x12345"
    )
    ledger_path = tmp_path / "address-savings.jsonl"
    savings.open_ledger(ledger_path)
    try:
        output = _normalize_memory_addresses(raw)
    finally:
        savings.close_ledger()

    assert output == (
        "first=0xADDR1 repeat=0xADDR1 "
        "second=0xADDR2 literal=0x12345"
    )
    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "tool_output_normalize"
    assert rows[0]["mechanism"] == "memory_address_normalization"
    assert rows[0]["change_count"] == 3


def test_memory_address_normalization_is_always_on_for_process_output(tmp_path):
    outputs = []
    ledger_path = tmp_path / "process-address-savings.jsonl"
    savings.open_ledger(ledger_path)
    try:
        for _ in range(2):
            output, exit_code, timed_out = tools_mod._run_in_sandbox(
                "python3 -c 'print(object())'",
                cwd=str(tmp_path),
                timeout=5,
                sandbox=False,
                bwrap_bin="/nonexistent",
                normalize_output=False,
            )
            assert exit_code == 0
            assert timed_out is False
            outputs.append(output)
    finally:
        savings.close_ledger()

    assert outputs == ["<object object at 0xADDR1>\n"] * 2
    rows = _ledger_rows(ledger_path)
    assert len(rows) == 2
    assert all(
        row["mechanism"] == "memory_address_normalization"
        for row in rows
    )


def test_memory_address_normalization_does_not_change_read_content(tmp_path):
    source = tmp_path / "constants.py"
    source.write_text("MASK = 0x12345678\n")

    output = tools_mod.read(
        "constants.py",
        cwd=str(tmp_path),
        cfg=make_config(),
    )

    assert "0x12345678" in output


@pytest.mark.parametrize(
    ("structured", "exit_code", "text", "last_failed", "mechanism"),
    [
        (False, 1, "/usr/bin/python: No module named pytest", False,
         "pytest_binary_missing_hint"),
        (True, 127, "bash: line 1: python: command not found", False,
         "pytest_binary_missing_hint"),
        (False, 4,
         "ERROR: file or directory not found: tests/missing.py\nno tests ran",
         False, "pytest_path_missing_hint"),
        (True, 1, "/usr/bin/python: No module named pytest", False,
         "pytest_binary_missing_hint"),
        (True, 4,
         "ERROR: file or directory not found: tests/missing.py\nno tests ran",
         False, "pytest_path_missing_hint"),
        (True, 5, "no tests ran in 0.00s", True,
         "pytest_lf_cache_empty_hint"),
    ],
)
def test_run_tests_advice_is_accounted(
    tmp_path,
    structured,
    exit_code,
    text,
    last_failed,
    mechanism,
):
    cfg = make_config(
        tools_run_tests_enabled=True,
        tools_run_tests_structured_output=structured,
    )
    ledger_path = tmp_path / "run-tests-savings.jsonl"
    savings.open_ledger(ledger_path)
    try:
        with _patch_sandbox({}, exit_code=exit_code, text=text):
            output = run_tests(
                last_failed=last_failed,
                cwd=str(tmp_path),
                cfg=cfg,
            )
        assert _ledger_rows(ledger_path) == []
        for injection in output.user_turn_injections:
            record_user_turn_delivery(injection.for_tool_call("call-1"))
    finally:
        savings.close_ledger()

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "advice_injection"
    assert rows[0]["mechanism"] == mechanism
    assert rows[0]["surface"] == "injected_message"
    assert rows[0]["tool_call_id"] == "call-1"
    assert rows[0]["ctx"]["tool_name"] == "run_tests"
    assert rows[0]["input_sha256"] != rows[0]["output_sha256"]
    assert "[HARNESS:" not in output
    assert "[HARNESS:" in _advice_text(output)


# ── Schema filtering ─────────────────────────────────────────────────────

class TestSchemaFiltering:

    def test_run_tests_schema_present_in_tool_schemas(self):
        names = [s["function"]["name"] for s in get_tool_schemas("minimal")]
        assert "run_tests" in names

    def test_filter_drops_run_tests_when_disabled(self):
        cfg = make_config(tools_run_tests_enabled=False)
        schemas = get_tool_schemas("minimal")
        filtered = _filter_disabled_tools(schemas, cfg)
        names = [s["function"]["name"] for s in filtered]
        assert "run_tests" not in names
        assert "bash" in names
        assert "done" in names

    def test_filter_keeps_run_tests_when_enabled(self):
        cfg = make_config(tools_run_tests_enabled=True)
        schemas = get_tool_schemas("minimal")
        filtered = _filter_disabled_tools(schemas, cfg)
        names = [s["function"]["name"] for s in filtered]
        assert "run_tests" in names

    def test_dispatch_registers_run_tests_handler(self):
        from llm_solver.harness.tools import build_tool_registry
        reg = build_tool_registry()
        assert "run_tests" in reg.handlers


# ── Bash() backward compatibility (refactor regression check) ───────────

class TestBashStillWorks:
    """The bash() function was refactored to share _run_in_sandbox with
    run_tests. Its public string-only contract must be unchanged."""

    def test_bash_appends_exit_code_on_failure(self, tmp_path):
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=2, text="boom"):
            out = tools_mod.bash(
                "false", cwd=str(tmp_path), timeout=5,
                sandbox=False, bwrap_bin="/nonexistent",
            )
        assert "boom" in out
        assert "[exit code: 2]" in out

    def test_bash_no_exit_code_marker_on_success(self, tmp_path):
        captured: dict = {}
        with _patch_sandbox(captured, exit_code=0, text="ok"):
            out = tools_mod.bash(
                "true", cwd=str(tmp_path), timeout=5,
                sandbox=False, bwrap_bin="/nonexistent",
            )
        assert out == "ok"
        assert "[exit code:" not in out

    def test_bash_timeout_string(self, tmp_path):
        captured: dict = {}
        with _patch_sandbox(captured, timed_out=True):
            out = tools_mod.bash(
                "sleep 99", cwd=str(tmp_path), timeout=11,
                sandbox=False, bwrap_bin="/nonexistent",
            )
        assert out == "ERROR: command timed out after 11s"

    @pytest.mark.parametrize("module", ["numpy", "zope.interface"])
    def test_bash_missing_python_module_uses_actionable_generic_advice(
        self, tmp_path, module,
    ):
        captured: dict = {}
        with _patch_sandbox(
            captured,
            exit_code=1,
            text=f"ModuleNotFoundError: No module named '{module}'",
        ):
            out = tools_mod.bash(
                f"python -c 'import {module}'", cwd=str(tmp_path), timeout=5,
                sandbox=False, bwrap_bin="/nonexistent",
            )
        advice = _advice_text(out)
        assert f"cannot import `{module}`" in advice
        assert f"-c 'import {module}'" in advice
        assert ".venv/bin/python" in advice
        assert "/opt/conda/envs/*/bin/python" in advice
        assert "/opt/miniconda3/envs/*/bin/python" in advice
        assert "Rerun the failed command with the printed interpreter" in advice
        assert "`uv run`" in advice
        assert "SWE-bench" not in advice
        assert "testbed" not in advice
        assert "[HARNESS:" not in out

    def test_bash_python_install_failure_uses_generic_advice(self, tmp_path):
        captured: dict = {}
        with _patch_sandbox(
            captured,
            exit_code=1,
            text=(
                "ERROR: Could not find a version that satisfies the "
                "requirement numpy (from versions: none)"
            ),
        ):
            out = tools_mod.bash(
                "pip install numpy", cwd=str(tmp_path), timeout=5,
                sandbox=False, bwrap_bin="/nonexistent",
            )
        advice = _advice_text(out)
        assert "Python package installation failed" in advice
        assert "project's dependency files" in advice
        assert "pip" in advice
        assert "uv" in advice
        assert "Conda" in advice
        assert "local cache" in advice
        assert "[HARNESS:" not in out
