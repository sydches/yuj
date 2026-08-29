"""Tests for harness/savings.py — the token-accounting ledger."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.llm_solver.bash_quirks._redactions import (
    RedactionRule,
    apply_redactions,
)
from scripts.llm_solver.harness import savings
from scripts.llm_solver.analysis.savings_summary import (
    _aggregate,
    _chain_breaks,
    _discover,
    format_markdown,
)


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_null_ledger_is_default():
    """Before any open_ledger call, get_ledger returns a no-op."""
    savings.close_ledger()  # ensure clean slate
    ledger = savings.get_ledger()
    assert isinstance(ledger, savings._NullLedger)
    # No-op methods don't raise.
    ledger.record("b", "l", "m", input_chars=100, output_chars=50)
    ledger.set_turn(1, 5)


def test_open_ledger_writes_records(tmp_path: Path):
    """open_ledger + record produces one JSONL line per event."""
    path = tmp_path / ".savings.jsonl"
    ledger = savings.open_ledger(path)
    try:
        ledger.set_turn(session=2, turn=7)
        ledger.record("bash_output_condense", "L2_bash_quirks",
                      "pytest_passed_stripping",
                      input_chars=10000, output_chars=2000,
                      ctx={"passed_stripped": 42})
    finally:
        savings.close_ledger()

    records = _read_records(path)
    assert len(records) == 1
    r = records[0]
    assert r["schema_version"] == savings.SCHEMA_VERSION
    assert r["session"] == 2
    assert r["turn"] == 7
    assert r["bucket"] == "bash_output_condense"
    assert r["input_chars"] == 10000
    assert r["output_chars"] == 2000
    assert r["delta_chars"] == -8000     # negative = saved
    assert r["delta_tokens_est"] == -2000  # chars_div_4
    assert r["measure_type"] == "exact"
    assert r["ctx"] == {"passed_stripped": 42}


def test_positive_delta_represents_cost(tmp_path: Path):
    """One-time costs (system prompt) emit a positive delta."""
    path = tmp_path / ".savings.jsonl"
    ledger = savings.open_ledger(path)
    try:
        ledger.record("system_prompt", "harness", "commandments_injection",
                      input_chars=0, output_chars=2400, measure_type="exact")
    finally:
        savings.close_ledger()

    rec = _read_records(path)[0]
    assert rec["delta_chars"] == 2400    # cost paid, not saved
    assert rec["delta_tokens_est"] == 600


def test_close_ledger_resets_to_null(tmp_path: Path):
    """close_ledger drops the file handle and reverts to no-op."""
    path = tmp_path / ".savings.jsonl"
    savings.open_ledger(path)
    savings.close_ledger()
    assert isinstance(savings.get_ledger(), savings._NullLedger)
    # Subsequent record on null ledger silently drops — file should have zero records.
    savings.get_ledger().record("x", "y", "z", input_chars=1, output_chars=1)
    assert not path.read_text()


def test_ledger_appends_across_open_cycles(tmp_path: Path):
    """Re-opening an existing ledger file appends rather than truncates."""
    path = tmp_path / ".savings.jsonl"

    ledger = savings.open_ledger(path)
    ledger.record("b1", "l", "m1", input_chars=100, output_chars=50)
    savings.close_ledger()

    ledger = savings.open_ledger(path)
    ledger.record("b2", "l", "m2", input_chars=200, output_chars=30)
    savings.close_ledger()

    records = _read_records(path)
    assert len(records) == 2
    assert records[0]["bucket"] == "b1"
    assert records[1]["bucket"] == "b2"


def test_estimate_tag_preserved(tmp_path: Path):
    """measure_type='estimate' round-trips unchanged."""
    path = tmp_path / ".savings.jsonl"
    ledger = savings.open_ledger(path)
    try:
        ledger.record("gate_block", "harness", "rumination_gate_block",
                      input_chars=0, output_chars=-2500,  # negative output = counterfactual avoided
                      measure_type="estimate",
                      ctx={"proxy": "mean_prior_bash_chars"})
    finally:
        savings.close_ledger()

    rec = _read_records(path)[0]
    assert rec["measure_type"] == "estimate"
    assert rec["ctx"]["proxy"] == "mean_prior_bash_chars"


def test_counts_mode_records_exact_bytes_hashes_and_chain(tmp_path: Path):
    path = tmp_path / "task.jsonl"
    ledger = savings.open_ledger(path, task="task-17")
    try:
        ledger.set_turn(3, 11)
        with savings.transform_scope("call-9"):
            middle = "café\n\nnext"
            assert ledger.record_transform(
                "filter", "harness", "unicode_change",
                before="café\n\n\nnext",
                after=middle,
                surface="tool_output",
            )
        # A parallel dispatch worker and the main post-processing loop enter
        # separate ContextVar scopes. The ledger must continue the same call.
        with savings.transform_scope("call-9"):
            assert ledger.record_transform(
                "filter", "harness", "suffix",
                before=middle,
                after=middle + "!",
                surface="tool_output",
            )
            assert not ledger.record_transform(
                "filter", "harness", "noop",
                before=middle,
                after=middle,
                surface="tool_output",
            )
    finally:
        savings.close_ledger()

    records = _read_records(path)
    assert len(records) == 2
    first, second = records
    assert first["event"] == "transformation"
    assert first["task"] == "task-17"
    assert first["session"] == 3
    assert first["turn"] == 11
    assert first["tool_call_id"] == "call-9"
    assert first["input_bytes"] == len("café\n\n\nnext".encode("utf-8"))
    assert first["output_bytes"] == len("café\n\nnext".encode("utf-8"))
    assert first["delta_bytes"] == -1
    assert first["input_sha256"] == hashlib.sha256(
        "café\n\n\nnext".encode("utf-8")
    ).hexdigest()
    assert first["output_sha256"] == second["input_sha256"]
    assert first["chain_id"] == second["chain_id"]
    assert [first["chain_step"], second["chain_step"]] == [1, 2]
    for record in records:
        assert record["log_mode"] == "counts"
        assert "changes" not in record
        assert "input_full_path" not in record
        assert "output_full_path" not in record
        assert "before" not in record
        assert "after" not in record
    assert _chain_breaks(records) == []


def test_debug_mode_saves_located_snippets_and_complete_values(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    before = "previous line\n\n\nnext line\nTOKEN=secret"
    after = "previous line\n\nnext line\nTOKEN=[REDACTED]"
    ledger = savings.open_ledger(
        path, transform_log_mode="debug", task="sample-task",
    )
    try:
        ledger.record_transform(
            "filter", "harness", "combined_example",
            before=before,
            after=after,
            surface="tool_output",
            change_count=2,
            ctx={"path": "logs/output.txt"},
        )
    finally:
        savings.close_ledger()

    record = _read_records(path)[0]
    assert record["log_mode"] == "debug"
    assert record["change_count"] == 2
    assert record["changes"]
    assert all("input_byte_range" in change for change in record["changes"])
    assert all("output_byte_range" in change for change in record["changes"])
    assert any("previous line" in change["before"] for change in record["changes"])
    input_path = path.parent / record["input_full_path"]
    output_path = path.parent / record["output_full_path"]
    assert input_path.read_text() == before
    assert output_path.read_text() == after
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == record["input_sha256"]
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == record["output_sha256"]


def test_redaction_suppresses_sensitive_debug_content(tmp_path: Path):
    path = tmp_path / "sensitive.jsonl"
    secret = "TOKEN=secret"
    ledger = savings.open_ledger(path, transform_log_mode="debug")
    try:
        result = apply_redactions(
            secret,
            [RedactionRule(
                name="token",
                pattern=re.compile(r"secret"),
                replace="[REDACTED]",
            )],
        )
    finally:
        savings.close_ledger()

    assert result == "TOKEN=[REDACTED]"
    record = _read_records(path)[0]
    assert record["debug_content_suppressed"] is True
    assert record["input_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
    assert "changes" not in record
    assert "input_full_path" not in record
    assert "output_full_path" not in record
    assert not (tmp_path / "sensitive.transform_debug").exists()


def test_invalid_transform_log_mode_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="transform_log_mode"):
        savings.open_ledger(tmp_path / "bad.jsonl", transform_log_mode="verbose")
    savings.close_ledger()


def test_unavailable_ledger_does_not_stop_the_run(tmp_path: Path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied")

    ledger = savings.open_ledger(blocked_parent / "task.jsonl")
    assert isinstance(ledger, savings._NullLedger)
    ledger.record_transform(
        "filter", "harness", "ignored",
        before="before", after="after", surface="tool_output",
    )


def test_write_failure_disables_accounting_without_raising(tmp_path: Path):
    class BrokenFile:
        def write(self, _value):
            raise OSError("disk full")

        def close(self):
            pass

    ledger = savings.open_ledger(tmp_path / "task.jsonl")
    ledger._file.close()
    ledger._file = BrokenFile()
    ledger.record_transform(
        "filter", "harness", "write_failure",
        before="before", after="after", surface="tool_output",
    )
    assert ledger._file is None
    savings.close_ledger()


def test_invalid_unicode_cannot_break_the_run(tmp_path: Path):
    path = tmp_path / "task.jsonl"
    ledger = savings.open_ledger(path, transform_log_mode="debug")
    assert not ledger.record_transform(
        "filter", "harness", "invalid_unicode",
        before="\udcff", after="valid", surface="tool_output",
    )
    savings.close_ledger()
    assert path.read_text() == ""


def test_large_debug_values_use_bounded_json_and_exact_sidecars(tmp_path: Path):
    path = tmp_path / "large.jsonl"
    before = "head\n" + ("x" * 210_000) + "\nold\ntail"
    after = "head\n" + ("x" * 210_000) + "\nnew\ntail"
    ledger = savings.open_ledger(path, transform_log_mode="debug")
    try:
        ledger.record_transform(
            "filter", "harness", "large_value",
            before=before, after=after, surface="tool_output",
        )
    finally:
        savings.close_ledger()

    raw_record = path.read_text()
    record = json.loads(raw_record)
    assert "\\nold\\n" in raw_record
    assert len(record["changes"][0]["before"]) < 1000
    input_path = path.parent / record["input_full_path"]
    output_path = path.parent / record["output_full_path"]
    assert input_path.read_text() == before
    assert output_path.read_text() == after
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == record["input_sha256"]
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == record["output_sha256"]


def test_aggregate_keeps_transform_bytes_separate_from_legacy_chars():
    records = [
        {
            "event": "transformation",
            "surface": "tool_output",
            "layer": "harness",
            "bucket": "filter",
            "mechanism": "blank_lines",
            "input_bytes": 10,
            "output_bytes": 8,
            "delta_bytes": -2,
            "input_chars": 10,
            "output_chars": 8,
            "delta_chars": -2,
            "change_count": 1,
        },
        {
            "event": "savings",
            "layer": "config",
            "bucket": "system_prompt",
            "mechanism": "header",
            "measure_type": "exact",
            "delta_chars": 100,
        },
    ]
    aggregate = _aggregate(records)
    assert aggregate["transformations"]["totals"]["delta_bytes"] == -2
    assert aggregate["transformations"]["totals"]["bytes_removed"] == 2
    assert aggregate["totals"]["exact_delta"] == 100
    assert aggregate["totals"]["exact_count"] == 1


def test_discovery_keeps_same_task_separate_across_runs(tmp_path: Path):
    for run_name, delta in (("control", "short"), ("treatment", "tiny")):
        ledger_path = tmp_path / run_name / "savings" / "task-a.jsonl"
        ledger = savings.open_ledger(ledger_path, task="task-a")
        ledger.record_transform(
            "filter", "harness", "example",
            before="long value", after=delta, surface="tool_output",
        )
        savings.close_ledger()

    discovered = _discover([tmp_path / "control", tmp_path / "treatment"])
    assert [(run, task) for run, task, _path in discovered] == [
        ("control", "task-a"),
        ("treatment", "task-a"),
    ]
    per_source = {}
    for run, task, ledger_path in discovered:
        aggregate = _aggregate(_read_records(ledger_path))
        aggregate["source"] = {"run": run, "task": task}
        per_source[f"{run}/{task}"] = aggregate
    report = format_markdown(per_source)
    assert "| control | task-a |" in report
    assert "| treatment | task-a |" in report


def test_real_output_filter_chain_has_no_gap_or_overlap(tmp_path: Path):
    from scripts.llm_solver.harness._tool_filters import _filter_bash_output

    path = tmp_path / "filter.jsonl"
    raw = "\x1b[31msame\x1b[0m\nsame\n\n\nend"
    cfg = SimpleNamespace(
        strip_ansi=True,
        collapse_blank_lines=True,
        collapse_duplicate_lines=True,
        collapse_similar_lines=False,
        max_output_chars=1000,
    )
    ledger = savings.open_ledger(path)
    try:
        with savings.transform_scope("call-filter"):
            result = _filter_bash_output(raw, "demo", cfg)
    finally:
        savings.close_ledger()

    records = _read_records(path)
    assert [record["mechanism"] for record in records] == [
        "strip_ansi",
        "collapse_blank_lines",
        "collapse_duplicate_lines",
    ]
    assert _chain_breaks(records) == []
    assert records[-1]["output_sha256"] == hashlib.sha256(
        result.encode("utf-8")
    ).hexdigest()
    assert sum(record["delta_bytes"] for record in records) == (
        len(result.encode("utf-8")) - len(raw.encode("utf-8"))
    )


def test_forbidden_command_logs_requested_and_executed_bytes(tmp_path: Path):
    from _config_helpers import make_config
    from scripts.llm_solver.bash_quirks.transforms import load_forbidden_rules
    from scripts.llm_solver.harness import tools as tools_module

    path = tmp_path / "forbidden.jsonl"
    requested = "cd /home/other/secret && cat passwd"
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return "", 1, False

    ledger = savings.open_ledger(path)
    try:
        with patch.object(tools_module, "_run_in_sandbox", side_effect=fake_run):
            tools_module.dispatch(
                "bash",
                {"cmd": requested},
                cwd=str(tmp_path),
                cfg=make_config(),
                forbidden_rules=load_forbidden_rules(),
                tool_call_id="forbidden-call",
            )
    finally:
        savings.close_ledger()

    records = _read_records(path)
    assert _chain_breaks(records) == []
    command = next(
        record for record in records
        if record["surface"] == "execution_command"
    )
    assert command["mechanism"] == "forbidden:cd_home_other"
    assert command["tool_call_id"] == "forbidden-call"
    assert command["input_bytes"] == len(requested.encode("utf-8"))
    assert command["output_bytes"] == len(captured["cmd"].encode("utf-8"))
    assert command["input_sha256"] == hashlib.sha256(
        requested.encode("utf-8")
    ).hexdigest()
    assert command["output_sha256"] == hashlib.sha256(
        captured["cmd"].encode("utf-8")
    ).hexdigest()
