"""Tests for harness/savings.py — the token-accounting ledger."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.llm_solver.harness import savings


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
