"""Focused tests for the read-before-edit ledger."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.llm_solver.harness.stale_guard import (
    StaleFileGuard,
    StaleGuardError,
    classify_single_file_read,
)


def test_block_mode_rejects_unread_file_with_trace_event(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)

    decision = guard.check_edit("src.py")

    assert decision.blocked is True
    assert decision.reason == "unread"
    assert decision.message == "ERROR: stale_file: read src.py first"
    assert events == [{
        "event": "stale_guard",
        "path": "src.py",
        "reason": "unread",
        "mode": "block",
        "blocked": True,
        "expected": None,
        "current": None,
    }]


def test_edit_after_read_is_fresh(tmp_path):
    (tmp_path / "src.py").write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    fingerprint = guard.observe_read("src.py")
    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert decision.reason == "fresh"
    assert guard.ledger_snapshot()["src.py"] == fingerprint


def test_external_content_change_after_read_is_blocked(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    guard.observe_read("src.py")

    target.write_text("changed\n")
    decision = guard.check_edit("src.py")

    assert decision.blocked is True
    assert decision.reason == "modified"
    assert decision.message == "ERROR: stale_file: read src.py first"
    assert events[-1]["event"] == "stale_guard"
    assert events[-1]["expected"]["sha256"] != events[-1]["current"]["sha256"]


def test_missing_observed_file_is_stale(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    guard.observe_read("src.py")
    target.unlink()

    decision = guard.check_edit("src.py")

    assert decision.reason == "missing"
    assert decision.blocked is True


def test_warn_mode_reports_hit_but_allows_edit(tmp_path):
    (tmp_path / "src.py").write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="warn")

    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert decision.message == "WARNING: stale_file: read src.py first"


def test_warn_is_the_default_mode(tmp_path):
    (tmp_path / "src.py").write_text("old\n")

    decision = StaleFileGuard(cwd=tmp_path).check_edit("src.py")

    assert decision.mode == "warn"
    assert decision.allowed is True


def test_off_mode_performs_no_file_check_or_trace(tmp_path):
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="off", event_sink=events.append)

    decision = guard.check_edit("does-not-exist.py")

    assert decision.allowed is True
    assert decision.reason == "off"
    assert events == []


@pytest.mark.parametrize("source", ["write", "edit", "apply_patch"])
def test_successful_mutation_refreshes_ledger(tmp_path, source):
    target = tmp_path / "src.py"
    target.write_text("old\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")
    guard.observe_read("src.py")
    target.write_text("our edit\n")

    guard.observe_mutation("src.py", source=source)

    assert guard.check_edit("src.py").allowed is True


def test_metadata_only_touch_keeps_content_fresh(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("same\n")
    events = []
    guard = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    before = guard.observe_read("src.py")
    os.utime(target, ns=(before.mtime_ns + 1_000_000, before.mtime_ns + 1_000_000))

    decision = guard.check_edit("src.py")

    assert decision.allowed is True
    assert events[-1]["source"] == "metadata_refresh"


@pytest.mark.parametrize(
    ("command", "verb", "path"),
    [
        ("cat src/app.py", "cat", "src/app.py"),
        ("LC_ALL=C head -n 20 -- src/app.py", "head", "src/app.py"),
        ("tail -n 5 src/app.py", "tail", "src/app.py"),
        ("sed -n '1,40p' src/app.py", "sed", "src/app.py"),
        ("grep -n 'needle' src/app.py", "grep", "src/app.py"),
        ("rg --line-number 'needle' src/app.py", "rg", "src/app.py"),
    ],
)
def test_single_file_shell_read_classifier_accepts_safe_reads(command, verb, path):
    classified = classify_single_file_read(command)

    assert classified is not None
    assert (classified.verb, classified.path) == (verb, path)


@pytest.mark.parametrize(
    "command",
    [
        "cat one.py two.py",
        "cat src.py && echo done",
        "cat src.py | head",
        "cat < src.py",
        "cat $(printf src.py)",
        "sed -i 's/a/b/' src.py",
        "sed -n '1p' one.py two.py",
        "grep -r needle src",
        "grep -c needle src.py",
        "rg --count needle src.py",
        "rg needle .",
        "wc -l src.py",
    ],
)
def test_single_file_shell_read_classifier_rejects_ambiguous_or_aggregate(command):
    assert classify_single_file_read(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "cat src.py",
        "sed -n '1p' src.py",
        "grep needle src.py",
    ],
)
def test_shell_classifier_observation_satisfies_guard(tmp_path, command):
    (tmp_path / "src.py").write_text("needle\n")
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    assert guard.observe_shell_read(command) is not None
    assert guard.check_edit("src.py").allowed is True


def test_shell_observation_requires_existing_single_file(tmp_path):
    guard = StaleFileGuard(cwd=tmp_path, mode="block")

    assert guard.observe_shell_read("cat missing.py") is None
    assert guard.ledger_snapshot() == {}


def test_resume_rebuilds_ledger_from_trace_and_detects_later_change(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("observed\n")
    events = []
    live = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    live.observe_read("src.py")

    resumed = StaleFileGuard.from_trace(
        cwd=tmp_path, mode="block", events=events
    )
    assert resumed.check_edit("src.py").allowed is True
    target.write_text("external\n")

    assert resumed.check_edit("src.py").reason == "modified"


def test_deleted_path_event_removes_resumed_observation(tmp_path):
    target = tmp_path / "src.py"
    target.write_text("observed\n")
    events = []
    live = StaleFileGuard(cwd=tmp_path, mode="block", event_sink=events.append)
    live.observe_read("src.py")
    target.unlink()
    live.forget("src.py")

    resumed = StaleFileGuard.from_trace(
        cwd=tmp_path, mode="block", events=events
    )

    assert resumed.ledger_snapshot() == {}


def test_trace_rebuild_rejects_path_escape(tmp_path):
    events = [{
        "event": "stale_guard_observe",
        "path": "../secret",
        "source": "read",
        "fingerprint": {"mtime_ns": 1, "size": 1, "sha256": "x"},
    }]

    with pytest.raises(StaleGuardError, match="invalid path"):
        StaleFileGuard.from_trace(cwd=tmp_path, mode="block", events=events)
