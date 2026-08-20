"""Invisible per-turn git snapshots — contract tests.

The feature's two promises:
  1. any snapshotted turn can be restored exactly (rewind/branch point);
  2. the model can never observe that snapshots happen (no log entry, no
     status change, no index disturbance).
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.llm_solver._shared.telemetry_paths import telemetry_dir
from scripts.llm_solver.harness.turn_snapshots import (
    MAP_NAME,
    ensure_snapshot_setup,
    read_map,
    sha_at_or_before,
    snapshot,
)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


def _make_repo(tmp_path):
    repo = tmp_path / "run" / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)])
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_snapshot_returns_sha_and_writes_map(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    sha = snapshot(repo, 5)
    assert sha and len(sha) >= 7
    assert read_map(repo) == [(5, sha)]


def test_restore_rebuilds_exact_state(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("fresh = True\n")
    sha = snapshot(repo, 7)
    # mutate further (later turns), then rewind to turn 7's snapshot
    (repo / "a.py").write_text("x = 999\n")
    (repo / "new.py").unlink()
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", sha, "--", "."])
    assert (repo / "a.py").read_text() == "x = 2\n"
    assert (repo / "new.py").read_text() == "fresh = True\n"


def test_invisible_to_log_status_and_index(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    log_before = _git(repo, "log", "--oneline", "--all")
    (repo / "a.py").write_text("x = 3\n")
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--name-only")
    assert snapshot(repo, 9)
    assert _git(repo, "log", "--oneline", "--all") == log_before
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--name-only") == staged_before


def test_snapshot_object_survives_and_parents_head(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    ensure_snapshot_setup(repo)
    (repo / "a.py").write_text("x = 4\n")
    sha = snapshot(repo, 3)
    head = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-parse", f"{sha}^") == head
    assert _git(repo, "cat-file", "-t", sha) == "commit"


def test_multiple_turns_and_rewind_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    shas = {}
    for turn, content in ((4, "x = 40\n"), (9, "x = 90\n"), (15, "x = 150\n")):
        (repo / "a.py").write_text(content)
        shas[turn] = snapshot(repo, turn)
    assert sha_at_or_before(repo, 9) == shas[9]
    assert sha_at_or_before(repo, 12) == shas[9]
    assert sha_at_or_before(repo, 3) is None
    assert len({s for s in shas.values()}) == 3


def test_failure_is_silent_and_warned_once(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    not_repo = tmp_path / "run" / "repo"
    not_repo.mkdir(parents=True)  # no git here -> snapshot must fail softly
    session = SimpleNamespace()
    assert snapshot(not_repo, 1, session=session) is None
    assert getattr(session, "_snapshot_warned") is True
    assert snapshot(not_repo, 2, session=session) is None
    assert read_map(not_repo) == []


def test_map_lives_in_telemetry_not_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    repo = _make_repo(tmp_path)
    (repo / "a.py").write_text("x = 5\n")
    snapshot(repo, 2)
    assert (telemetry_dir(repo) / MAP_NAME).exists()
    assert not (repo / MAP_NAME).exists()
