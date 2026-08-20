"""Tests for done-gate handling of file changes made through bash."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness._guardrails.checks_pre import (
    _cwd_has_uncommitted_changes,
    done_guard,
)
from llm_solver.harness._guardrails.state import Action, GuardrailState


def _make_cfg(**overrides):
    """Minimal cfg with the fields done_guard touches."""
    cfg = SimpleNamespace(
        done_guard_enabled=True,
        done_require_mutation=True,
        done_require_verify=True,
        done_require_pretest_parity=False,
        done_parity_runs_required=1,
        done_loop_abort_after=0,
        done_reject_no_mutation="REJECTED: No code changes since session start. Use write, edit, or apply_patch.",
        done_reject_no_verify="REJECTED: Run verification before done.",
        done_reject_parity_no_run="",
        done_reject_parity_still_failing="",
        done_reject_parity_regression="",
        done_reject_parity_streak="",
        done_loop_abort_text="",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _init_git_repo(d: str) -> None:
    """Initialize a git repo at d with an initial commit."""
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", **os.environ}
    subprocess.run(["git", "init", "-q"], cwd=d, check=True, env=env)
    Path(d, "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "-A"], cwd=d, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True, env=env)


def test_cwd_has_uncommitted_changes_clean_repo_returns_false():
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        assert _cwd_has_uncommitted_changes(d) is False


def test_cwd_has_uncommitted_changes_dirty_repo_returns_true():
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        # Simulate bash-fallback mutation
        Path(d, "seed.txt").write_text("seed\nmore\n")
        assert _cwd_has_uncommitted_changes(d) is True


def test_cwd_has_uncommitted_changes_non_git_returns_false():
    with tempfile.TemporaryDirectory() as d:
        # Not a git repo
        assert _cwd_has_uncommitted_changes(d) is False


def test_cwd_has_uncommitted_changes_none_cwd_returns_false():
    assert _cwd_has_uncommitted_changes(None) is False


def test_done_gate_rescues_bash_fallback_mutation():
    """Accept when Git shows changes even if has_mutated is false."""
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        Path(d, "seed.txt").write_text("seed\nbash-edited\n")

        state = GuardrailState()
        state.has_mutated = False
        state.verified_since_mutation = True  # satisfy the verify check
        cfg = _make_cfg()

        decision = done_guard(state, cfg, tc_name="done", cwd=d)

        assert decision.action == Action.PASS, f"Expected PASS, got {decision.action}: {decision.text}"
        assert state.has_mutated is True, "should flip has_mutated on rescue"


def test_done_gate_rejects_when_no_changes():
    """Sanity: clean repo + has_mutated=False still rejects."""
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)

        state = GuardrailState()
        state.has_mutated = False
        state.verified_since_mutation = True
        cfg = _make_cfg()

        decision = done_guard(state, cfg, tc_name="done", cwd=d)

        assert decision.action == Action.BLOCK
        assert "No code changes since session start" in decision.text


def test_done_gate_rejects_when_no_cwd_and_not_mutated():
    """No cwd / not a git repo → preserves prior reject behavior."""
    state = GuardrailState()
    state.has_mutated = False
    state.verified_since_mutation = True
    cfg = _make_cfg()

    # No cwd at all
    decision = done_guard(state, cfg, tc_name="done")
    assert decision.action == Action.BLOCK
    assert "No code changes since session start" in decision.text


def test_done_gate_unchanged_when_has_mutated():
    """has_mutated=True (harness tools used) → behaviour unchanged."""
    state = GuardrailState()
    state.has_mutated = True
    state.verified_since_mutation = True
    cfg = _make_cfg()

    decision = done_guard(state, cfg, tc_name="done", cwd="/some/path/not/checked")

    assert decision.action == Action.PASS


def test_done_gate_verify_check_still_fires_after_rescue():
    """After rescue: state.has_mutated flips True; verify check still runs."""
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        Path(d, "seed.txt").write_text("seed\nbash-edited\n")

        state = GuardrailState()
        state.has_mutated = False
        state.verified_since_mutation = False  # NOT verified
        cfg = _make_cfg()

        decision = done_guard(state, cfg, tc_name="done", cwd=d)

        # Rescue passed but verify rejects
        assert decision.action == Action.BLOCK
        assert "Run verification" in decision.text or "verify" in decision.text.lower()
