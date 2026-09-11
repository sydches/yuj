"""Tests for done-gate handling of file changes made through bash."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness._guardrails.checks_pre import (
    done_guard,
)
from llm_solver.harness._guardrails._git_dirty import cwd_has_uncommitted_changes as _cwd_has_uncommitted_changes
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
        done_reject_no_mutation="REJECTED: No task-file change has been recorded in this session.",
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


@pytest.mark.parametrize("command", ["python tests/runtests.py unit", "python3.12 ./run_tests.py unit"])
def test_project_runner_pass_satisfies_independent_formal_gate(command):
    from llm_solver.harness._guardrails.verification import observe_post_mutation_verification
    state = GuardrailState()
    state.has_mutated = True
    cfg = _make_cfg(done_guard_enabled=False, post_mutation_verification_gate_after=3,
                    done_reject_no_formal_verification="formal suite required")
    assert done_guard(state, cfg, tc_name="done").action == Action.BLOCK
    observe_post_mutation_verification(state, cfg, tc_name="bash", tc_args={"cmd": command},
                                      result="3 passed", gate_blocked=False,
                                      execution_metadata={"executed": True, "exit_status_known": True,
                                                          "exit_status": 0, "verification_status": "passed"})
    assert state.formal_verification_passed_since_mutation
    assert done_guard(state, cfg, tc_name="done").action == Action.PASS


@pytest.mark.parametrize("result,unavailable", [
    ("/usr/bin/python: No module named pytest\n[exit code: 1]", False),
    ("pytest: command not found\n[exit code: 127]", True),
    ("1 failed\n[exit code: 1]", False),
    ("ModuleNotFoundError: No module named 'hypothesis'\n[exit code: 1]", False),
])
def test_formal_gate_preserves_only_runner_unavailable_exception(result, unavailable):
    from llm_solver.harness._guardrails.verification import observe_post_mutation_verification
    state = GuardrailState()
    state.has_mutated = True
    state.post_mutation_automatic_verification_unavailable = True
    cfg = _make_cfg(done_guard_enabled=False, post_mutation_verification_gate_after=3,
                    done_reject_no_formal_verification="formal suite required")
    observe_post_mutation_verification(state, cfg, tc_name="bash",
                                      tc_args={"cmd": "python -m pytest tests/test_source.py"},
                                      result=result, gate_blocked=False,
                                      execution_metadata={"executed": True, "exit_status_known": True,
                                                          "exit_status": 127 if unavailable else 1,
                                                          "verification_status": "runner_unavailable" if unavailable else "failed"})
    assert state.post_mutation_automatic_verification_unavailable is unavailable
    assert done_guard(state, cfg, tc_name="done").action == (Action.PASS if unavailable else Action.BLOCK)
    observe_post_mutation_verification(state, cfg, tc_name="edit", tc_args={"path": "source.py"},
                                      result="OK", gate_blocked=False, source_write_paths=("source.py",))
    assert not state.post_mutation_automatic_verification_unavailable


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


def test_done_gate_does_not_credit_unobserved_git_dirt():
    """Git dirt cannot establish activity during this session."""
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        Path(d, "seed.txt").write_text("seed\nbash-edited\n")

        state = GuardrailState()
        state.has_mutated = False
        state.verified_since_mutation = True  # satisfy the verify check
        cfg = _make_cfg()

        decision = done_guard(state, cfg, tc_name="done", cwd=d)

        assert decision.action == Action.BLOCK
        assert state.has_mutated is False


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
        assert "No task-file change has been recorded" in decision.text


def test_done_gate_rejects_when_no_cwd_and_not_mutated():
    """No cwd / not a git repo → preserves prior reject behavior."""
    state = GuardrailState()
    state.has_mutated = False
    state.verified_since_mutation = True
    cfg = _make_cfg()

    # No cwd at all
    decision = done_guard(state, cfg, tc_name="done")
    assert decision.action == Action.BLOCK
    assert "No task-file change has been recorded" in decision.text


def test_done_gate_unchanged_when_has_mutated():
    """has_mutated=True (harness tools used) → behaviour unchanged."""
    state = GuardrailState()
    state.has_mutated = True
    state.verified_since_mutation = True
    cfg = _make_cfg()

    decision = done_guard(state, cfg, tc_name="done", cwd="/some/path/not/checked")

    assert decision.action == Action.PASS


def test_done_gate_verify_check_still_fires_after_recorded_mutation():
    """Recorded mutation does not bypass the separate verification check."""
    with tempfile.TemporaryDirectory() as d:
        _init_git_repo(d)
        Path(d, "seed.txt").write_text("seed\nbash-edited\n")

        state = GuardrailState()
        state.has_mutated = True
        state.verified_since_mutation = False  # NOT verified
        cfg = _make_cfg()

        decision = done_guard(state, cfg, tc_name="done", cwd=d)

        # Mutation activity is recorded but verification is absent.
        assert decision.action == Action.BLOCK
        assert "Run verification" in decision.text or "verify" in decision.text.lower()
