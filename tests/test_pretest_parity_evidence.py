"""Parity requires observed results for both baseline sets, including omissions."""
from dataclasses import replace
import re
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
from scripts.llm_solver.harness._guardrails.state import Action, GuardrailState
from scripts.llm_solver.harness._loop.state_projection import (
    update_parity_from_parsed, update_parity_from_report,
)


def _session(*, failing=(), passing=("existing",), enabled=True):
    cfg = make_config(
        done_guard_enabled=True, done_require_pretest_parity=enabled,
        done_require_mutation=False, done_require_verify=False,
        post_mutation_verification_gate_after=0, done_loop_abort_after=0,
        done_parity_runs_required=1,
    )
    return SimpleNamespace(
        cfg=cfg,
        _guards=GuardrailState(
            pretest_failing_tests=set(failing), pretest_passing_tests=set(passing),
        ),
        _emit=MagicMock(), _session_number=1,
    )


def _report(session, parsed):
    update_parity_from_report(session, dict(
        parsed, invocation=uuid.uuid4().hex, status="available",
    ))


def _done(session):
    return done_guard(session._guards, session.cfg, tc_name="done")


@pytest.mark.parametrize("status", [None, "SKIPPED", "UNKNOWN"])
def test_omitted_or_unverified_previous_pass_cannot_establish_parity(status):
    session = _session(failing=("repair",))
    results = {"repair": "PASSED"}
    if status is not None:
        results["existing"] = status
    _report(session, {"tests": results})
    assert session._guards.green_parity_streak == 0
    decision = _done(session)
    assert decision.action == Action.BLOCK
    assert "unverified" in decision.text.lower()
    assert "existing" in decision.text
    assert "regression" not in decision.text.lower()


@pytest.mark.parametrize("results", [{}, {"unrelated": "PASSED"}, {"existing": "FAILED"}])
def test_all_passing_baseline_does_not_bypass_parity(results):
    session = _session()
    _report(session, {"tests": results})
    assert _done(session).action == Action.BLOCK


@pytest.mark.parametrize("failing", [(), ("repair",)])
def test_complete_results_pass_then_empty_run_invalidates_prior_record(failing):
    session = _session(failing=failing)
    results = {test: "PASSED" for test in (*failing, "existing")}
    _report(session, {"tests": results})
    assert _done(session).action == Action.PASS
    _report(session, {"tests": {}})
    assert session._guards.latest_test_parsed == {}
    assert session._guards.green_parity_streak == 0
    assert _done(session).action == Action.BLOCK


def test_regression_and_required_streak_preserve_distinct_reasons():
    session = _session(failing=("repair",))
    session.cfg = replace(session.cfg, done_parity_runs_required=2)
    _report(session, {"tests": {"repair": "PASS", "existing": "FAIL"}})
    assert "regression" in _done(session).text.lower()
    for expected in (Action.BLOCK, Action.PASS):
        _report(session, {"tests": {"repair": "PASS", "existing": "PASS"}})
        assert _done(session).action == expected


def test_disabled_mode_preserves_observation_without_parity_gate():
    session = _session(enabled=False)
    update_parity_from_parsed(session, {"tests": {"existing": "FAILED"}})
    assert session._guards.prev_test_parsed == {"existing": "FAILED"}
    assert _done(session).action == Action.PASS


def test_display_projection_cannot_grant_parity(tmp_path, monkeypatch):
    from scripts.llm_solver.bash_quirks import OutputControl, OutputParser
    from scripts.llm_solver.harness._loop.state_projection import project_and_sink

    session = _session()
    session.cfg = replace(
        session.cfg, bash_transforms_structured_output_enabled=True,
        bash_transforms_sink_threshold_chars=0,
    )
    session.cwd = tmp_path
    session._sink_counter = 0
    session.output_parser = OutputParser(
        summary_fields={},
        per_test_regex=re.compile(r"^(?P<verdict>PASSED) (?P<test_id>\S+)", re.MULTILINE),
    )
    session.output_control = OutputControl(
        "", "PASSED", "FAILED", (re.compile(r"check"),), session.output_parser,
    )
    monkeypatch.setattr("scripts.llm_solver.harness.savings.get_ledger", MagicMock())
    project_and_sink(session, "bash", "check", "PASSED existing\n", 1)
    assert _done(session).action == Action.BLOCK
    project_and_sink(session, "bash", "check", "runner stopped before reporting\n", 2)
    assert session._guards.latest_test_parsed == {}
    assert _done(session).action == Action.BLOCK


def test_observed_mutation_invalidates_native_parity():
    from scripts.llm_solver.harness._guardrails.checks_post import rumination_ladder

    session = _session()
    _report(session, {"tests": {"existing": "PASSED"}})
    assert _done(session).action == Action.PASS
    rumination_ladder(session._guards, session.cfg, tc_name="write",
                      result="written", gate_blocked=False,
                      already_blocked_this_turn=False)
    assert session._guards.latest_test_parsed == {}
    assert _done(session).action == Action.BLOCK
