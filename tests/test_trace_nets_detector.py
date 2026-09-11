"""trace_nets detector backend: mechanical soft-tier nets in the live slot."""
from types import SimpleNamespace
import pytest

from scripts.llm_solver.harness.action_metadata import action_metadata

from scripts.llm_solver.harness.adaptive_control.trace_nets_detector import (
    evaluate_trace_nets,
)


def _ev(turn, args="cmd", sha="s0", pf="fail", write="False", execution_sha=""):
    row = {"event": "tool_call", "turn_number": turn, "args_summary": args,
           "output_sha256": sha, "pass_fail": pf, "source_write_like": write}
    row["action_sha256"] = action_metadata("bash", {"cmd": args})["action_sha256"]
    if execution_sha:
        row["execution_output_sha256"] = execution_sha
    return row


def test_full_call_identity_distinguishes_shared_summary_prefixes():
    from scripts.llm_solver.trace_net_facts import args_reread_after_gap, identical_repeat_plateau_start
    prefix = "print('" + "x" * 220
    first, second = _ev(1, prefix + "A')"), _ev(5, prefix + "B')")
    first["args_summary"] = second["args_summary"] = prefix[:200]
    assert identical_repeat_plateau_start([first, second], 1) is None
    assert args_reread_after_gap([first, second], 1, min_args_len=20, min_gap=3, max_gap=30) is None
    second["action_sha256"] = first["action_sha256"]
    assert identical_repeat_plateau_start([first, second], 1) is not None
    assert args_reread_after_gap([first, second], 1, min_args_len=20, min_gap=3, max_gap=30) is not None


def test_legacy_summary_alone_cannot_prove_identical_calls():
    from scripts.llm_solver.trace_net_facts import args_reread_after_gap, identical_repeat_plateau_start
    rows = [_ev(1, "long matching summary"), _ev(5, "long matching summary")]
    for row in rows:
        row.pop("action_sha256")
    assert identical_repeat_plateau_start(rows, 1) is None
    assert args_reread_after_gap(rows, 1, min_args_len=20, min_gap=3, max_gap=30) is None


@pytest.mark.parametrize("evidence", ["", "   ", "T3:"])
def test_adaptive_advice_without_evidence_is_not_delivered(evidence):
    from scripts.llm_solver.harness.adaptive_control.executors import user_turn_msg_only_apply
    session = SimpleNamespace(_trace_events=[])
    result = user_turn_msg_only_apply(session, evidence=evidence, rung=2, turn=3)
    assert not result.applied and result.blocked_reason == "no_evidence"
    assert not getattr(session, "_adaptive_user_turn_pending", None)


def test_adaptive_warning_describes_a_continuing_session():
    from scripts.llm_solver.harness.adaptive_control.executors import compose_user_turn_message
    message = compose_user_turn_message(SimpleNamespace(_trace_events=[]), evidence="T3:repeated call",
                                        rung=2, hurdle_family="repeat_wall", turn=3)
    assert message.startswith("Harness observation at turn 3: repeated call.")
    assert "stopped" not in message and "end the session" not in message
    assert "warning" in message


def _session(events, arm_after=0):
    return SimpleNamespace(cfg=SimpleNamespace(guardrails_arm_after_turn=arm_after),
                           _trace_events=events)


@pytest.mark.parametrize("turn", [4, 5])
@pytest.mark.parametrize("progress", [False, True])
def test_native_abstention_cannot_clear_or_escalate_pending_watch(monkeypatch, turn, progress):
    from unittest.mock import Mock
    from scripts.llm_solver.harness.adaptive_control import llm_detector_apply as live
    from scripts.llm_solver.harness.adaptive_control import watch

    session = _session([])
    pending = {"detector_family": "loop_churn", "watch_window_start": 1,
               "watch_window_end": 5}
    session._llm_detector_pending_watch = pending
    monkeypatch.setattr(live, "_post_intervention_slots", lambda *args: [{}])
    monkeypatch.setattr(watch, "material_progress", lambda slot: progress)
    restore, escalate = Mock(), Mock()
    monkeypatch.setattr(live, "_restore_baseline_for_watch_close", restore)
    monkeypatch.setattr(live, "_select_and_apply_ranked_ladder", escalate)
    row = {}
    live._handle_pending_watch_verdict(
        session, turn, evaluate_trace_nets(session, turn), row, pending,
    )
    restore.assert_not_called()
    escalate.assert_not_called()
    transition = row["watch_transition"]
    assert transition["watch_status"] == ("closed" if turn == 5 else "continuing")
    if turn == 5:
        assert transition["episode_transition"] == "unknown"
        assert transition["closure_reason"] == "observation_allowance_exhausted"


def test_reread_advice_keeps_broader_verification_available():
    from scripts.llm_solver.harness.adaptive_control.executors import compose_user_turn_message
    message = compose_user_turn_message(SimpleNamespace(_trace_events=[]),
                                        evidence="T51:repeated inspection", rung=1,
                                        hurdle_family="reread_slump", turn=51)
    assert "broader regression test run is useful" in message
    assert "stop re-reading and re-verifying" not in message
    assert "seek new evidence" in message


@pytest.mark.parametrize("rung", range(1, 6))
def test_guard_description_reports_selection_not_new_activation(rung):
    from scripts.llm_solver.harness.adaptive_control.executors import compose_user_turn_message
    for include_guard in (True, False):
        message = compose_user_turn_message(SimpleNamespace(_trace_events=[]),
                                            evidence="T9:repeated call", rung=rung,
                                            hurdle_family="repeat_wall", turn=9,
                                            include_guard=include_guard)
        assert ("Selected guard response:" in message) is include_guard
        assert "now" not in message
        assert "will warn" not in message


@pytest.mark.parametrize("include_guard", [False, True])
def test_adaptive_advice_dates_edit_state_in_retained_history(include_guard):
    from scripts.llm_solver.harness.adaptive_control.executors import compose_user_turn_message
    session = SimpleNamespace(_trace_events=[])
    before = compose_user_turn_message(session, evidence="repeat", rung=1,
                                      hurdle_family="repeat_wall", turn=21,
                                      include_guard=include_guard)
    session._trace_events.append({"turn_number": 24, "source_write_like": True})
    after = compose_user_turn_message(session, evidence="repeat", rung=1,
                                     hurdle_family="repeat_wall", turn=36,
                                     include_guard=include_guard)
    historical = compose_user_turn_message(session, evidence="repeat", rung=1,
                                          hurdle_family="repeat_wall", turn=21,
                                          include_guard=include_guard)
    assert before == historical
    assert "observation at turn 21" in before
    assert "At that point, no source edit had been recorded." in before
    assert "observation at turn 36" in after
    assert "At that point, the last recorded source edit was at turn 24." in after
    assert "You have not made any source edits yet" not in before + after


def test_identical_repeat_plateau_does_not_prove_stalled_progress():
    ev = [_ev(t, args=f"c{t}", sha=f"s{t}") for t in range(10)]
    ev += [_ev(10, args="same", sha="X"), _ev(11, args="same", sha="X")]
    v = evaluate_trace_nets(_session(ev), 11)
    assert v.hurdle_present == "uncertain" and v.new_facts_still_appearing is None


@pytest.mark.parametrize("threshold", [0, 5, 8])
def test_repeated_full_calls_with_changing_output_cannot_select_a_hurdle(threshold):
    from scripts.llm_solver.config import Config
    count = threshold or Config.__dataclass_fields__["loop_detect_threshold"].default
    ev = [_ev(t, args="run tests", sha=f"timing-{t}", pf="pass") for t in range(count)]
    session = _session(ev)
    session.cfg.loop_detect_threshold = threshold
    assert evaluate_trace_nets(session, count - 2).hurdle_present == "uncertain"
    result = evaluate_trace_nets(session, count - 1)
    assert result.hurdle_family == ""
    assert result.hurdle_present == "uncertain"
    assert result.new_facts_still_appearing is None
    session._trace_events.append(_ev(count, args="different action", sha="new", pf="pass"))
    assert evaluate_trace_nets(session, count).hurdle_present == "uncertain"


def test_repeated_action_fact_requires_full_identity_and_no_source_write():
    from scripts.llm_solver.trace_net_facts import same_action_repeat
    events = [_ev(t, args="same", sha=str(t), pf="pass") for t in range(5)]
    events[-1].pop("action_sha256")
    assert same_action_repeat(events, 4, min_streak=5) is None
    events[-1] = _ev(4, args="same", sha="4", write=True)
    assert same_action_repeat(events, 4, min_streak=5) is None


def test_nonconsecutive_shared_failure_hash_stays_silent():
    ev = [_ev(t, args=f"c{t}", sha="DEAD", pf="fail") for t in (3, 7, 12)]
    ev = [_ev(t, args=f"x{t}", sha=f"s{t}", pf="pass") for t in range(3)] + ev
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "uncertain"


def test_unrelated_actions_with_equal_failed_output_do_not_establish_a_wall():
    ev = [_ev(t, args=f"c{t}", sha="DEAD", pf="fail") for t in range(9, 13)]
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "uncertain"


def test_repeated_request_with_changed_output_does_not_establish_reread_slump():
    args = "read a sufficiently long source path"
    ev = [_ev(0, "initial novel command", "z0", "pass"),
          _ev(1, args, "a1", "pass"), _ev(2, "novel command one", "b1", "pass"),
          _ev(5, args, "a2", "pass")]
    v = evaluate_trace_nets(_session(ev), 5)
    assert v.hurdle_present == "uncertain"
    assert v.new_facts_still_appearing is None


def test_legacy_maximum_gap_does_not_promote_changed_output_to_slump():
    args = "read a sufficiently long source path"
    ev = [_ev(t, f"novel command {t}", f"s{t}", "pass") for t in range(31)]
    ev[0] = _ev(0, args, "first", "pass")
    ev[30] = _ev(30, args, "second", "pass")

    v = evaluate_trace_nets(_session(ev), 30)

    assert v.hurdle_present == "uncertain"


def test_equal_execution_hash_alone_does_not_establish_stalled_work():
    ev = [
        _ev(8, args="first", sha="first"),
        _ev(9, args="second", sha="second"),
        _ev(10, args="same", sha="decorated-a", execution_sha="raw"),
        _ev(11, args="same", sha="decorated-b", execution_sha="raw"),
    ]
    v = evaluate_trace_nets(_session(ev), 11)
    assert v.hurdle_present == "uncertain" and v.new_facts_still_appearing is None


def test_healthy_novel_turns_stay_silent():
    ev = [_ev(t, args=f"c{t}", sha=f"s{t}", pf="pass") for t in range(12)]
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "uncertain"


def _output_recurrence_events(status):
    events = [_ev(t, args=f"c{t}", sha="repeat" if t != 2 else "other", pf="")
              for t in (1, 2, 3, 5)]
    for event in events:
        event.pop("pass_fail")
        event.update(status)
    return events


@pytest.mark.parametrize("status", [
    {}, {"pass_fail": ""}, {"pass_fail": "unknown"},
    {"outcome": "unknown"}, {"exit_status": None}, {"exit_status": "unavailable"},
])
def test_unknown_output_status_cannot_establish_passing_recurrence(status):
    events = _output_recurrence_events(status)
    verdict = evaluate_trace_nets(_session(events), 5)
    assert verdict.hurdle_present == "uncertain"
    assert not verdict.evidence_refs


@pytest.mark.parametrize("unknown_index", [0, 2, 3])
def test_each_counted_output_needs_success_evidence(unknown_index):
    from scripts.llm_solver.trace_net_facts import same_passing_output_recurrence
    events = _output_recurrence_events({"pass_fail": "pass"})
    events[unknown_index].pop("pass_fail")
    assert same_passing_output_recurrence(
        events, 3, lookback=20, min_prior=2, min_gap=2,
    ) is None
    assert evaluate_trace_nets(_session(events), 5).hurdle_present == "uncertain"


@pytest.mark.parametrize("status", [
    {"pass_fail": "pass"}, {"outcome": "ok"}, {"exit_status": 0},
    {"exit_status": "0"},
])
def test_recorded_success_retains_output_recurrence_fact(status):
    from scripts.llm_solver.trace_net_facts import same_passing_output_recurrence
    events = _output_recurrence_events(status)
    fact = same_passing_output_recurrence(events, 3, lookback=20, min_prior=2, min_gap=2)
    assert fact is not None
    assert fact.evidence_turns == (1, 3, 5)
    assert fact.occurrences == 3
    # The historical fact remains available without promoting it to a hurdle.
    verdict = evaluate_trace_nets(_session(events), 5)
    assert verdict.hurdle_present == "uncertain"


@pytest.mark.parametrize("failure", [
    {"pass_fail": "fail"}, {"outcome": "error"},
    {"exit_status": 1}, {"error_class": "timeout"},
])
def test_failure_evidence_prevents_passing_recurrence(failure):
    from scripts.llm_solver.trace_net_facts import same_passing_output_recurrence
    events = _output_recurrence_events({"pass_fail": "pass", "outcome": "ok", "exit_status": 0})
    events[-1].update(failure)
    assert same_passing_output_recurrence(events, 3, lookback=20, min_prior=2, min_gap=2) is None


def test_future_events_cannot_change_prefix_verdict():
    prefix = [_ev(t, args=f"unique {t}", sha=f"u{t}", pf="pass") for t in range(4)]
    future = [_ev(t, args="future repeat", sha="future", pf="fail") for t in range(10, 14)]

    prefix_only = evaluate_trace_nets(_session(prefix), 3)
    with_future_loaded = evaluate_trace_nets(_session(prefix + future), 3)

    assert prefix_only.hurdle_present == "uncertain"
    assert with_future_loaded == prefix_only


def test_warmup_suppresses():
    ev = [_ev(10, args="same", sha="X"), _ev(11, args="same", sha="X")]
    v = evaluate_trace_nets(_session(ev, arm_after=15), 11)
    assert v.hurdle_present == "uncertain" and v.abstain_reason == "warmup"


def test_quiet_period_retains_candidate_evidence_without_delivering_it():
    from scripts.llm_solver.harness.repeated_observations import read_observation
    ev = [_ev(turn, args="same", sha="same", pf="fail") for turn in range(1, 5)]
    for row in ev:
        row["observation_receipt"] = read_observation("same")
    session = _session(ev, arm_after=10)
    quiet = evaluate_trace_nets(session, 4)
    session.cfg.guardrails_arm_after_turn = 0
    active = evaluate_trace_nets(session, 4)
    assert active.hurdle_present == "uncertain"
    assert quiet.hurdle_present == "uncertain"
    assert quiet.hurdle_family == ""
    assert quiet.abstain_reason == "warmup"
    assert quiet.evidence_refs == active.evidence_refs
    assert quiet.evidence_refs
    assert quiet.rejected_families == []
    assert quiet.rejected_reason == "stalled_progress_not_established"


def test_quiet_period_records_insufficient_evidence_too():
    quiet = evaluate_trace_nets(_session([], arm_after=10), 4)
    active = evaluate_trace_nets(_session([], arm_after=10), 11)
    assert quiet.hurdle_present == active.hurdle_present == "uncertain"
    assert quiet.abstain_reason == "warmup"
    assert quiet.rejected_reason == active.abstain_reason
    assert quiet.evidence_refs == []
    assert quiet.rejected_families == []


def _reread_events():
    """A re-read at gap=4 that fires under the frozen default (min_gap=3)."""
    args = "read a sufficiently long source path"
    return [_ev(0, args, "a0", "pass"),
            _ev(1, "novel one", "b1", "pass"),
            _ev(2, "novel two", "b2", "pass"),
            _ev(3, "novel three", "b3", "pass"),
            _ev(4, args, "a1", "pass")]


def test_legacy_gap_setting_does_not_supply_observation_evidence():
    ev = _reread_events()
    cfg = SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=6)
    sess = SimpleNamespace(cfg=cfg, _trace_events=ev)
    assert evaluate_trace_nets(sess, 4).hurdle_present == "uncertain"


def test_legacy_gap_default_does_not_authorize_a_hurdle():
    ev = _reread_events()
    for cfg in (SimpleNamespace(guardrails_arm_after_turn=0),
                SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=0),
                SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=-5)):
        sess = SimpleNamespace(cfg=cfg, _trace_events=ev)
        v = evaluate_trace_nets(sess, 4)
        assert v.hurdle_present == "uncertain"


def test_language_does_not_promote_repetition_to_stalled_progress():
    """Equal command/output fields have the same limits in every language."""
    commands = {
        "pytest": "python -m pytest tests/test_x.py -q",
        "go":     "go test ./... -run TestX",
        "cargo":  "cargo test --package foo mymod::test_x",
        "jest":   "npx jest src/x.test.ts -t 'does the thing'",
    }
    for lang, cmd in commands.items():
        # Identical failure output is not evidence that progress is impossible.
        ev = [_ev(t, args=cmd, sha="OUT", pf="fail", execution_sha="DEAD")
              for t in range(9, 13)]
        v = evaluate_trace_nets(_session(ev), 12)
        assert v.hurdle_present == "uncertain" and v.new_facts_still_appearing is None, lang


def test_live_hook_applies_via_trace_nets_backend(tmp_path):
    """Full loop: nets fire -> family lookup -> candidate applied."""
    import sys
    sys.path.insert(0, "tests")
    from test_adaptive_control_llm_detector import (
        _llm_atlas, _write_family_lookup, _live_detector_cfg, _event,
    )
    from types import SimpleNamespace
    from scripts.llm_solver.harness.adaptive_control.llm_detector import (
        maybe_run_llm_hurdle_detector,
    )

    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled = true\n")
    lookup_path = _write_family_lookup(tmp_path / "family_lookup.tsv",
                                       candidate_config_path=candidate)
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        backend="trace_nets",
    )

    # repeat-wall shaped trace: identical (args, sha) pair at the tail
    events = [_event(t) for t in range(4)]
    for e, t in zip(events, range(4)):
        e["args_summary"] = "probe X"; e["output_sha256"] = "SAME"
        e["pass_fail"] = "fail"; e["turn_number"] = t
    session = SimpleNamespace(
        cfg=cfg,
        client=None,  # trace_nets must never touch the client
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=events,
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )
    out = maybe_run_llm_hurdle_detector(session, turn=3)
    assert out is not None


def test_trace_nets_cannot_open_an_episode_from_repeated_actions(tmp_path):
    """The live hook preserves configuration across unchanged and changed results."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, "tests")
    from test_adaptive_control_llm_detector import (
        _event,
        _live_detector_cfg,
        _llm_atlas,
        _write_family_lookup,
    )
    from scripts.llm_solver.harness.adaptive_control.llm_detector import (
        maybe_run_llm_hurdle_detector,
    )

    atlas_path = _llm_atlas(tmp_path / "hurdle_dictionary.llm.v1.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled = true\n")
    lookup_path = _write_family_lookup(
        tmp_path / "family_lookup.tsv",
        candidate_config_path=candidate,
    )
    baseline_path = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(
        tmp_path,
        atlas_path,
        lookup_path,
        baseline_path,
        backend="trace_nets",
        max_interventions=2,
    )

    events = [_event(t) for t in range(4)]
    for event in events:
        event["args_summary"] = "probe X"
        event["action_sha256"] = action_metadata("bash", {"cmd": "probe X"})["action_sha256"]
        event["output_sha256"] = "SAME"
        event["pass_fail"] = "fail"
    session = SimpleNamespace(
        cfg=cfg,
        client=None,
        _trace_path=tmp_path / ".trace.jsonl",
        _trace_events=events,
        adaptive_control_baseline_config_paths=(str(baseline_path),),
        attempt_id="attempt-1",
        instance_id="repo__task-1",
    )

    for turn in range(3, 10):
        if turn > 3:
            session._trace_events.append(_ev(turn, args="probe X",
                                             sha=f"changed-{turn}", pf="pass"))
        observed = maybe_run_llm_hurdle_detector(session, turn=turn)
        assert observed is not None
        assert observed.get("intervention_apply", {}).get("apply_status") != "applied"
        assert session.cfg.loop_detect_enabled is False
    machine = getattr(session, "_adaptive_control_episode_machine", None)
    assert machine is None or machine.episodes_opened == 0
