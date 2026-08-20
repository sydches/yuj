"""trace_nets detector backend: mechanical soft-tier nets in the live slot."""
from types import SimpleNamespace

from scripts.llm_solver.harness.adaptive_control.trace_nets_detector import (
    evaluate_trace_nets,
)


def _ev(turn, args="cmd", sha="s0", pf="fail", write="False", execution_sha=""):
    row = {"event": "tool_call", "turn_number": turn, "args_summary": args,
           "output_sha256": sha, "pass_fail": pf, "source_write_like": write}
    if execution_sha:
        row["execution_output_sha256"] = execution_sha
    return row


def _session(events, arm_after=0):
    return SimpleNamespace(cfg=SimpleNamespace(guardrails_arm_after_turn=arm_after),
                           _trace_events=events)


def test_identical_repeat_plateau_fires_repeat_wall():
    ev = [_ev(t, args=f"c{t}", sha=f"s{t}") for t in range(10)]
    ev += [_ev(10, args="same", sha="X"), _ev(11, args="same", sha="X")]
    v = evaluate_trace_nets(_session(ev), 11)
    assert v.hurdle_present == "yes" and v.hurdle_family == "repeat_wall"


def test_nonconsecutive_shared_failure_hash_stays_silent():
    ev = [_ev(t, args=f"c{t}", sha="DEAD", pf="fail") for t in (3, 7, 12)]
    ev = [_ev(t, args=f"x{t}", sha=f"s{t}", pf="pass") for t in range(3)] + ev
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "no"


def test_consecutive_failed_output_repeat_fires():
    ev = [_ev(t, args=f"c{t}", sha="DEAD", pf="fail") for t in range(9, 13)]
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "yes" and v.hurdle_family == "repeat_wall"


def test_rereads_fire_slump():
    args = "read a sufficiently long source path"
    ev = [_ev(0, "initial novel command", "z0", "pass"),
          _ev(1, args, "a1", "pass"), _ev(2, "novel command one", "b1", "pass"),
          _ev(5, args, "a2", "pass")]
    v = evaluate_trace_nets(_session(ev), 5)
    assert v.hurdle_present == "yes" and v.hurdle_family == "reread_slump"


def test_reread_at_maximum_gap_remains_in_live_window():
    args = "read a sufficiently long source path"
    ev = [_ev(t, f"novel command {t}", f"s{t}", "pass") for t in range(31)]
    ev[0] = _ev(0, args, "first", "pass")
    ev[30] = _ev(30, args, "second", "pass")

    v = evaluate_trace_nets(_session(ev), 30)

    assert v.hurdle_present == "yes" and v.hurdle_family == "reread_slump"


def test_execution_hash_ignores_decorated_output_hash():
    ev = [
        _ev(8, args="first", sha="first"),
        _ev(9, args="second", sha="second"),
        _ev(10, args="same", sha="decorated-a", execution_sha="raw"),
        _ev(11, args="same", sha="decorated-b", execution_sha="raw"),
    ]
    v = evaluate_trace_nets(_session(ev), 11)
    assert v.hurdle_present == "yes" and v.hurdle_family == "repeat_wall"


def test_healthy_novel_turns_stay_silent():
    ev = [_ev(t, args=f"c{t}", sha=f"s{t}", pf="pass") for t in range(12)]
    v = evaluate_trace_nets(_session(ev), 12)
    assert v.hurdle_present == "no"


def test_future_events_cannot_change_prefix_verdict():
    prefix = [_ev(t, args=f"unique {t}", sha=f"u{t}", pf="pass") for t in range(4)]
    future = [_ev(t, args="future repeat", sha="future", pf="fail") for t in range(10, 14)]

    prefix_only = evaluate_trace_nets(_session(prefix), 3)
    with_future_loaded = evaluate_trace_nets(_session(prefix + future), 3)

    assert prefix_only.hurdle_present == "no"
    assert with_future_loaded == prefix_only


def test_warmup_suppresses():
    ev = [_ev(10, args="same", sha="X"), _ev(11, args="same", sha="X")]
    v = evaluate_trace_nets(_session(ev, arm_after=15), 11)
    assert v.hurdle_present == "no" and v.abstain_reason == "warmup"


def _reread_events():
    """A re-read at gap=4 that fires under the frozen default (min_gap=3)."""
    args = "read a sufficiently long source path"
    return [_ev(0, args, "a0", "pass"),
            _ev(1, "novel one", "b1", "pass"),
            _ev(2, "novel two", "b2", "pass"),
            _ev(3, "novel three", "b3", "pass"),
            _ev(4, args, "a1", "pass")]


def test_cfg_threshold_override_silences_reread():
    """A run may raise reread_min_gap so its normal rhythm stops firing."""
    ev = _reread_events()
    cfg = SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=6)
    sess = SimpleNamespace(cfg=cfg, _trace_events=ev)
    assert evaluate_trace_nets(sess, 4).hurdle_present == "no"


def test_cfg_threshold_default_when_unset_or_nonpositive():
    """An unset or non-positive threshold uses the default."""
    ev = _reread_events()
    for cfg in (SimpleNamespace(guardrails_arm_after_turn=0),
                SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=0),
                SimpleNamespace(guardrails_arm_after_turn=0, trace_nets_reread_min_gap=-5)):
        sess = SimpleNamespace(cfg=cfg, _trace_events=ev)
        v = evaluate_trace_nets(sess, 4)
        assert v.hurdle_present == "yes" and v.hurdle_family == "reread_slump"


def test_detector_is_language_agnostic():
    """The nets fire on the same field-shape regardless of surface language.

    The detector reads only sha / pass_fail / args_summary / source_write
    fields — never Python-specific text — so a failing-test wall built from
    go, rust, and js commands fires identically to a pytest one. This is
    the multilingual guarantee: give the fields the right values (the
    resolved language quirk does that upstream) and the detector is neutral.
    """
    commands = {
        "pytest": "python -m pytest tests/test_x.py -q",
        "go":     "go test ./... -run TestX",
        "cargo":  "cargo test --package foo mymod::test_x",
        "jest":   "npx jest src/x.test.ts -t 'does the thing'",
    }
    for lang, cmd in commands.items():
        # 4 consecutive turns, identical failing execution-sha -> repeat_wall
        ev = [_ev(t, args=cmd, sha="OUT", pf="fail", execution_sha="DEAD")
              for t in range(9, 13)]
        v = evaluate_trace_nets(_session(ev), 12)
        assert v.hurdle_present == "yes" and v.hurdle_family == "repeat_wall", lang


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


def test_trace_nets_watch_restores_baseline_on_first_unlock_turn(tmp_path):
    """A five-turn budget closes early when the live trace proves progress."""
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
    )

    events = [_event(t) for t in range(4)]
    for event in events:
        event["args_summary"] = "probe X"
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

    fired = maybe_run_llm_hurdle_detector(session, turn=3)
    assert fired["intervention_apply"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is True

    session._trace_events.append(_event(
        4,
        args_summary="edit src/module.py",
        output_sha256="CHANGED",
        pass_fail="pass",
        write_like=True,
        source_write_like=True,
        source_write_paths=["src/module.py"],
    ))
    cleared = maybe_run_llm_hurdle_detector(session, turn=4)

    assert cleared["watch_transition"]["episode_transition"] == "cleared_to_progress"
    assert cleared["watch_transition"]["budget_exhausted"] is False
    assert cleared["baseline_restore"]["apply_status"] == "applied"
    assert session.cfg.loop_detect_enabled is False
