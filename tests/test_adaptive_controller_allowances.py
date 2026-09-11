"""Declared controller allowances survive applications and segment boundaries."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import _ac_bootstrap  # noqa: F401
from llm_solver.harness.adaptive_control import episode, persistence
from llm_solver.harness.adaptive_control.llm_detector_apply import _watch_window, _episode_caps_for_detector, _post_intervention_slots
from llm_solver.harness.adaptive_control.llm_detector_runtime import maybe_run_llm_hurdle_detector
from llm_solver.harness import time_budget
from llm_solver.harness.loop import Session
from test_adaptive_control_llm_detector import (
    _llm_atlas, _write_family_lookup, _live_detector_cfg,
    _FakeDetectorClient, _response, _event,
)


def session(tmp_path, **changes):
    atlas = _llm_atlas(tmp_path / "atlas.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled = true\n")
    lookup = _write_family_lookup(tmp_path / "lookup.tsv", candidate_config_path=candidate)
    baseline = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(tmp_path, atlas, lookup, baseline,
                             max_interventions=2, cooldown_after_apply_slots=9)
    client = _FakeDetectorClient(_response(hurdle_present="yes", hurdle_family="loop_churn",
                                           confidence="high", evidence_refs=["T24"]))
    return SimpleNamespace(cfg=replace(cfg, **changes), cwd=str(tmp_path), client=client,
        _trace_path=tmp_path / ".trace.jsonl", _trace_events=[_event(i) for i in range(25)],
        adaptive_control_baseline_config_paths=(str(baseline),),
        attempt_id="fixture", instance_id="fixture")


def test_real_application_saves_updated_counters_watch_and_remaining_cooldown(tmp_path):
    current = session(tmp_path, adaptive_control_delivery="stop_resume")
    applied = maybe_run_llm_hurdle_detector(current, 24)
    assert applied["intervention_apply"]["apply_status"] == "applied"
    assert applied["intervention_selection"]["resolved_episode_caps"]["max_interventions_per_attempt"] == 2
    resumed = SimpleNamespace(cwd=str(tmp_path), cfg=replace(current.cfg, max_turns=2))
    assert persistence.load_state(resumed)
    machine = episode.machine(resumed)
    assert machine.interventions_total == machine.current.attempt_index == machine.episodes_opened == 1
    assert len(machine.current.applied_intervention_ids) == 1
    assert resumed._llm_detector_pending_watch["watch_window_start"] == 0
    assert resumed._llm_detector_pending_watch["watch_window_end"] == 1
    assert machine.cooldown_until == 8
    assert episode.plan_apply(machine, _episode_caps_for_detector(resumed.cfg),
                              machine.current.online_signal_id, 0).block_reason == episode.COOLDOWN_ACTIVE
    zero = replace(resumed.cfg, adaptive_control_max_interventions_per_attempt=0)
    assert episode.plan_apply(machine, _episode_caps_for_detector(zero), "later", 8).block_reason == episode.PER_ATTEMPT_EXHAUSTED


@pytest.mark.parametrize("turn,limit,cadence,cooldown,verdict", [
    (0, 2, 3, 9, "uncertain"), (0, 6, 1, 0, "yes"), (2, 6, 3, 9, "yes"),
])
def test_fresh_session_restores_watch_before_first_routing(tmp_path, turn, limit, cadence, cooldown, verdict):
    current = session(tmp_path, adaptive_control_delivery="stop_resume",
                      adaptive_control_cooldown_after_apply_slots=cooldown)
    assert maybe_run_llm_hurdle_detector(current, 24)["intervention_apply"]["apply_status"] == "applied"
    cfg = replace(current.cfg, max_turns=limit, llm_hurdle_detector_cadence_turns=cadence,
                  loop_detect_enabled=True)
    resumed = Session(cfg, MagicMock(), "system", "task", str(tmp_path),
                      adaptive_control_baseline_config_paths=current.adaptive_control_baseline_config_paths)
    resumed.client = _FakeDetectorClient(_response(
        hurdle_present=verdict, hurdle_family="loop_churn" if verdict == "yes" else "",
        confidence="high" if verdict == "yes" else "low", evidence_refs=[f"T{turn}"]))
    resumed._trace_events = [_event(i) for i in range(turn + 1)]
    with patch.object(persistence, "load_state", wraps=persistence.load_state) as load:
        row = resumed._maybe_run_llm_hurdle_detector(turn)
        assert row is not None
        assert row["watch_transition"]["watch_status"] == "continuing"
        assert "intervention_selection" not in row
        machine = episode.machine(resumed)
        assert machine.interventions_total == 1
        assert machine.current.attempt_index == 1
        assert resumed._llm_detector_pending_watch["watch_window_start"] == 0
        assert resumed._llm_detector_pending_watch["watch_window_end"] < limit
        resumed._trace_events.append(_event(turn + 1))
        resumed._maybe_run_llm_hurdle_detector(turn + 1)
        assert load.call_count == 1


@pytest.mark.parametrize("disabled", ["llm_hurdle_detector_enabled", "adaptive_control_enabled"])
def test_disabled_controller_does_not_consume_saved_state(tmp_path, disabled):
    current = session(tmp_path, adaptive_control_delivery="stop_resume")
    assert maybe_run_llm_hurdle_detector(current, 24)["intervention_apply"]["apply_status"] == "applied"
    resumed = Session(replace(current.cfg, **{disabled: False}), MagicMock(), "system", "task", str(tmp_path),
                      adaptive_control_baseline_config_paths=current.adaptive_control_baseline_config_paths)
    resumed.client = _FakeDetectorClient(_response(hurdle_present="yes", hurdle_family="loop_churn",
                                                  confidence="high", evidence_refs=["T0"]))
    resumed._trace_events = [_event(0)]
    with patch.object(persistence, "load_state", wraps=persistence.load_state) as load:
        resumed._maybe_run_llm_hurdle_detector(0)
        load.assert_not_called()


def test_failed_state_save_cancels_stop_and_does_not_consume_application(tmp_path, monkeypatch):
    current = session(tmp_path, adaptive_control_delivery="stop_resume")
    monkeypatch.setattr(persistence, "save_state", lambda session: False)
    row = maybe_run_llm_hurdle_detector(current, 24)
    assert row["intervention_apply"]["blocked_reason"] == "controller_state_write_failed"
    assert not current._adaptive_stop_requested
    assert current._llm_detector_pending_watch is None
    assert episode.machine(current).interventions_total == 0
    assert episode.machine(current).episodes_opened == 0


def test_resumed_watch_includes_observations_from_turn_zero():
    current = SimpleNamespace(_trace_events=[_event(i) for i in range(3)])
    slots = _post_intervention_slots(current, {"watch_window_start": 0}, 2)
    assert [slot["slot_idx"] for slot in slots] == [0, 1, 2]


@pytest.mark.parametrize("turn, limit, width, expected", [
    (10, 12, 5, (11, 11)), (11, 12, 5, (12, 11)),
    (2, 12, 0, (3, 2)), (2, 12, 5, (3, 7)),
])
def test_watch_allowance_is_capped_by_remaining_declared_turns(turn, limit, width, expected):
    cfg = SimpleNamespace(max_turns=limit, adaptive_control_watch_window_turns=width)
    assert _watch_window(turn, cfg) == expected


@pytest.mark.parametrize("width", [0, 5])
def test_no_remaining_observations_blocks_actual_application(tmp_path, width):
    current = session(tmp_path, max_turns=25, adaptive_control_watch_window_turns=width)
    row = maybe_run_llm_hurdle_detector(current, 24)
    assert row["intervention_selection"]["selection_blocked_reason"] == "observation_budget_exhausted"
    assert episode.machine(current).interventions_total == 0
    assert not current.cfg.loop_detect_enabled


def test_exhausted_run_time_blocks_detector_request_before_application(tmp_path, monkeypatch):
    current = session(tmp_path)
    now = [0.0]
    monkeypatch.setattr(time_budget.time, "monotonic", lambda: now[0])
    with time_budget.run_time_budget(1):
        now[0] = 2
        row = maybe_run_llm_hurdle_detector(current, 24)
    assert "intervention_selection" not in row
    assert row["detector_request_allowance"]["run_remaining_seconds"] == 0
    assert row["detector_request_allowance"]["status"] == "exhausted"
    assert row["detector_error"].startswith("BudgetExhausted:")
    assert current.client.calls == []
    assert episode.machine(current).interventions_total == 0
