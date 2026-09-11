"""Repeated information must come from tool receipts, not benchmark turn gaps."""
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness.repeated_observations import read_observation, process_observation
from scripts.llm_solver.harness.adaptive_control.observation_notice import (
    repeated_observation, record_observation_notice, record_loop_observation_notice,
)


def row(turn, receipt=None, **changes):
    return {"event": "tool_call", "turn_number": turn, "session_number": 1,
            "tool_call_id": str(turn),
            **action_metadata("read", {"path": "x"}),
            "observation_receipt": read_observation("same") if receipt is None else receipt,
            **changes}


def session(rows, **changes):
    return SimpleNamespace(_trace_events=rows, _session_number=1,
        cfg=SimpleNamespace(guardrails_arm_after_turn=0, max_turns=100, **changes))


@pytest.mark.parametrize("gap", [1, 3, 7, 8, 31])
def test_native_repetition_does_not_depend_on_gap_or_summary_length(gap):
    current = session([row(0, args_summary="x"), row(gap, args_summary="a" * 200)])
    fact = repeated_observation(current, gap)
    assert fact["prior_turn"] == 0
    assert fact["current_turn"] == gap
    assert fact["progress"] == "unknown"


@pytest.mark.parametrize("replacement", [
    {"observation_receipt": None}, {"observation_receipt": read_observation("changed")},
    {"observation_receipt": process_observation("p", True, 0, 1, None)},
    {"gate_blocked": True}, {"executed": False}, {"session_number": 2}, {"action_sha256": "unknown"},
])
def test_unknown_changed_pending_blocked_and_other_session_results_do_not_match(replacement):
    last = row(8)
    last.update(replacement)
    assert repeated_observation(session([row(0), last]), 8) is None


def test_nearest_changed_observation_breaks_comparison_with_older_match():
    current = session([row(0), row(3, read_observation("changed")), row(8)])
    assert repeated_observation(current, 8) is None


def test_future_or_stale_latest_result_cannot_trigger_notice():
    current = session([row(0), row(8)])
    assert repeated_observation(current, 3) is None
    assert repeated_observation(current, 9) is None


def test_completed_process_identity_cursor_and_status_are_bound():
    prior = row(0, process_observation("p", False, 4, 4, 0))
    last = row(8, process_observation("p", False, 4, 4, 0))
    assert repeated_observation(session([prior, last]), 8)
    last["observation_receipt"] = process_observation("p", False, 4, 8, 0)
    assert repeated_observation(session([prior, last]), 8) is None


def test_guard_and_detector_share_notice_delivery_without_duplicate_injection():
    current = session([row(0), row(8)], loop_detect_enabled=True)
    current._emit = MagicMock()
    current._queue_user_turn_injection = MagicMock(return_value=True)
    record_loop_observation_notice(current, 8)
    detector = {}
    record_observation_notice(current, 8, detector)
    assert current._queue_user_turn_injection.call_count == 1
    assert current._emit.call_args.kwargs["delivery"] == "queued"
    assert detector["repeated_observation"]["delivery"] == "already_notified"


@pytest.mark.parametrize("field, changed", [("sha256", "b" * 64), ("start_line", 2), ("path", "/other")])
def test_same_rendered_text_does_not_hide_changed_inspection_binding(field, changed):
    inspection = {"namespace": "local_filesystem", "path": "/x", "sha256": "a" * 64,
                  "start_line": 1, "line_count": 1, "total_lines": 4}
    prior, last = row(0, inspection_evidence=inspection), row(8, inspection_evidence=dict(inspection))
    assert repeated_observation(session([prior, last]), 8)["file_revision_bound"]
    last["inspection_evidence"][field] = changed
    assert repeated_observation(session([prior, last]), 8) is None


def test_quiet_observation_does_not_consume_notice_and_delivery_is_deduplicated():
    current = session([row(0), row(8)])
    current.cfg.guardrails_arm_after_turn = 8
    current._queue_user_turn_injection = MagicMock(return_value=True)
    quiet = {}
    record_observation_notice(current, 8, quiet)
    assert quiet["repeated_observation"]["delivery"] == "withheld"
    current._trace_events.append(row(9))
    active = {}
    record_observation_notice(current, 9, active)
    assert active["repeated_observation"]["delivery"] == "queued"
    current._trace_events.append(row(10))
    repeated = {}
    record_observation_notice(current, 10, repeated)
    assert repeated["repeated_observation"]["delivery"] == "already_notified"
    assert current._queue_user_turn_injection.call_count == 1


@pytest.mark.parametrize("condition", ["disabled", "last_turn", "exhausted_time"])
@pytest.mark.parametrize("offset", [0, 7])
def test_notice_respects_transformation_and_remaining_resource_controls(condition, offset, monkeypatch):
    from scripts.llm_solver.harness.adaptive_control import observation_notice
    current = session([row(offset), row(offset + 8)])
    current._turn_start_offset = offset
    current._queue_user_turn_injection = MagicMock(return_value=True)
    if condition == "disabled":
        current.cfg.transformations_explicit = True
        current.cfg.detector_activated_guardrails = False
    elif condition == "last_turn":
        current.cfg.max_turns = 9
    else:
        monkeypatch.setattr(observation_notice, "remaining_run_seconds", lambda: 0)
    result = {}
    record_observation_notice(current, offset + 8, result)
    assert result["repeated_observation"]["delivery"] == "withheld"
    current._queue_user_turn_injection.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("offset", [0, 7])
def test_actual_session_delivers_only_receipt_grounded_notice_without_extra_model_calls(tmp_path, changed, offset):
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage
    from test_adaptive_control_llm_detector import _llm_atlas, _live_detector_cfg, _write_family_lookup
    atlas = _llm_atlas(tmp_path / "atlas.tsv")
    candidate = tmp_path / "candidate.toml"
    candidate.write_text("[loop]\nloop_detect_enabled=true\n")
    lookup = _write_family_lookup(tmp_path / "lookup.tsv", candidate_config_path=candidate)
    baseline = tmp_path / "baseline.toml"
    cfg = _live_detector_cfg(tmp_path, atlas, lookup, baseline, backend="trace_nets")
    cfg = replace(cfg, max_turns=4, tools_output_dedup_enabled=False,
                  duplicate_guard_enabled=False, require_intent=False)
    client = MagicMock()
    calls = []
    def chat(messages, tools, **kwargs):
        turn = len(calls)
        calls.append(str(messages))
        if turn == 2 and changed:
            (tmp_path / "x").write_text("changed")
        tool_calls = [] if turn == 3 else [ToolCall(id=str(turn), name="read",
            arguments={"path": "y" if turn == 1 else "x"})]
        return TurnResult(content="finished" if turn == 3 else None, tool_calls=tool_calls,
                          finish_reason="stop" if turn == 3 else "tool_calls",
                          usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    (tmp_path / "x").write_text("same")
    (tmp_path / "y").write_text("different")
    trace = io.StringIO()
    current = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    current._turn_start_offset = offset
    result = current.run()
    assert result.turns == 4
    assert len(calls) == 4
    assert ("same completed observation" in calls[-1]) is (not changed)
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    notices = [event for event in events if event.get("mechanism") == "completed_observation_notice"]
    assert len(notices) == (0 if changed else 1)
