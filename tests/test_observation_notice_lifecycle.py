"""Notice evidence and suppression follow the restored conversation branch."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import _ac_bootstrap  # noqa: F401
from llm_solver.harness.action_metadata import action_metadata
from llm_solver.harness.adaptive_control.observation_notice import (
    repeated_observation, record_loop_observation_notice,
)
from llm_solver.harness.tools import dispatch
from llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
from llm_solver.harness.turn_snapshots import capture_conversation_snapshot
from test_conversation_rewind import _session
from test_completed_observation_notice import row
from llm_solver.harness.repeated_observations import read_observation


@pytest.fixture
def rig(tmp_path):
    work, artifacts = tmp_path / "work", tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()
    (work / "x").write_text("same information\n")
    (work / "y").write_text("different information\n")
    return _session(work, artifacts, config_overrides={
        "loop_detect_enabled": True, "guardrails_arm_after_turn": 0,
        "max_turns": 8, "llm_hurdle_detector_enabled": False,
    })


def read(rig, path, turn, *, snapshot=False):
    current, store = rig
    args, call_id = {"path": path}, f"read-{turn}"
    current._current_turn = turn
    facts = {}
    result = dispatch("read", args, cwd=current.cwd, cfg=current.cfg,
                      execution_metadata=facts, tool_call_id=call_id)
    current.context.add_assistant({"role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "read", "arguments": json.dumps(args)}}]})
    current.context.add_tool_result(call_id, str(result), tool_name="read")
    current._emit("tool_call", session_number=current._session_number,
        turn_number=turn, tool_name="read", tool_call_id=call_id,
        args_summary=f"path={path!r}", **action_metadata("read", args),
        **build_tool_call_trace_fields(current, tool_name="read", args_summary=f"path={path!r}",
            result=result, turn=turn, gate_blocked=False, execution_metadata=facts))
    if snapshot:
        store.capture(turn)
        capture_conversation_snapshot(current, turn)


def test_discarded_read_cannot_supply_a_match(rig):
    current, _ = rig
    read(rig, "y", 0, snapshot=True)
    read(rig, "x", 1, snapshot=True)
    current._current_turn = 2
    current.rewind_to(0, reason="discard inspection")
    read(rig, "x", 3)
    assert repeated_observation(current, 3) is None
    record_loop_observation_notice(current, 3)
    assert current._deliver_pending_user_turn_injections() == 0
    assert any(event.get("tool_call_id") == "read-1" for event in current._trace_events)


def test_discarded_change_cannot_hide_active_match():
    rows = [row(0), row(1, read_observation("changed")),
            {"event": "rewind", "session_number": 1, "turn_number": 1, "to_turn": 0}, row(2)]
    current = SimpleNamespace(_trace_events=rows, _session_number=1)
    assert repeated_observation(current, 2)["prior_turn"] == 0


@pytest.mark.parametrize("target,deliver", [(0, True), (2, True), (0, False)])
def test_notice_suppression_follows_retained_deliveries(rig, target, deliver):
    current, store = rig
    read(rig, "y", 0, snapshot=True)
    read(rig, "x", 1)
    read(rig, "x", 2)
    record_loop_observation_notice(current, 2)
    if deliver:
        assert current._deliver_pending_user_turn_injections() == 1
    store.capture(2)
    capture_conversation_snapshot(current, 2)
    current._current_turn = 3
    current.rewind_to(target, reason="restore selected notice history")
    assert not current._pending_user_turn_injections
    read(rig, "x", 4)
    read(rig, "x", 5)
    record_loop_observation_notice(current, 5)
    assert current._deliver_pending_user_turn_injections() == (1 if target == 0 else 0)


def test_delivery_ledger_records_user_turn_phase(rig):
    current, _ = rig
    read(rig, "x", 0)
    read(rig, "x", 1)
    record_loop_observation_notice(current, 1)
    with patch("llm_solver.harness.savings.get_ledger") as ledger:
        assert current._deliver_pending_user_turn_injections() == 1
        assert ledger.return_value.record_transform.call_args.kwargs["ctx"]["delivery"] == "user_turn"
    assert current._trace_events[-1]["delivery"] == "user_turn"
