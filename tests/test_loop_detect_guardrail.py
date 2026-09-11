"""Request repetition stays diagnostic; native observations drive notices."""
import io
import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from tests._config_helpers import make_config
from scripts.llm_solver.harness.guardrails import Action, init_guardrail_state, loop_detect
from scripts.llm_solver.harness._loop.run_step import _run_post_turn_hooks
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


@pytest.mark.parametrize("threshold", [0, 1, 5, 8])
def test_request_signatures_never_warn_or_end_even_after_legacy_warning(threshold):
    cfg = make_config(loop_detect_enabled=True, loop_detect_threshold=threshold)
    state = init_guardrail_state(cfg)
    state.loop_detect_warned = True
    for index in range(12):
        decision = loop_detect(state, cfg, tool_calls_sig=("same",),
                               allow_intervention=index > 3)
        assert decision.action == Action.PASS
        assert state.loop_detect_streak == index + 1
        assert not state.loop_detect_warned
    loop_detect(state, cfg, tool_calls_sig=("changed",))
    assert state.loop_detect_streak == 1
    cfg = replace(cfg, loop_detect_enabled=False)
    loop_detect(state, cfg, tool_calls_sig=("same",))
    assert state.loop_detect_streak == 0


@pytest.mark.parametrize("mode", ["stable", "changing", "respond_to_notice"])
@pytest.mark.parametrize("arm_after", [0, 3])
def test_actual_reads_and_recovery_respect_observations_and_budget(tmp_path, mode, arm_after):
    cfg = make_config(max_turns=8, loop_detect_enabled=True, loop_detect_threshold=2,
                      duplicate_guard_enabled=False, adaptive_control_enabled=False,
                      guardrails_arm_after_turn=arm_after, require_intent=False,
                      tools_output_dedup_enabled=False)
    client = MagicMock()
    calls = []
    def chat(messages, tools, **kwargs):
        turn = len(calls)
        calls.append(str(messages))
        content = str(turn) if mode == "changing" else "same"
        if mode == "respond_to_notice" and "same completed observation" in str(messages):
            content = "new evidence after notice"
        (tmp_path / "status.txt").write_text(content)
        return TurnResult(content=None, tool_calls=[ToolCall(id=str(turn), name="read",
            arguments={"path": "status.txt"})], finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    session._llm_detector_pending_watch = {"watch_window_end": 99}
    result = session.run()
    assert result.finish_reason == "max_turns"
    assert len(calls) == 8
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    reads = [row for row in rows if row["event"] == "tool_call" and row["tool_name"] == "read"]
    assert len(reads) == 8
    assert len({row["inspection_evidence"]["sha256"] for row in reads}) == {
        "changing": 8, "stable": 1, "respond_to_notice": 2,
    }[mode]
    notices = [row for row in rows if row["event"] == "completed_observation_notice"
               and row["delivery"] == "queued"]
    assert len(notices) == {"changing": 0, "stable": 1, "respond_to_notice": 2}[mode]
    if mode != "changing":
        assert notices[0]["current_turn"] == max(1, arm_after + 1)
        assert notices[0]["progress"] == "unknown"
        assert "same completed observation" in calls[-1]
        assert "stalled progress;" in calls[-1]


def test_completed_repetition_allowance_still_stops_at_its_declared_budget(tmp_path):
    cfg = make_config(max_turns=8, loop_detect_enabled=True, loop_detect_threshold=1,
                      duplicate_guard_enabled=True, duplicate_abort=3,
                      adaptive_control_enabled=False, guardrails_arm_after_turn=0,
                      require_intent=False)
    (tmp_path / "same.txt").write_text("same")
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="read", name="read", arguments={"path": "same.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    result = Session(cfg, client, "system", "task", str(tmp_path)).run()
    assert result.finish_reason == "duplicate_abort"
    assert client.chat.call_count == 3


def test_blocked_turn_runs_all_post_turn_hooks():
    calls = []

    class Session:
        def _maybe_emit_harness_observation(self, turn):
            calls.append(("observation", turn))

        def _maybe_run_llm_hurdle_detector(self, turn):
            calls.append(("detector", turn))

        def _maybe_switch_adaptive_phase(self, turn):
            calls.append(("phase", turn))

        def _maybe_run_advisor(self, turn):
            calls.append(("advisor", turn))

    _run_post_turn_hooks(Session(), 36)

    assert calls == [
        ("observation", 36),
        ("detector", 36),
        ("phase", 36),
        ("advisor", 36),
    ]
