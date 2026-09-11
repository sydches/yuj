"""Consumed loop iterations are local counts, not restored turn labels."""
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage


@pytest.mark.parametrize("offset", [0, 7])
@pytest.mark.parametrize("exit_kind", ["stop", "no_tool_call", "length", "error", "max_turns"])
def test_result_counts_entered_iterations(tmp_path, offset, exit_kind):
    cfg = make_config(
        max_turns=3,
        allow_implicit_done=exit_kind != "no_tool_call",
        duplicate_guard_enabled=False,
        loop_detect_enabled=False,
    )
    tools = ([ToolCall(id="read-1", name="read", arguments={"path": "app.py"})]
             if exit_kind == "max_turns" else [])
    response = TurnResult(
        content="Continue" if tools else "Finished",
        tool_calls=tools,
        finish_reason="length" if exit_kind == "length" else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    client = MagicMock()
    client.build_assistant_message.return_value = {"role": "assistant", "content": "response"}
    session = Session(cfg, client, "system", "task", str(tmp_path))
    session._turn_start_offset = offset
    with (
        patch.object(session, "_get_server_ctx", return_value=0),
        patch.object(session, "_chat_with_retry", return_value=None if exit_kind == "error" else response) as chat,
        patch("llm_solver.harness.loop.dispatch", return_value="source"),
        patch.object(session, "_emit") as emit,
    ):
        result = session.run()
    expected = 3 if exit_kind == "max_turns" else 1
    assert result.finish_reason == exit_kind
    assert result.turns == expected
    assert chat.call_count == expected
    labels = [call.kwargs["turn_number"] for call in emit.call_args_list
              if call.args and call.args[0] == "turn"]
    assert labels == ([] if exit_kind == "error" else list(range(offset, offset + expected)))


@pytest.mark.parametrize("offset", [0, 7])
def test_lifecycle_block_consumes_no_iterations(tmp_path, offset):
    session = Session(make_config(), MagicMock(), "system", "task", str(tmp_path))
    session._turn_start_offset = offset
    session._lifecycle_hook_block_reason = "fixture block"
    result = session.run()
    assert result.finish_reason == "hook_block"
    assert result.turns == 0


def test_empty_iteration_allowance_is_zero(tmp_path):
    session = Session(make_config(max_turns=0), MagicMock(), "system", "task", str(tmp_path))
    session._turn_start_offset = 7
    result = session.run()
    assert result.finish_reason == "max_turns"
    assert result.turns == 0


def test_preflight_stop_counts_entered_work_without_a_model_call(tmp_path):
    session = Session(
        make_config(context_size=100, context_fill_ratio=0.5),
        MagicMock(), "system", "task", str(tmp_path),
    )
    session._turn_start_offset = 7
    session.context.add_user("x" * 1000)
    with patch.object(session, "_get_server_ctx", return_value=0):
        result = session.run()
    assert result.finish_reason == "context_full"
    assert result.turns == 1
    session.client.chat.assert_not_called()


def test_driver_records_local_units_and_aggregates_new_work(tmp_path):
    import json
    from llm_solver._shared.telemetry_paths import trace_path
    from llm_solver.harness.loop import solve_task

    (tmp_path / "prompt.txt").write_text("Complete the task")
    client = MagicMock()
    client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
    response = TurnResult(
        content="done", tool_calls=[], finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    run = Session.run

    def resumed_labels(session):
        session._turn_start_offset = 7
        return run(session)

    with (
        patch.object(Session, "run", resumed_labels),
        patch.object(Session, "_get_server_ctx", return_value=0),
        patch.object(Session, "_chat_with_retry", return_value=response),
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness._loop.driver.write_run_metrics") as metrics,
    ):
        assert solve_task(tmp_path, make_config(max_sessions=1), client)
    events = [json.loads(line) for line in trace_path(tmp_path).read_text().splitlines()]
    terminal = next(event for event in events if event["event"] == "session_end")
    assert terminal["turns"] == 1
    assert terminal["turns_unit"] == "loop_iterations"
    assert terminal["turns_scope"] == "session_invocation"
    assert terminal["turn_start_offset"] == 7
    recorded = metrics.call_args.args[1]
    assert recorded["total_turns"] == 1
    assert recorded["total_turns_unit"] == "loop_iterations"
    assert recorded["total_turns_scope"] == "solve_task_invocation"
