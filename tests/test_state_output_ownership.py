"""Automatic state output must not adopt task files or overwrite temp names."""
import json
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness._loop.state_projection import validate_state_destination
from scripts.llm_solver.harness.state_writer import write_state_from_events, write_state_from_trace


@pytest.mark.parametrize("writer", ["events", "trace"])
@pytest.mark.parametrize("link", [False, True])
def test_state_write_preserves_existing_temporary_name(tmp_path, writer, link):
    state = tmp_path / "state.json"
    prior_temp = tmp_path / "state.json.tmp"
    target = tmp_path / "task.txt"
    target.write_text("task-owned")
    if link:
        prior_temp.symlink_to(target)
    else:
        prior_temp.write_text("task-owned")
    if writer == "events":
        write_state_from_events([], state, max_result_chars=20000)
    else:
        trace = tmp_path / "trace.jsonl"
        trace.write_text("")
        write_state_from_trace(trace, state, max_result_chars=20000)
    assert json.loads(state.read_text())["meta"]["event_count"] == 0
    assert prior_temp.read_text() == target.read_text() == "task-owned"
    assert list(tmp_path.glob("state.json.*.tmp")) == []


@pytest.mark.parametrize("kind", ["file", "file_link", "directory_link"])
def test_solve_refuses_unowned_state_before_generation(tmp_path, kind):
    from scripts.llm_solver.harness.loop import solve_task
    task = tmp_path / "task"
    task.mkdir()
    (task / "prompt.txt").write_text("Inspect the project.")
    directory = task / ".solver"
    outside = tmp_path / "existing"
    outside.mkdir()
    if kind == "directory_link":
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir()
    state = directory / "state.json"
    original = outside / "notes.json"
    original.write_text('{"task": "keep me"}')
    if kind == "file_link":
        state.symlink_to(original)
    else:
        state.write_bytes(original.read_bytes())
    before = state.read_bytes()
    client = MagicMock()
    with pytest.raises(ValueError, match="Refusing to replace existing state output"):
        solve_task(task, make_config(max_sessions=1), client)
    assert state.read_bytes() == original.read_bytes() == before
    client.chat.assert_not_called()


@pytest.mark.parametrize("tamper", [False, True])
def test_prior_state_requires_actual_trace_prefix_not_schema(tmp_path, tamper):
    trace, state = tmp_path / "trace.jsonl", tmp_path / "state.json"
    events = [dict(event="tool_call", session_number=1, turn_number=0,
                   tool_name="read", args_summary="path='notes'", result_summary="observed")]
    cfg = make_config()
    write_state_from_events(events, state, max_result_chars=cfg.max_output_chars,
                            think_keep_turns=cfg.tools_think_keep_turns)
    trace.write_text("\n".join(map(json.dumps, events + [dict(event="session_end")])))
    if tamper:
        data = json.loads(state.read_text())
        data["state"]["current_attempt"] = "not from this trace"
        state.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Refusing"):
            validate_state_destination(state, trace, cfg)
    else:
        validate_state_destination(state, trace, cfg)
        write_state_from_events(events + [dict(event="session_end")], state,
                                max_result_chars=cfg.max_output_chars)
        assert json.loads(state.read_text())["meta"]["event_count"] == 2


def test_solve_can_refresh_its_prior_projection(tmp_path):
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.server.types import TurnResult, Usage

    (tmp_path / "prompt.txt").write_text("Inspect the project.")
    client = MagicMock()
    client.chat.return_value = TurnResult(content="done", tool_calls=[], finish_reason="stop",
                                         usage=Usage(prompt_tokens=10, completion_tokens=2))
    client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
    cfg = make_config(max_sessions=1)
    for _ in range(2):
        assert solve_task(tmp_path, cfg, client, artifacts_dir=tmp_path)
    assert client.chat.call_count == 2
    assert json.loads((tmp_path / ".solver/state.json").read_text())["meta"]["event_count"] > 0
