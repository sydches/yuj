"""Acceptance tests for issue #6 checkpoint/rewind exploration collapse."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._session_setup import build_context_manager
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness.checkpoint_rewind import (
    CheckpointBoundaryError,
    capture_context_checkpoint,
    rewind_context,
)
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.context_contract import build_context_contract
from scripts.llm_solver.harness.context_strategies import (
    list_context_modes,
    resolve_context_class,
)
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.state_writer import project, write_state_from_trace
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.replay_client import ReplayClient
from scripts.llm_solver.server.types import TurnResult, ToolCall, Usage


def _assistant_message(content: str | None, tool_calls: list[ToolCall]) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in tool_calls
        ]
    return message


class ScriptedClient:
    def __init__(self, turns: list[TurnResult]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict], turn: int = 0):
        self.requests.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "turn": turn,
        })
        return self.turns[len(self.requests) - 1]

    def build_assistant_message(
        self, content: str | None, tool_calls: list[ToolCall]
    ) -> dict:
        return _assistant_message(content, tool_calls)

    def query_server_context(self) -> int:
        return 0


def _turn(
    content: str | None,
    *tool_calls: ToolCall,
    finish_reason: str | None = None,
) -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def _scripted_turns() -> list[TurnResult]:
    return [
        _turn(
            "Mark a safe boundary, then inspect the probe.",
            ToolCall(
                id="checkpoint-0",
                name="checkpoint",
                arguments={"goal": "Find the relevant implementation fact."},
            ),
            ToolCall(
                id="read-0",
                name="read",
                arguments={"path": "probe.txt"},
            ),
        ),
        _turn(
            "Use one write as exploration evidence.",
            ToolCall(
                id="write-1",
                name="write",
                arguments={
                    "path": "target.txt",
                    "content": "exploration-only filesystem change\n",
                },
            ),
        ),
        _turn(
            "Collapse the exploration branch.",
            ToolCall(
                id="rewind-2",
                name="rewind",
                arguments={"report": "The probe establishes the retained finding."},
            ),
        ),
        _turn("Continue from the retained finding.", finish_reason="stop"),
    ]


def _write_replay_transcript(
    path: Path, requests: list[dict], turns: list[TurnResult]
) -> None:
    blocks: list[str] = []
    for number, (request, result) in enumerate(zip(requests, turns), start=1):
        blocks.append(f"=== turn {number:03d} input ===")
        blocks.append(json.dumps({
            "messages": request["messages"],
            "tools": request["tools"],
        }))
        message = _assistant_message(result.content, result.tool_calls)
        blocks.append(f"=== turn {number:03d} output ===")
        blocks.append(json.dumps({
            "choices": [{
                "message": message,
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
        }))
    path.write_text("\n".join(blocks) + "\n")


def _run_scripted_session(
    task_dir: Path,
    trace_path: Path,
    state_path: Path,
    client,
):
    cfg = make_config(
        max_turns=6,
        tools_checkpoint_enabled=True,
        tools_schema_validation="reject",
        state_writer_enabled=True,
        min_turns_before_context=0,
        turn_snapshots_enabled=False,
    )
    trace_path.touch()
    with trace_path.open("a+") as trace_file:
        session = Session(
            cfg,
            client,
            "SYSTEM",
            "TASK",
            str(task_dir),
            context_manager=FullTranscript(),
            trace_file=trace_file,
            trace_path=trace_path,
            state_path=state_path,
        )
        session._server_ctx_synced = True
        result = session.run()
    return result


def test_tools_share_one_default_off_config_and_profile_gate(tmp_path: Path) -> None:
    default_cfg = load_config()
    assert default_cfg.tools_checkpoint_enabled is False

    overlay = tmp_path / "checkpoint.toml"
    overlay.write_text("[tools]\ncheckpoint_enabled = true\n")
    enabled_cfg = load_config(user_config=overlay)
    assert enabled_cfg.tools_checkpoint_enabled is True

    invalid_overlay = tmp_path / "checkpoint-invalid.toml"
    invalid_overlay.write_text('[tools]\ncheckpoint_enabled = "yes"\n')
    with pytest.raises(ValueError, match="checkpoint_enabled must be a boolean"):
        load_config(user_config=invalid_overlay)

    client = SimpleNamespace(
        profile=SimpleNamespace(max_tools=8, simplify_schemas=False)
    )
    disabled_names = {
        schema["function"]["name"]
        for schema in apply_profile_to_schemas(
            get_tool_schemas("minimal"), default_cfg, client
        )
    }
    enabled_names = {
        schema["function"]["name"]
        for schema in apply_profile_to_schemas(
            get_tool_schemas("minimal"), enabled_cfg, client
        )
    }
    assert {"checkpoint", "rewind"}.isdisjoint(disabled_names)
    assert {"checkpoint", "rewind"} <= enabled_names
    assert build_context_contract(FullTranscript, enabled_cfg)[
        "checkpoint_rewind"
    ] == {
        "enabled": True,
        "tools": ["checkpoint", "rewind"],
        "checkpoint_boundary": "complete_tool_call_turn",
        "rewind_report_role": "user",
        "raw_trace": "append_only",
        "filesystem_restore": False,
    }


def test_rewind_without_checkpoint_is_a_typed_error_envelope(
    tmp_path: Path,
) -> None:
    cfg = make_config(tools_checkpoint_enabled=True)
    session = Session(
        cfg,
        ScriptedClient([]),
        "SYSTEM",
        "TASK",
        str(tmp_path),
        context_manager=FullTranscript(),
    )

    result = dispatch(
        "rewind",
        {"report": "There is no active mark."},
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=session._tool_registry,
    )

    assert '<tool_result tool_name="rewind" status="error"' in result
    assert 'error_kind="no_active_checkpoint"' in result
    assert "requires an active checkpoint" in result


def test_checkpoint_boundary_refuses_unpaired_tool_call_and_touches_no_files(
    tmp_path: Path,
) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(b"unchanged bytes\n")
    context = FullTranscript()
    context.add_system("SYSTEM")
    context.add_user("TASK")
    context.add_assistant(_assistant_message(
        "Inspect.",
        [ToolCall(id="read-0", name="read", arguments={"path": "task.txt"})],
    ))

    with pytest.raises(CheckpointBoundaryError, match="before tool result"):
        capture_context_checkpoint(context, goal="Inspect safely.", turn=0)

    context.add_tool_result("read-0", "unchanged bytes")
    checkpoint = capture_context_checkpoint(
        context, goal="Inspect safely.", turn=0
    )
    context.add_assistant(_assistant_message("Explore.", []))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    rewind_context(context, checkpoint, "The file remained unchanged.")

    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before
    assert context.get_messages()[-1]["role"] == "user"
    assert context.get_messages()[-1]["content"].startswith(
        '<rewind-report goal="Inspect safely.">'
    )


@pytest.mark.parametrize("mode", list_context_modes())
def test_every_context_strategy_rebuilds_without_exploration(
    mode: str, tmp_path: Path
) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir()
    (solver_dir / "state.json").write_text(json.dumps({
        "state": {}, "trace": [], "gates": [], "evidence": [], "inference": []
    }))
    cfg = make_config(min_turns_before_context=0)
    context = build_context_manager(
        resolve_context_class(mode), cfg, tmp_path, "TASK", 0, None
    )
    assert context is not None
    context.add_system("SYSTEM")
    context.add_user("TASK")
    context.add_assistant(_assistant_message(
        "Keep this turn.",
        [ToolCall(id="read-0", name="read", arguments={"path": "kept.txt"})],
    ))
    context.add_tool_result("read-0", "retained-result", tool_name="read")
    checkpoint = capture_context_checkpoint(
        context, goal="Keep the finding.", turn=0
    )
    context.add_assistant(_assistant_message(
        "Discard this turn.",
        [ToolCall(id="read-1", name="read", arguments={"path": "discard.txt"})],
    ))
    context.add_tool_result(
        "read-1", "EXPLORATION_SECRET", tool_name="read"
    )

    rewind_context(context, checkpoint, "Only the retained finding remains.")
    rendered = json.dumps(context.get_messages(), sort_keys=True)

    assert "EXPLORATION_SECRET" not in rendered
    assert "rewind-report" in rendered
    assert "Only the retained finding remains." in rendered


def test_runtime_trace_projection_filesystem_and_replay_contract(
    tmp_path: Path,
) -> None:
    source_task = tmp_path / "source-task"
    source_task.mkdir()
    (source_task / "probe.txt").write_text("probe fact\n")
    (source_task / "target.txt").write_text("before\n")
    source_trace = tmp_path / "source.trace.jsonl"
    source_state = source_task / ".solver" / "state.json"
    turns = _scripted_turns()
    source_client = ScriptedClient(turns)

    source_result = _run_scripted_session(
        source_task, source_trace, source_state, source_client
    )

    assert source_result.done is True
    assert (source_task / "target.txt").read_text() == (
        "exploration-only filesystem change\n"
    )
    final_messages = source_client.requests[-1]["messages"]
    assert "exploration-only filesystem change" not in json.dumps(final_messages)
    assert final_messages[-1] == {
        "role": "user",
        "content": (
            '<rewind-report goal="Find the relevant implementation fact.">\n'
            "The probe establishes the retained finding.\n"
            "</rewind-report>"
        ),
    }
    kept_assistant = final_messages[2]
    kept_ids = {call["id"] for call in kept_assistant["tool_calls"]}
    kept_results = {
        message["tool_call_id"]
        for message in final_messages[3:5]
        if message["role"] == "tool"
    }
    assert kept_ids == kept_results == {"checkpoint-0", "read-0"}

    raw_events = [
        json.loads(line)
        for line in source_trace.read_text().splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "tool_call"
        and event.get("tool_name") == "write"
        for event in raw_events
    )
    rewind_event = next(
        event for event in raw_events if event.get("event") == "rewind"
    )
    assert rewind_event["from_turn"] == 2
    assert rewind_event["to_turn"] == 0
    assert rewind_event["report_chars"] == len(
        "The probe establishes the retained finding."
    )

    # Session-level exit diagnostics append raw audit rows outside the live
    # in-memory mirror. Rebuild through the canonical artifact writer, then
    # require the on-disk artifact to equal the pure projection.
    write_state_from_trace(
        source_trace, source_state, max_result_chars=20_000
    )
    projected = json.loads(source_state.read_text())
    assert projected == project(raw_events, max_result_chars=20_000)
    projected_actions = [item["action"] for item in projected["trace"]]
    assert any(action.startswith("checkpoint(") for action in projected_actions)
    assert any(action.startswith("read(") for action in projected_actions)
    assert not any(action.startswith("write(") for action in projected_actions)
    assert not any(action.startswith("rewind(") for action in projected_actions)
    assert projected["meta"]["event_count"] == len(raw_events)
    assert projected["meta"]["projected_event_count"] < len(raw_events)
    assert projected["state"]["last_rewind"] == {
        "session_number": 0,
        "from_turn": 2,
        "to_turn": 0,
        "report_chars": len("The probe establishes the retained finding."),
    }

    transcript = tmp_path / "source.log"
    _write_replay_transcript(transcript, source_client.requests, turns)
    replay_task = tmp_path / "replay-task"
    replay_task.mkdir()
    (replay_task / "probe.txt").write_text("probe fact\n")
    (replay_task / "target.txt").write_text("before\n")
    replay_client = ReplayClient(transcript)
    replay_requests: list[list[dict]] = []
    replay_chat = replay_client.chat

    def capture_replay_request(messages, tools, turn=0):
        replay_requests.append(copy.deepcopy(messages))
        return replay_chat(messages, tools, turn)

    replay_client.chat = capture_replay_request
    replay_result = _run_scripted_session(
        replay_task,
        tmp_path / "replay.trace.jsonl",
        replay_task / ".solver" / "state.json",
        replay_client,
    )

    assert replay_result.done is True
    assert replay_requests == [
        request["messages"] for request in source_client.requests
    ]
