import json
from pathlib import Path

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.resume import (
    build_resumed_messages,
    parse_resume_transcript,
)
from scripts.llm_solver.harness.loop import build_resume_prompt_from_trace


def test_build_resume_prompt_from_trace_uses_last_session_actions(tmp_path: Path):
    trace_path = tmp_path / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_app.py'",
            "result_summary": "1 failed",
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "max_turns",
            "turns": 7,
            "total_prompt_tokens": 777,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    cfg = load_config()
    prompt = build_resume_prompt_from_trace(
        trace_path,
        cfg,
        task_description="Fix the failing application test.",
    )

    assert prompt is not None
    assert "Previous session ended after 7 turns: max_turns." in prompt
    assert "bash(cmd='pytest -q tests/test_app.py')" in prompt
    assert "Fix the failing application test." in prompt


def test_transparent_resume_preserves_balanced_boundary_exactly() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"app.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "source"},
    ]

    resumed = build_resumed_messages(messages, None, None)

    assert resumed == messages
    assert resumed is not messages
    assert resumed[-1]["role"] == "tool"


def test_transparent_resume_drops_incomplete_assistant_turn(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": messages})
        + "\n=== turn 001 output ===\n"
        + '{"choices":['
    )

    prior, incomplete = parse_resume_transcript(transcript)

    assert incomplete is None
    assert build_resumed_messages(prior, incomplete, None) == messages


def test_transparent_resume_rejects_completed_unbalanced_turn() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    completed = {
        "role": "assistant",
        "content": "I will inspect it.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": '{"path":"app.py"}',
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="before the next assistant response"):
        build_resumed_messages(messages, completed, None)


def test_empty_explicit_resume_message_is_rejected() -> None:
    with pytest.raises(ValueError, match="message is empty"):
        build_resumed_messages(
            [{"role": "user", "content": "task"}], None, "  \n"
        )
